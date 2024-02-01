# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:01:19 2023

@author: Genio
"""

import pandas as pd
import numpy  as np
import time

from scipy.optimize import fsolve
from scipy.stats    import mode
from scipy          import interpolate

from PV_Modeling.src.pv_utils import system_equations
from PV_Modeling.src.pv_utils import check_dataSheet

from PV_Modeling.src.pv_plots import utils_plot_vi_vp_Tcell_const
from PV_Modeling.src.pv_plots import utils_plot_vi_vp_Gpv_const
from PV_Modeling.src.pv_plots import utils_plot_regression
from PV_Modeling.src.pv_plots import utils_plot_Pmax_Tcell_Gpv

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

''' DataSheet

Reference  = {'Manufacturer': float ,
              'Series'      : float ,
              'Reference'   : float ,
              'CellType'    : float ,}

Dimensions = {'Ncells': float,
              'Width' : float,
              'Height': float,
              'Area'  : float,}
    
TempCoeff = {'k_Pmp': float,
             'k_Voc': float,
             'k_Isc': float}
    
NOCT = {'Pmp'  : float,
        'Vmp'  : float,
        'Imp'  : float,
        'Voc'  : float,
        'Isc'  : float,
        'Gpv'  : float,
        'Tamb' : float,
        'Tcell': float}
    
STC = {'Pmp'  : float,
       'Vmp'  : float,
       'Imp'  : float,
       'Voc'  : float,
       'Isc'  : float,
       'Gpv'  : float,
       'Tamb' : float,
       'Tcell': float}

DataSheet               = {}
DataSheet['Reference' ] = Reference
DataSheet['Dimensions'] = Dimensions
DataSheet['TempCoeff' ] = TempCoeff 
DataSheet['NOCT']       = NOCT
DataSheet['STC']        = STC

'''

class PVmodule:
    
      def __init__(self, DataSheet):
        
          self.DataSheet     = DataSheet
          self.Thermal_NOCT  = {}
          self.Circuit_STC   = {}
          self.Regression    = {}
          self.Thermal_const = {'k'    : 5.670e-8, # Stephan-Boltzmann constant
                                'e_b'  : 0.85,     # emissivity back
                                'e_f'  : 0.91,     # emissivity front
                                'e_g'  : 0.94,     # emissivity ground
                                'e_sky': 0.91,     # emissivity sky
                                'r_g'  : 0.10,}    # reflection glass
          
      def calc_data(self):
            '''Main method. If no errors are detected in the datashet, 
            the PV calculation methods are executed '''
            
            errors = check_dataSheet(self.DataSheet)
            
            if errors['check']:
                
                self.get_rad_con_coeff_NOCT()
                self.solve_Circuit_STC()
                self.get_Pmp_Gpv_Tamb()
                
            else:
                raise Exception('DataSheet ERROR',errors) 
                
            
      def __str__(self):
          '''Print class data'''
          
          print('{str:-^70}'.format(str='DataSheet') )
          print(pd.DataFrame(self.DataSheet), end='\n')
          
          print('{str:-^70}'.format(str='Thermal_const') )
          print(pd.DataFrame([self.Thermal_const]).T, end='\n')
          
          print('{str:-^70}'.format(str='Thermal_NOCT') )
          print(pd.DataFrame([self.Thermal_NOCT]).T, end='\n' )
          
          print('{str:-^70}'.format(str='Circuit_STC') )
          print(pd.DataFrame([self.Circuit_STC]).T, end='\n' )
          
          print('{str:-^70}'.format(str='Regression') )
          print(pd.DataFrame([self.Regression]).T, end='\n' )
         
          return ''
          
      def get_values(self,data,keys):
          
          return [data[key] for key in keys]
      
      def get_ref(self):
          '''Get manofacturer + module reference'''
          
          manufacturer = self.DataSheet['Reference']['Manufacturer']
          reference    = self.DataSheet['Reference']['Reference']
          return f'PV Module: {manufacturer} {reference}'
        
        
      def get_Vt(self,A,Tcell):
          '''Diode thermal voltage'''
          
          k  = 1.380649e-23
          q  = 1.60217663e-19
          return A*k*(Tcell+273)/q
      
        
      def get_rad_con_coeff_NOCT(self):
            '''Steady-state thermal Model in NOCT conditions:
               - Gpv:1000W/m2  Tamb:20ºC (293K)
               - Gpv*Area - p*Gpv*Area - Pmp - Qtotal = 0
               
               - Qrad = hrad*Area*(Tcell**4-Tamb**4)
               - Qcon = hcon*Area*(Tcell-Tamb)
               - hrad/hcon  radiative/convective coefficients
               - Qrad/Qcon  radiative/convective heat exchanges
               - Qtot = Qrad+Qconv  
                       
               WRITE: thermalNOCT
            '''
            
            k,e_b,e_f,e_g,e_sky,r_g = self.Thermal_const.values() 
            Tcell, Tamb, Gpv, Pmp   = self.get_values(self.DataSheet['NOCT'],['Tcell','Tamb','Gpv','Pmp'])
            Tcell, Tamb             = Tcell+273, Tamb+273
            Area                    = self.DataSheet['Dimensions']['Area']

            hrad_b = k/( (1-e_b)/e_b +1 + (1-e_g)/e_g)
            hrad_f = k/( (1-e_f)/e_f +1 + (1-e_sky)/e_sky)
            hrad   = hrad_f+hrad_b
            Qrad   = hrad*Area*(Tcell**4-Tamb**4)
             
            hcon  = ( Gpv*(1-r_g) - Pmp/Area - Qrad/Area )/( Tcell-Tamb )
            Qcon  = hcon*Area*(Tcell-Tamb)
            Qtot  = Qrad + Qcon
            
            self.Thermal_NOCT = {'hrad' :hrad,
                                 'hcon' :hcon,
                                 'Qrad' :Qrad,
                                 'Qcon' :Qcon,
                                 'Qtot' :Qtot}
            
            return self.Thermal_NOCT
        
        
      def solve_Circuit_STC(self):
          ''' Single-diode five-parameters model in STC conditions:
              The parameters are calculated by solving a system
              of non-linear equations
              
              - Rs : series resistance
              - Rsh: parallel (shunt) resistance
              - A  : diode quality (ideality) factor
              - Io : dark saturation current 
              - Iph: the photo-generated current
             
              WRITE: Circuit_STC
          '''
          
          Tcell,Isc,Voc,Vmp,Imp = self.get_values(self.DataSheet['STC'],['Tcell','Isc','Voc','Vmp','Imp'])
          ns                    = self.DataSheet['Dimensions']['Ncells']

          Nx         = 5
          Rs0        = np.linspace(0.1,1.5,Nx)
          Rsh0       = np.logspace(np.log10(1e2),np.log10(1e5),Nx)
          A0         = np.linspace(0.5,3,Nx)
          [X1,X2,X3] = np.meshgrid(Rs0,Rsh0,A0)
          X1, X2, X3 = X1.flatten(), X2.flatten(), X3.flatten()
          X,ier      = np.zeros((Nx**3,3)), np.zeros(Nx**3)

          for i,(x1,x2,x3) in enumerate(zip(X1,X2,X3)):
              
              # ier is an integer flag thet set to 1 if a solution was found
              X[i,:],_,ier[i],_ = fsolve(system_equations,
                                                         x0=[x1,x2,x3],
                                                         args=(Isc,Voc,Vmp,Imp,ns,Tcell),
                                                         full_output=True)   
          X        = X[ier==1,:]
          X[X<0  ] = np.nan
          X[X>1e6] = np.nan
          X        = X[~np.isnan(X).any(axis=1)]
          solution = mode(X,keepdims=False).mode
          
          Rs  = solution[0]
          Rsh = solution[1]
          A   = solution[2]

          Vt  = self.get_Vt(A,Tcell)
          I0  = (Isc-(Voc-Isc*Rs)/Rsh)*np.exp(-Voc/(ns*Vt))
          Iph = I0*np.exp(Voc/(ns*Vt))+Voc/Rsh
          
          funVI = lambda v,i:-i+Iph-I0*( np.exp((v+i*Rs)/(ns*Vt))-1)-(v+i*Rs)/Rsh
          
          self.Circuit_STC = {'Rs' : Rs,
                              'Rsh': Rsh,
                              'A'  : A,
                              'I0' : I0,
                              'Iph': Iph,
                              'sol_Isc': funVI(0,Isc),
                              'sol_Voc': funVI(Voc,0),
                              'solutions':X}
          return solution,X
      
        
      def get_vi_vp_curve(self,Gpv,Tcell):
           '''Return I=f(V),P=f(V) curve extrapolating the STC single-diode model to
              (Gpv,Tcell) conditions
              V: array(1,n) -> 0...Vmp...Voc
              I: array(1,n) -> 0...Imp...Isc 
              P: array(1,n) -> 0...Pmp...0
              Vmp: float
              Imp: float
              Pmp: float
           '''
           
           Isc_, Voc_   = self.get_values( self.DataSheet['STC']      , ['Isc','Voc'    ])
           ki  , kv     = self.get_values( self.DataSheet['TempCoeff'], ['k_Isc','k_Voc'])
           Rs  , Rsh, A = self.get_values( self.Circuit_STC           , ['Rs','Rsh','A' ])
           ns           = self.DataSheet['Dimensions']['Ncells']
           Vt           = self.get_Vt(A,Tcell)
           
           Isc_T = Isc_*( 1+ki/100*(Tcell-25) )
           Voc_T = Voc_*( 1+kv/100*(Tcell-25) )
            
           I0_T  = (Isc_T-(Voc_T-Isc_T*Rs)/Rsh)*np.exp(-Voc_T/(ns*Vt))
           Iph_T = I0_T*np.exp(Voc_T/(ns*Vt))+Voc_T/Rsh
            
           Iph   = Iph_T*Gpv/1000
           Isc   = Isc_T*Gpv/1000
            
           fun1   = lambda Voc_GT : Voc_GT - np.log((Iph*Rsh-Voc_GT)/(I0_T*Rsh))*ns*Vt
           x0     = Voc_
           Voc_GT = fsolve(fun1,x0)
            
           I0_GT  = (Isc-(Voc_GT-Isc*Rs)/Rsh)*np.exp(-Voc_GT/(ns*Vt))
           Iph_GT = I0_GT*np.exp(Voc_GT/(ns*Vt))+Voc_GT/Rsh

           N = 50
           V = np.linspace(0,Voc_GT,N)
           f = np.log(V+1)
           V = Voc_GT*f/np.max(f)
           
           I    = np.zeros_like(V)
           fun2 = lambda ipv:-ipv+Iph_GT-I0_GT*( np.exp((v+ipv*Rs)/(ns*Vt))-1)-(v+ipv*Rs)/Rsh
           x0   = Isc
           
           for i,v in enumerate(V):
         
               I[i] = fsolve(fun2,x0)
               x0   = I[i] 
               
           P    = V*I
           kmax = np.argmax(P)
           Pmp  = P[kmax][0]
           Vmp  = V[kmax][0]
           Imp  = I[kmax][0]
           
           return P,V,I,Pmp,Vmp,Imp


      def plot_vi_vp_Tcell_const(self, Tcell=25, Gpv=np.arange(200,1200,200) ):
        
          Pv_reff = self.get_ref()
          
          colum_vectors_P, colum_vectors_V, colum_vectors_I = [],[],[]
          
          Vec_Pmp,Vec_Vmp,Vec_Imp = [],[],[]
          
          for G in Gpv:
              
              P,V,I,Pmp,Vmp,Imp = self.get_vi_vp_curve(G,Tcell)
              
              Vec_Pmp.append(Pmp)
              Vec_Vmp.append(Vmp)
              Vec_Imp.append(Imp)
              
              colum_vectors_P.append(P)
              colum_vectors_V.append(V)
              colum_vectors_I.append(I)
              
          Mat_P = np.hstack( colum_vectors_P )
          Mat_V = np.hstack( colum_vectors_V )
          Mat_I = np.hstack( colum_vectors_I )
        
          utils_plot_vi_vp_Tcell_const(Pv_reff,Gpv,Tcell,Mat_P,Mat_V,Mat_I,Vec_Pmp,Vec_Vmp,Vec_Imp)
          
          
      def plot_vi_vp_Gpv_const(self, Tcell=np.arange(-20,60,20), Gpv=1000 ):
        
          Pv_reff = self.get_ref()
          
          colum_vectors_P, colum_vectors_V, colum_vectors_I = [],[],[]
          
          Vec_Pmp,Vec_Vmp,Vec_Imp = [],[],[]
          
          for T in Tcell:
              
              P,V,I,Pmp,Vmp,Imp = self.get_vi_vp_curve(Gpv,T)
              
              Vec_Pmp.append(Pmp)
              Vec_Vmp.append(Vmp)
              Vec_Imp.append(Imp)
              
              colum_vectors_P.append(P)
              colum_vectors_V.append(V)
              colum_vectors_I.append(I)
              
          Mat_P = np.hstack( colum_vectors_P )
          Mat_V = np.hstack( colum_vectors_V )
          Mat_I = np.hstack( colum_vectors_I )
        
          utils_plot_vi_vp_Gpv_const(Pv_reff,Gpv,Tcell,Mat_P,Mat_V,Mat_I,Vec_Pmp,Vec_Vmp,Vec_Imp)          

      
        
      def solve_cell_temp(self,Qtotal,Tamb,Area,h_rad,h_con):
            
           Tamb  = Tamb + 273
           fun   = lambda Tcell : Qtotal-Area*( h_rad*(Tcell**4-Tamb**4)+h_con*(Tcell-Tamb) )
           Tcell = fsolve(fun,Tamb)-273
          
           return Tcell[0]
       
        
      def get_Pmp_Tcell(self,Gpv,Tamb):
            
            Area   = self.DataSheet['Dimensions']['Area']
            r_g    = self.Thermal_const['r_g']
            h_rad  = self.Thermal_NOCT['hrad']
            h_con  = self.Thermal_NOCT['hcon']
            Tcell0 = Tamb
            
            for i in range(3):
                _,_,_,Pmp,_,_ = self.get_vi_vp_curve(Gpv,Tcell0)
                Qtotal        = Area*Gpv*(1-r_g)-Pmp
                Tcell1        = self.solve_cell_temp(Qtotal,Tamb,Area,h_rad,h_con)
                Tcell0        = Tcell1
                    
            return Pmp,Tcell1
     
    
      def get_Pmp_Gpv_Tamb(self,Gpv=[10,1000],Tamb=[-5,40]):
           
           start        = time.time()
           nPoints      = 15
           G_x          = np.linspace(Gpv[0] ,Gpv[1] ,nPoints)
           Tamb_y       = np.linspace(Tamb[0],Tamb[1],nPoints)
           [G_X,Tamb_Y] = np.meshgrid(G_x,Tamb_y)
          
           Pmp_Z   = np.zeros_like(G_X)
           Tcell_Z = np.zeros_like(G_X)
          
           for G_i in range(nPoints):  
               for T_j in range(nPoints):
                          
                    Gpv       = G_X[G_i,T_j]
                    Tamb      = Tamb_Y[G_i,T_j]
                    Pmp,Tcell = self.get_Pmp_Tcell(Gpv,Tamb)
                    
                    Pmp_Z[G_i,T_j]   = Pmp
                    Tcell_Z[G_i,T_j] = Tcell
           
           # numpy        : coeff
           # least square : coeff_LS
           coeff, coeff_LS, Pmp_ZR, mean_error= self.calc_regression(G_X,Tamb_Y,Pmp_Z)
        
           f_Z = interpolate.interp2d(G_x, Tamb_y, Pmp_Z,  kind='cubic')
           f_R = interpolate.interp2d(G_x, Tamb_y, Pmp_ZR, kind='cubic')
           
           
           self.Regression['Pout'    ] = 'P = Gpv(a0 + a1*Gpv*Tamb + a2*Tamb + a3*Gpv)'
           self.Regression['Gpv_lim' ] = [np.min(G_x)   , np.max(G_x)   ]                              
           self.Regression['Tamb_lim'] = [np.min(Tamb_y), np.max(Tamb_y)]
           self.Regression['nPoints' ] = nPoints
           
           keys = ["a0","a1","a2","a3"]
           self.Regression.update( {key: np.array(value)      for key,value in zip(keys,coeff)}    )
           self.Regression.update( {key+'LS': np.array(value) for key,value in zip(keys,coeff_LS)} )
           
           self.Regression["coeff_regre"] = {key: value for key,value in zip(keys,coeff)}
           self.Regression['coeff'      ] = coeff
           self.Regression['coeff_LS'   ] = coeff_LS
           self.Regression['Gpv_X'      ] = G_X
           self.Regression['Tamb_Y'     ] = Tamb_Y
           self.Regression['Pmp_Z'      ] = Pmp_Z
           self.Regression['Pmp_ZR'     ] = Pmp_ZR
           self.Regression['MeanError'  ] = mean_error
                 
           self.Regression['P_nominal'] = self.DataSheet['NOCT']['Pmp']
           self.Regression['P_regress'] = self.eval_regression(800,20)
           self.Regression['Z_interp' ] = f_Z(800,20)[0]
           self.Regression['ZR_interp'] = f_R(800,20)[0]
           
           lapse_time = time.time()-start
           print(f'Lapse time Regression {lapse_time} s')
               
           
      def calc_regression(self,X,Y,Z):
           ''' Regression in Kelvin'''
           Y  = Y+273
           XR = X.flatten()  # G_X
           YR = Y.flatten()  # Tamb_Y
           ZR = Z/X          # Pmp_Z/ G_X
           
           A = np.array( [XR*0+1, XR*YR,YR,XR] ).T
           B = ZR.flatten()
           
           ''' np.linalg.lstsq'''
           coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
           
           ZR    = X*( coeff[0] + coeff[1]*X*Y + coeff[2]*Y + coeff[3]*X )
           error = np.sum(np.abs(Z-ZR)) / Z.size
        
           ''' Least-squares LS
               Ax = b
               A'Ax = A'b
               x = (A'A)^-1 A'b
               M = (A'A)^-1 A'  Ppseudoinverse Matrix
               x = Mb
               x ~= coeff
           '''
           
           M = np.linalg.inv(A.T@ A) @ A.T
           x = M@B
           
           coeff_LS=x
           
           return coeff,coeff_LS,ZR,error
      
      def eval_regression(self,Gpv,Tamb):
          ''' Regression in Kelvin'''
          
          Tamb = Tamb + 273
          (a0,a1,a2,a3) = self.Regression['coeff']
          
          Pout = Gpv*( a0 + a1*Gpv*Tamb + a2*Tamb + a3*Gpv )
    
          return Pout
          
         
      def plot_regression(self):
         
          Pv_reff = self.get_ref()
          utils_plot_regression(Pv_reff,self.Regression['Gpv_X' ],
                                        self.Regression['Tamb_Y'],
                                        self.Regression['Pmp_Z' ],
                                        self.Regression['Pmp_ZR'])
      
      def plot_Pmax_Tcell_Gpv(self):
          
          Pv_reff = self.get_ref()
          
          nGpv = 5
          Tamb = [-20,-10,0,10,20,30,40]
          Gpv  = np.linspace(50,950,nGpv).reshape(nGpv,1)
          
          colum_vectors_Gpv   = []
          colum_vectors_Tcell = []
 
          for T_amb in Tamb:
             
              Pmp   = np.zeros_like(Gpv)
              Tcell = np.zeros_like(Gpv)
              
              for i,G_pv in enumerate(Gpv):
                  Pmp[i],Tcell[i] = self.get_Pmp_Tcell(G_pv,T_amb)
              
              colum_vectors_Gpv  .append(Pmp)
              colum_vectors_Tcell.append(Tcell)
              
          Matrix_Gpv   = np.hstack( colum_vectors_Gpv   )
          Matrix_Tcell = np.hstack( colum_vectors_Tcell )
          
          utils_plot_Pmax_Tcell_Gpv(Pv_reff,Gpv,Tamb,Matrix_Gpv,Matrix_Tcell)
          
      def get_regre_coeff(self,disp=False):
          
          
          if disp:
              reff = self.get_ref()
              print('',end='\n\n')
              print(f'Reference: {reff}',end='\n')
              print('##################'*3,end='\n')
              print('P = Gpv(a0 + a1*Gpv*Tamb + a2*Tamb + a3*Gpv)',end='\n')
              d = self.Regression['coeff_regre']
              for key in d:
                  print(f'{key}: {d[key]}',end='\n')
                  
              print('eval_regression(Gpv=800,Tamb=20)')
              print(f"Pnominal: {self.Regression['P_nominal']:0.3f} W")
              print(f"Pregresi: {self.Regression['P_regress']:0.3f} W")
              print('',end='\n\n')
          
          return self.Regression['coeff']
   
          
          