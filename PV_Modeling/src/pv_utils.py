# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:16:22 2023

@author: Genio
"""

import requests
import pandas as pd
import numpy  as np
import json


def read_excel_range( file_name, sheet_name, usecols, start_row, end_row):
    
    '''ExamPle:
       data = read_excel_range('path/filename.xlsx', 'sheet_name', 'K:F', 10, 12)'''
       
    start_row -= 1
    end_row   -= 1
    
    df = pd.read_excel(file_name,
                                sheet_name=sheet_name, 
                                usecols=usecols, 
                                skiprows=range(1, start_row), 
                                nrows=end_row - start_row + 1)
    return df.to_numpy().flatten()

# NEW METHOD OLD
###################
def read_template_pv(file_name,usecols):
    
    '''Read only pv dataSheet values for a single pv technology'''
    #       read_excel_range( file_name, sheet_name,      usecols, start_row, end_row)
    
    data0 = read_excel_range( file_name, '1.1_Config_TC', usecols, 10       , 12) # Reference
    data1 = read_excel_range( file_name, '1.1_Config_TC', usecols, 35       , 41) # Width-Length
    data2 = read_excel_range( file_name, '1.1_Config_TC', usecols, 43       , 56) # Electrical -Thermal  
    
    
    Reference  = {'CellType'    :data0[0],
                  'Reference'   :data0[1],
                  'Manufacturer':data0[2]}

    Dimensions = {'Ncells':data1[0],
                  'Width' :data1[1]*1e-3,
                  'Height':data1[2]*1e-3,
                  'Area'  :data1[1]*data1[2]*1e-6}

    STC =  {'Pmp'  :data2[0],
            'Vmp'  :data2[1],
            'Imp'  :data2[2],
            'Voc'  :data2[3],
            'Isc'  :data2[4],
            'Gpv'  :1000,
            'Tamb' :np.nan,
            'Tcell':25}
    
    NOCT = {'Pmp'  :data2[5],
            'Vmp'  :data2[6],
            'Imp'  :data2[7],
            'Voc'  :data2[8],
            'Isc'  :data2[9],
            'Gpv'  :800,
            'Tamb' :20,
            'Tcell':data2[13]}
    
    TempCoeff = {'k_Isc':data2[10],
                 'k_Voc':data2[11],
                 'k_Pmp':data2[12]}
    
    DataSheet               = {}
    DataSheet['STC']        = STC
    DataSheet['NOCT']       = NOCT
    DataSheet['TempCoeff' ] = TempCoeff 
    DataSheet['Dimensions'] = Dimensions
    DataSheet['Reference' ] = Reference

    return DataSheet

# READ METHOD OLD
###################
def read_dataSheet(file_name,row):
    
    excel_data = pd.read_excel(file_name,sheet_name='PV-Modules',skiprows=[0,1])
    module_row = excel_data.iloc[row-4]
    
    Reference  = {'Manufacturer':module_row[0],
                  'Series'      :module_row[1],
                  'Reference'   :module_row[2],
                  'CellType'    :module_row[3]}

    Dimensions = {'Ncells':module_row[7],
                  'Width' :module_row[8]*1e-3,
                  'Height':module_row[9]*1e-3,
                  'Area'  :module_row[8]*module_row[9]*1e-6}
    
    TempCoeff = {'k_Pmp':module_row[25],
                 'k_Voc':module_row[26],
                 'k_Isc':module_row[27]}
    
    NOCT = {'Pmp'  :module_row[30],
            'Vmp'  :module_row[31],
            'Imp'  :module_row[32],
            'Voc'  :module_row[33],
            'Isc'  :module_row[34],
            'Gpv'  :800,
            'Tamb' :20,
            'Tcell':module_row[23]}
    
    STC =  {'Pmp'  :module_row[35],
            'Vmp'  :module_row[36],
            'Imp'  :module_row[37],
            'Voc'  :module_row[38],
            'Isc'  :module_row[39],
            'Gpv'  :1000,
            'Tamb' :np.nan,
            'Tcell':25}
    
    DataSheet               = {}
    DataSheet['STC']        = STC
    DataSheet['NOCT']       = NOCT
    DataSheet['TempCoeff' ] = TempCoeff 
    DataSheet['Dimensions'] = Dimensions
    DataSheet['Reference' ] = Reference

    return DataSheet


def check_dataSheet(DataSheet):
    
    errors = {'check' : [],
              'STC'   : [],
              'NOCT'  : []}
    
    check  = [False,False]
    
    k_Isc  = DataSheet['TempCoeff']['k_Isc']
    k_Pmp  = DataSheet['TempCoeff']['k_Pmp']
    k_Voc  = DataSheet['TempCoeff']['k_Voc']

    bool_1 = np.sign(k_Isc)==1 and np.sign(k_Pmp)==-1 and np.sign(k_Voc)==-1
    bool_2 = DataSheet['STC']['Pmp']>DataSheet['NOCT']['Pmp']

    for i,key in enumerate(['STC','NOCT']):
      
        Pmp = DataSheet[key]['Pmp']
        Vmp = DataSheet[key]['Vmp'] 
        Imp = DataSheet[key]['Imp']
        Voc = DataSheet[key]['Voc']
        Isc = DataSheet[key]['Isc']

        bool_3 = Isc > Imp and np.abs(Isc-Imp)<Imp
        bool_4 = Vmp < Voc and np.abs(Voc-Vmp)<Vmp
        bool_5 = np.abs(Vmp*Imp-Pmp) < Pmp*0.05
        
        check[i] =  bool_1 and bool_2 and bool_3 and bool_4 and bool_5
        
        if check[i]:
           errors[key].append('data is correct')
        else :
        
           if not(bool_1):
              errors[key].append('Thermal Coeff error')
           if not(bool_2):
              errors[key].append('Max Power error')
           if not(bool_3):
              errors[key].append('Currents error')
           if not(bool_4):
              errors[key].append('Voltages error')
           if not(bool_5):
              errors[key].append('Voltages*Currents error')  
              
    errors['check'] = check[0] and check[1]
        
    return errors
        
def system_equations(X,Isc,Voc,Vmp,Imp,ns,Tcell):
    '''Isc,Voc,Vmp,Imp,ns,Tcell(K) in STC conditions'''

    Rs  = X[0]
    Rsh = X[1]
    A   = X[2]
    
    k  = 1.380649e-23
    q  = 1.60217663e-19
    Vt = A*k*(Tcell+273)/q
    
    C1 = ns*Vt*Rsh
    C2 = np.exp( (Vmp+Imp*Rs-Voc)/(ns*Vt) )
    C3 = np.exp( (Isc*Rs-Voc)/(ns*Vt) )
    C4 = Isc*Rsh-Voc+Isc*Rs
    
    # 12 FUN(1) Imp
    # 18 FUN(2) dP/dV
    # 19 FUN(3) dI/dV
    
    F    = [0,0,0]
    F[0] = -Imp+Isc-(Vmp+Imp*Rs-Isc*Rs)/Rsh-(Isc-(Voc-Isc*Rs)/Rsh)*C2     
    F[1] =  Imp+Vmp*( (-C4*C2/C1 -1/Rsh) / ( 1 + C4*C2/C1 + Rs/Rsh) )    
    F[2] =  1/Rsh +   (-C4*C3/C1 -1/Rsh) / ( 1 + C4*C3/C1 + Rs/Rsh)    
    
    return F
        
 
def get_TMY(lat,lon):
    
    # Example
    # lat = 45
    # lon = 8
    # tmy = get_TMY(lat,lon)
    # t   = tmy['Tamb']
    
    URL  = f'https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat={lat}&lon={lon}&outputformat=json'
    r    = requests.get(url = URL)
    data = r.json()['outputs']['tmy_hourly']
    
    N    = 8760
    Tamb = np.zeros(N)
    Gh   = np.zeros(N)
    Gb   = np.zeros(N)
    Gd   = np.zeros(N)
    
    for i,d in enumerate(data):
        
       Tamb[i] = d['T2m'  ]  # Air temperature.
       Gh[i]   = d['G(h)' ]  # Global horizontal irradiance.
       Gb[i]   = d['Gb(n)']  # Direct (beam) irradiance
       Gd[i]   = d['Gd(h)']  # Diffuse horizontal irradiance.
       
    return {'Tamb':Tamb,
            'Gh'  :Gh,
            'Gb'  :Gb,
            'Gd'  :Gd}


def regression2json(obj_array,json_path):
    
    ''' Example: 
        p0         = 
        obj_array  = [pv0,pv1]
        pv,col in zip( [p0,pv1],["K","L"] )
    '''
    dict_array = []
    
    for pv,col in zip( obj_array,["K","L"] ):
        
        data              = pv.Regression["coeff_regre"]
        data["reference"] = pv.DataSheet["Reference"]["Reference"]
        data["column"]    = col
        data["Pnominal"]  = pv.Regression['P_nominal']
        data["Pregress"]  = pv.Regression['P_regress']
        
        dict_array.append(data)
        pv.get_regre_coeff( disp=False )
        

    with open(json_path, "w") as outfile: 
         json.dump( dict_array , outfile,  indent=4)

    print()

