# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:05:07 2023

@author: Genio
"""

import matplotlib.pyplot as plt
from   cycler     import cycler
from   matplotlib import cm
import numpy as np


# set defaults
plt.rcParams.update(plt.rcParamsDefault)

SMALL_SIZE  = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

#  fonts
plt.rc('font',size=SMALL_SIZE)

# title
plt.rc('axes'  ,titlesize=MEDIUM_SIZE)
plt.rc('figure',titlesize=BIGGER_SIZE)

# xy-labells
plt.rc('axes',labelsize=MEDIUM_SIZE)

# xy-ticks
plt.rc('xtick',labelsize=SMALL_SIZE)
plt.rc('ytick',labelsize=SMALL_SIZE)

# legend
plt.rc('legend',fontsize =SMALL_SIZE)
plt.rc('legend',facecolor='white')
plt.rc('legend',framealpha=0.9)

# lines
plt.rc('lines',linewidth=1.5)

# grid
plt.rc('axes',grid=True)

plt.rc('axes',edgecolor='gray')

# pixel in inches
px2inch = 1/plt.rcParams['figure.dpi']


styles = [
          'Solarize_Light2',#0
          '_classic_test_patch',#1    
          '_mpl-gallery',#2
          '_mpl-gallery-nogrid',#3
          'bmh',#4
          'classic',#5
          'dark_background',#6
          'fast',#7
          'fivethirtyeight',#8
          'ggplot',#9
          'grayscale',#10
          'seaborn',#11
          'seaborn-bright',#12
          'seaborn-colorblind',#13
          'seaborn-dark',#14
          'seaborn-dark-palette',#15
          'seaborn-darkgrid',#16
          'seaborn-deep',#17
          'seaborn-muted',#18
          'seaborn-notebook',#19
          'seaborn-paper',#20
          'seaborn-pastel',#21
          'seaborn-poster',#22
          'seaborn-talk',#23
          'seaborn-ticks',#24
          'seaborn-white',#25
          'seaborn-whitegrid',#26
          'tableau-colorblind10',#27
          ]


# style
#########
lines_colors   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

default_cycler = cycler('color', lines_colors)
plt.style.use(styles[0])
plt.rc('figure', facecolor='white')
plt.rc('axes'  , prop_cycle=default_cycler)


#figure sizes
###############
size_vi_vp   = (800*px2inch,800*px2inch)
size_regre   = (1600*px2inch,800*px2inch)
size_thermal = (700*px2inch,900*px2inch)


def utils_plot_vi_vp_Tcell_const(Pv_reff,Gpv,Tcell,Mat_P,Mat_V,Mat_I,Vec_Pmp,Vec_Vmp,Vec_Imp):
    
         fig, ax = plt.subplots(2,1, constrained_layout=True,figsize=size_vi_vp)

         axis_title    = f'$T_{{Cell}}$ = {Tcell}ºC'
         legend_labels = [f'$G_{{PV}}={G}$ $W/m^2$' for G in Gpv ]
        
         fig.suptitle(Pv_reff)
         ax[0].set_title(axis_title);ax[0].grid(True);ax[0].set_box_aspect(0.5)
         ax[0].plot(Mat_V,Mat_I,zorder=1)
         ax[0].scatter(Vec_Vmp,Vec_Imp,zorder=2,color='white',edgecolors='gray')
         ax[0].set_xlim(0,np.max(Mat_V)+1  )
         ax[0].set_ylim(0,np.max(Mat_I)+0.75);ax[0].set_ylabel('$Current$ $(A)$')
         for i,label in enumerate(legend_labels):
                ax[0].text(1,Mat_I[0,i],label,va='top')
                
         ax[1].set_title(axis_title);ax[1].set_box_aspect(0.5);ax[1].grid(True)
         ax[1].plot(Mat_V,Mat_P,label=legend_labels,zorder=1)
         ax[1].scatter(Vec_Vmp,Vec_Pmp,zorder=2,color='white',edgecolors='gray')
         ax[1].set_xlim(0,np.max(Mat_V)+1 );ax[1].set_xlabel('$Voltage$ $(V)$')
         ax[1].set_ylim(0,np.max(Mat_P)+25);ax[1].set_ylabel('$Power$ $(W)$')
         ax[1].legend(loc="upper left")
         for x,y in zip(Vec_Vmp,Vec_Pmp):
                ax[1].text(x,y-25,f'{round(y)} W',va='top',ha='center')
        
         fig.show()

def utils_plot_vi_vp_Gpv_const(Pv_reff,Gpv,Tcell,Mat_P,Mat_V,Mat_I,Vec_Pmp,Vec_Vmp,Vec_Imp):
    
         fig, ax = plt.subplots(2,1, constrained_layout=True, figsize=size_vi_vp)

         axis_title    = f'$G_{{PV}}$ = {Gpv} $W/m^2$'
         legend_labels = [f'$T_{{Cell}}$ = {T:d} ºC' for T in Tcell ]
       
         fig.suptitle(Pv_reff)
         ax[0].set_title(axis_title);ax[0].grid(True);ax[0].set_box_aspect(0.5)
         ax[0].plot(Mat_V,Mat_I,label=legend_labels,zorder=1)
         ax[0].scatter(Vec_Vmp,Vec_Imp,zorder=2,color='white',edgecolors='gray')
         ax[0].set_xlim(0,np.max(Mat_V)+1  )
         ax[0].set_ylim(0,np.max(Mat_I)+0.75);ax[0].set_ylabel('$Current$ $(A)$')
         ax[0].legend(loc="center left")

         ax[1].set_title(axis_title);ax[1].set_box_aspect(0.5);ax[1].grid(True)
         ax[1].plot(Mat_V,Mat_P,label=legend_labels,zorder=1)
         ax[1].scatter(Vec_Vmp,Vec_Pmp,zorder=2,color='white',edgecolors='gray')
         ax[1].set_xlim(0,np.max(Mat_V)+1 );ax[1].set_xlabel('$Voltage$ $(V)$')
         ax[1].set_ylim(0,np.max(Mat_P)+25);ax[1].set_ylabel('$Power$ $(W)$')
         ax[1].legend(loc="upper left")
        
         fig.show()

def utils_plot_regression(Pv_reff,X,Y,Z,ZR):
    
          fig,ax=plt.subplots(1,2,subplot_kw={"projection":"3d"},
                                  constrained_layout=True,
                                  figsize=size_regre)
    
          C     = Y
          COLOR = plt.cm.jet( (C-C.min())/float((C-C.min()).max()) )
          
          fig.suptitle(Pv_reff)
         
          azimut = -120
          elevat =  30
          ax[0].set_title('Surface $P_{MP}$  and  Regression Scatter')
          ax[0].plot_surface(X,Y,Z,alpha=0.75,cmap=cm.jet,facecolors=COLOR)
          ax[0].scatter(X,Y,ZR,color='green',edgecolors='white',alpha=1,lw=0.5)
        
          ax[0].set_xlim(0,np.max(X))
          ax[0].set_ylim(np.min(Y),np.max(Y))
          ax[0].set_xlabel('$G_{PV} $ $(W/m^2)$')
          ax[0].set_ylabel('$T_{Amb}$ $(ºC)$')
          ax[0].set_zlabel('$P_{MP} $ $(W)$')
          ax[0].view_init(elev=elevat, azim=azimut)
          
          Error = np.abs(Z-ZR)
          ax[1].set_title('Regression Error')
          ax[1].plot_surface(X,Y,Error, cmap=cm.jet)
          ax[1].set_xlabel('$G_{PV}$ $(W/m^2)$')
          ax[1].set_ylabel('$T_{Amb}$ $(ºC)$')
          ax[1].set_zlabel('$|P_{MP}-Regression|$  $(W)$')
          ax[1].view_init(elev=elevat, azim=azimut)
          ax[1].set_zlim(0,np.max(Error))
          
          fig.show()
        
        
def utils_plot_Pmax_Tcell_Gpv(Pv_reff,Gpv,Tamb,Matrix_Gpv,Matrix_Tcell):
          
          fig, ax = plt.subplots(2,1, constrained_layout=True,figsize=size_thermal)
          
          labels_vector = [f'$T_{{Amb}}$ = {tamb}ºC' for tamb in Tamb]

          fig.suptitle(Pv_reff)
          
          ax[0].plot(Gpv,Matrix_Gpv,label=labels_vector)
          ax[1].plot(Gpv,Matrix_Tcell)
          
          pbaspect = 0.75
          ax[0].legend(loc="upper left")
          ax[0].set_box_aspect(pbaspect)
          ax[0].set_xlim(0,1000)
          ax[0].set_ylabel('$P_{MP}$  $(W)$')
          
          ax[1].set_box_aspect(pbaspect)
          ax[1].set_xlim(0,1000)
          ax[1].set_xlabel('$G_{PV}$ $(W/m^2)$')
          ax[1].set_ylabel('$T_{Cell}$ $(ºC)$')
          
          fig.show()
   
