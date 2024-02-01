# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:27:06 2024

@author: Genio
"""
import json
import matplotlib.pyplot as plt
from PV_Modeling.src.pv_utils import read_dataSheet
from PV_Modeling.src.pv_class import PVmodule

# READ METHOD OLD
###################
file_name = r'PV_Modeling/docs/dataSheets.xlsx'
excel_row = 30
dataSheet = read_dataSheet(file_name,excel_row)

pv = PVmodule( dataSheet )
pv.calc_data()
pv.get_regre_coeff( disp=True )
     
plots_ON = False
if plots_ON:
    plt.close('all')
    pv.plot_vi_vp_Tcell_const()
    pv.plot_vi_vp_Gpv_const(Tcell=[10,25,40,55,70])
    pv.plot_regression()
    pv.plot_Pmax_Tcell_Gpv()
    plt.show()   
     