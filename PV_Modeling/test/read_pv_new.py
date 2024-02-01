# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:27:06 2024

@author: Genio
"""

from PV_Modeling.src.pv_utils import read_template_pv, regression2json
from PV_Modeling.src.pv_class import PVmodule

# READ METHOD NEW PROJECT TEMPLATE
#####################################

path_read  = r'PV_Modeling/docs/Final_Template_noBlocked.xlsx'
path_write = r'PV_Modeling/test/data_regression.json'
obj_array  = []

for cols_range in ['K:K','L:L']:

    dataSheet = read_template_pv( path_read ,cols_range)
    pv        = PVmodule( dataSheet )
    pv.calc_data()

    obj_array.append(pv)

regression2json( obj_array , path_write )
     
     
     