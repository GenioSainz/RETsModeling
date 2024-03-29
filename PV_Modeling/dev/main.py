from PV_Modeling.src.pv_utils import read_template_pv, regression2json
from PV_Modeling.src.pv_class import PVmodule
import sys

json_name  = "data_regression.json"
path_write = f"PV_Modeling/dev/{json_name}"
arguments  = sys.argv[1:]
file_name  = arguments[0]

def main( file_name ):

    path_read  = f"/writing-api/files/temp/{ file_name }.xlsx"
    obj_array  = []

    for cols_range in ['K:K','L:L']:

        dataSheet = read_template_pv( path_read ,cols_range)
        pv        = PVmodule( dataSheet )
        pv.calc_data()

        obj_array.append(pv)

    regression2json( obj_array , path_write )

if __name__ == "__main__":
   main(  file_name )
   print( f'##{json_name}' )



   