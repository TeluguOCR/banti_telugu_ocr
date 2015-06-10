"""
Add new scalers to this file both as imports and in the dictionary!
"""
import ast
from .scaler_absolute import Absolute
from .scaler_relative import Relative

scalers_dict = {
    "Absolute":Absolute,
    "Relative":Relative
}

def ScalerFactory(params):
    if type(params) is str:
        with open(params) as scaler_fp:
            params = ast.literal_eval(scaler_fp.read())

    return scalers_dict[params["type"]](params)
