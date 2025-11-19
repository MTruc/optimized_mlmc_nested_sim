import typing
import json
from pathlib import Path

class StructuralConstantsWE1:

    def __init__(self : typing.Self,
                 alpha : float,
                 c1 : float,
                 sigma_bar_1 : float):
        
        self.alpha = alpha
        self.c1 = c1
        self.sigma_bar_1 = sigma_bar_1
        
class StructuralConstantsWE1Var(StructuralConstantsWE1):
    
    def __init__(self : typing.Self,
                 alpha : float,
                 c1 : float,
                 sigma_bar_1 : float,
                 beta : float,
                 V1 : float):
        self.beta = beta
        self.V1 = V1
        super().__init__(alpha, c1, sigma_bar_1)
        
class StructuralConstantsWERVar(StructuralConstantsWE1Var):
    
    def __init__(self : typing.Self,
                 alpha : float,
                 c1 : float,
                 sigma_bar_1 : float,
                 beta : float,
                 V1 : float,
                 a : float):
        self.a = a
        super().__init__(alpha, c1, sigma_bar_1, beta, V1)
        
def load_structural_constants(file_path : Path) -> StructuralConstantsWERVar:
    
    with open(file_path, "r") as file:

        params_dict = json.loads(file.read())
    
    c1 = params_dict["c1"]
    sigma_bar_1 = params_dict["sigma_bar_1"]
    a = params_dict["a"]
    alpha = params_dict["alpha"]
    V1 = params_dict["V1"]
    beta = params_dict["beta"]
    a = params_dict["a"]
    #K_bar = params_dict["K_bar"]

    return StructuralConstantsWERVar(alpha, c1, sigma_bar_1, beta, V1, a)