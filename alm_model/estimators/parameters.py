import numpy as np
import typing
import pandas as pd
from pathlib import Path

class MLMCParameters:
    
    J : float
    q : np.ndarray
    K : float
    R : int
    
    def __init__(self : typing.Self,
                 J : float,
                 q : np.ndarray,
                 K : float,
                 R : int):
        self.J = J
        self.q = q
        self.K = K
        self.R = R
        
    def __str__(self):
        
        return "J = {:.2e}, q = {}, K = {:.2e}, R = {}".format(self.J, self.q, self.K, self.R)

class NestedMCParameters(MLMCParameters):
    
    def __init__(self, J, K):
        R = 1
        q = np.ndarray([1.0])
        super().__init__(J, q, K, R)
        
def export_dict_MLMCParameters(dict_params : dict[any, MLMCParameters],
                               output_file : Path) -> None:
    
    res = {}
    fields = list(MLMCParameters.__annotations__.keys())
    for f in fields:
        
        res[f] = {}
        
    max_len_q = 0
    #Get infos
    for key in dict_params.keys():
        
        params = dict_params[key]
        for f in fields:
            val = getattr(params, f)
            res[f][key] = val
            
            #Get longest length of q
            if f =="q" and hasattr(val, "__len__"):
                
                if max_len_q < len(val):
                    max_len_q = len(val)
    
    #Post treatment for q
    #Add q columns
    for i in range(1, max_len_q+1):
        res["q{}".format(i)] = {}
    
    #Populate q columns
    for key in dict_params.keys():
        
        for i in range(1, max_len_q+1):
            
            if i-1 >= len(res["q"][key]):
                res["q{}".format(i)][key] = 0
            else:
                res["q{}".format(i)][key] = res["q"][key][i-1]
    
    #Remove tmps q column
    if "q" in res.keys():
        del res["q"]
        
    return pd.DataFrame(res).to_csv(output_file, sep=";")
