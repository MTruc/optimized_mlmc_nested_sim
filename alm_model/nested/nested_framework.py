from abc import ABC, abstractmethod
import numpy as np
import typing
import math
from dataclasses import dataclass
import cupy as cp
from pathlib import Path
import json

@dataclass
class NestedFrameworkParameters:

    payoff : callable
    primary_cost : int
    secondary_cost : int
    
    def __post_init__(self : typing.Self):

        self.tau = self.primary_cost / self.secondary_cost
        
    def set_costs(self, primary_cost, secondary_cost):
        
        self.primary_cost = primary_cost
        self.secondary_cost = secondary_cost
        self.tau = self.primary_cost / self.secondary_cost

def load_nested_framework_parameters(file_path : "Path") -> NestedFrameworkParameters:

    with open(file_path, "r") as file:

        params_dict = json.loads(file.read())
    
    primary_cost = params_dict["primary_cost"]
    secondary_cost = params_dict["secondary_cost"]

    if params_dict["payoff"] == "cdf_type_indicator":

        def payoff(x, eta):

            return np.where(x <= eta, 1.0, 0.0)
        
    if params_dict["payoff"] == "cdf_type_indicator_gpu":

        def payoff(x, eta):

            return cp.where(x <= eta, 1.0, 0.0)

    return NestedFrameworkParameters(payoff, primary_cost, secondary_cost)

class NestedFramework(ABC):
    
    def __init__(self : typing.Self,
                 params_nested : NestedFrameworkParameters):

        self.params_nested = params_nested
    
    @abstractmethod 
    def sample_F_matrix(self, K, J, rng):
        """ Sample a matrix of F(X_j, U_jk) where X_j are i.i.d and
        U_jk are i.i.d
        
        Args :
            - K (int) : A number of secondary samples
            - J (int) : A number of primary samples

        Return :
            - py torch tensor : (F(X_j, U_jk)) as a tensor  with J rows and K columns
        """

        pass
    
    @abstractmethod
    def sample_F_matrix_gpu(self, K, J, rng):
        pass

    def compute_EK(self, F_mat):

        return np.mean(F_mat, axis=1)
    
    def compute_EK_gpu(self : typing.Self,
                       F_mat : cp.ndarray) -> cp.ndarray:

        return cp.mean(F_mat, axis=1)
    
    def sample_EK(self, K, J, rng):

        F_mat = self.sample_F_matrix(K, J, rng)

        return self.compute_EK(F_mat)
    
    def compute_EK_anti(self, F_mat):

        Kf = F_mat.shape[1]
        Kc = int(Kf / 2)

        E_Kf = np.mean(F_mat, axis=1)
        E_1_Kf = np.mean(F_mat[:, :Kc], axis=1)
        E_2_Kf = np.mean(F_mat[:, Kc:], axis=1)

        return (E_Kf, E_1_Kf, E_2_Kf)
    
    def compute_EK_anti_gpu(self : typing.Self,
                            F_mat : cp.ndarray) -> tuple[cp.ndarray,...]:

        Kf = F_mat.shape[1]
        Kc = int(Kf / 2)

        E_Kf = cp.mean(F_mat, axis=1)
        E_1_Kf = cp.mean(F_mat[:, :Kc], axis=1)
        E_2_Kf = cp.mean(F_mat[:, Kc:], axis=1)

        return (E_Kf, E_1_Kf, E_2_Kf)
    
    def compute_EK_anti_general(self, F_mat, nb_level):

        E_K_levels = []
        Kf = F_mat.shape[1]
        for i in range(nb_level):

            curr_level_EK = []
            cursor = 0
            curr_K = int(Kf / 2**(i))
            for j in range(2**i):
                curr_level_EK.append(np.mean(F_mat[:, cursor:(cursor + curr_K)], axis = 1))
                cursor += curr_K
            E_K_levels.append(curr_level_EK)

        return E_K_levels
    
    def compute_EK_anti_general_gpu(self, F_mat, nb_level):
        
        E_K_levels = []
        Kf = F_mat.shape[1]
        for i in range(nb_level):

            curr_level_EK = []
            cursor = 0
            curr_K = int(Kf / 2**(i))
            for j in range(2**i):
                curr_level_EK.append(cp.mean(F_mat[:, cursor:(cursor + curr_K)], axis = 1))
                cursor += curr_K
            E_K_levels.append(curr_level_EK)

        return E_K_levels

    def compute_YK(self, e_k, eta):

        return self.params_nested.payoff(e_k, eta)
    
    def compute_YK_general(self, levels, eta):

        res = []
        for i in range(len(levels)):

            sub_res = []

            for j in range(len(levels[i])):
                e_k = levels[i][j]
                sub_res.append(self.params_nested.payoff(e_k, eta))
            res.append(sub_res)
        
        return res
    
    def compute_D1_both_types_gpu(self, F_mat, eta):

        E_K, E_1_K, E_2_K = self.compute_EK_anti_gpu(F_mat)
        Y_K = self.compute_YK(E_K, eta)
        Y_1_K = self.compute_YK(E_1_K, eta)
        Y_2_K = self.compute_YK(E_2_K, eta)

        non_anti = Y_K - Y_1_K
        anti = Y_K - 0.5 * (Y_1_K + Y_2_K)
        return anti, non_anti
    
    def compute_D2_gpu(self, F_mat, eta):
        
        nb_levels = 3
        levels = self.compute_EK_anti_general_gpu(F_mat, nb_levels)
        YKs = self.compute_YK_general(levels, eta)

        return 2 * YKs[0][0] - 3/2 * (YKs[1][0] + YKs[1][1]) + (1/4)*(YKs[2][0] + YKs[2][1] + YKs[2][2] + YKs[2][3])
    
    def sample_D1_both_types_D2_gpu(self, Kf, J, eta, rng):

        F_mat = self.sample_F_matrix_gpu(Kf, J, rng)
        D1_anti, D1_non_anti = self.compute_D1_both_types_gpu(F_mat, eta)
        D2 = self.compute_D2_gpu(F_mat, eta)

        return (D1_anti, D1_non_anti, D2)

    def set_costs(self, primary_cost, secondary_cost):

        self.params_nested.set_costs(primary_cost, secondary_cost)