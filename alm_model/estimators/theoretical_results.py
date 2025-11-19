from abc import ABC, abstractmethod
from alm_model.nested.nested_framework import NestedFrameworkParameters
from alm_model.nested.structural_consts import (StructuralConstantsWE1,
                                                StructuralConstantsWE1Var,
                                                StructuralConstantsWERVar)
from .parameters import (NestedMCParameters,
                         MLMCParameters)
import typing
import math
import numpy as np
import numpy.typing as npt

class GeneralMLMCTheoreticalResults(ABC):
    
    def __init__(self : typing.Self,
                 params_nested : NestedFrameworkParameters):
        self.params_nested = params_nested
        super().__init__()
    
    @abstractmethod
    def theoretical_bias(self : typing.Self,
                         mlmc_params : MLMCParameters) -> float:
        pass
    
    @abstractmethod
    def sigma_bar(self : typing.Self,
                  r : int,
                  K : float,
                  R : int = 1) -> float:
        pass
    
    def theoretical_variance(self : typing.Self,
                             mlmc_params : MLMCParameters) -> float:
        
        R = mlmc_params.R
        J = mlmc_params.J
        q = mlmc_params.q
        K = mlmc_params.K
        
        acc_var = 0
        for r in range(1, R+1):
            
            Jr = math.ceil(J * q[r-1])
            acc_var += self.sigma_bar(r, K, R)**2 / Jr
        
        return acc_var
    
    def theoretical_mse(self : typing.Self,
                        mlmc_params : MLMCParameters) -> float:
        
        return self.theoretical_bias(mlmc_params)**2 + self.theoretical_variance(mlmc_params)
    
    def gamma(self : typing.Self,
              K : int) -> int:
        
        return self.params_nested.primary_cost + K * self.params_nested.secondary_cost
        
    def cost(self : typing.Self,
             mlmc_params : MLMCParameters) -> int:
        
        R = mlmc_params.R
        J = mlmc_params.J
        q = mlmc_params.q
        K = mlmc_params.K
        
        acc_cost = 0
        for r in range(1, R+1):
            
            Jr = math.ceil(J * q[r-1])
            Kr = math.ceil(K) * 2**(r-1)
            acc_cost += Jr * self.gamma(Kr)

        return acc_cost
    
class NestedMCTheoreticalResults(GeneralMLMCTheoreticalResults):
    
    def __init__(self : typing.Self,
                 structural_constants : StructuralConstantsWE1,
                 params_nested : NestedFrameworkParameters):
        
        self.structural_constants = structural_constants
        super().__init__(params_nested)
    
    def theoretical_bias(self : typing.Self,
                         nested_params : NestedMCParameters) -> float:
        
        alpha = self.structural_constants.alpha
        c1 = self.structural_constants.c1
        K = nested_params.K
        
        return c1 / math.ceil(K)**(alpha)
    
    def sigma_bar(self, r, K, R=1):
        
        if r != 1:
            assert ValueError("sigma_bar exists only for r == 1 in a NestedMC. (got {})".format(r))
        
        return self.structural_constants.sigma_bar_1

class BaseMLMCTheoreticalResults(GeneralMLMCTheoreticalResults):
    
    def __init__(self : typing.Self,
                 structural_constants : StructuralConstantsWE1Var,
                 params_nested : NestedFrameworkParameters):
        
        self.structural_constants = structural_constants
        super().__init__(params_nested)
    
    @abstractmethod
    def get_weights(self : typing.Self,
                    R : int) -> np.ndarray:
        
        pass
    
    def sigma_bar(self, r, K, R):
        
        weights = self.get_weights(R)
        beta = self.structural_constants.beta
        if r == 1:
        
            return self.structural_constants.sigma_bar_1
        
        elif r > 1:
            Kr = math.ceil(K) * 2**(r-1)
            return math.sqrt(self.structural_constants.V1) * abs(weights[r-1]) / Kr**(0.5*beta)


class ML2RTheoreticalResults(BaseMLMCTheoreticalResults):
    
    structural_constants : StructuralConstantsWERVar
    
    def __init__(self : typing.Self,
                 structural_constants : StructuralConstantsWERVar,
                 params_nested : NestedFrameworkParameters):
        
        super().__init__(structural_constants, params_nested)
    
    def theoretical_bias(self : typing.Self,
                         mlmc_params : MLMCParameters) -> float:
        
        K = mlmc_params.K
        R = mlmc_params.R
        alpha = self.structural_constants.alpha

        c_R = self.structural_constants.c1 * self.structural_constants.a**(R-1)
        return ((-1)**(R-1) * c_R) / (math.ceil(K)**(alpha*R) * 2**(alpha*0.5*R*(R-1)))

    def get_weights(self : typing.Self, R : int) -> npt.NDArray[np.float64]:
        """Compute vector of ML2R weights.

        Args:
            self (typing.Self): Self.
            R (int): Number of levels.

        Returns:
            npt.NDArray[np.float64]: Weights.
        """

        alpha = self.structural_constants.alpha

        weights = np.empty(R, dtype=np.float64)
        w = np.empty(R, dtype=np.float64)

        for i in range(1, R+1):
            acc = 1.0

            for j in range(1, R+1):

                if j != i:
                    acc *= abs(1 - 2**(-alpha*(i-j)))

            w[i-1] = (-1)**(R-i) / acc
        
        for i in range(R):

            weights[i] = np.sum(w[i:])
        
        return weights

class MLMCTheoreticalResults(BaseMLMCTheoreticalResults):
    
    def __init__(self : typing.Self,
                 structural_constants : StructuralConstantsWE1Var,
                 params_nested : NestedFrameworkParameters):
        super().__init__(structural_constants, params_nested)
    
    def get_weights(self, R):
        return np.ones(R)
    
    def theoretical_bias(self : typing.Self,
                         mlmc_params : MLMCParameters) -> float:
        
        c1 = self.structural_constants.c1
        alpha = self.structural_constants.alpha
        R = mlmc_params.R
        K = mlmc_params.K
        
        return c1 / (math.ceil(K) * 2**(R - 1))**(alpha)
