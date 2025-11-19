from abc import ABC, abstractmethod
from pathlib import Path
from .parameters import (MLMCParameters,
                         export_dict_MLMCParameters)
from alm_model.nested.nested_framework import NestedFrameworkParameters
from alm_model.nested import (StructuralConstantsWE1,
                              StructuralConstantsWE1Var,
                              StructuralConstantsWERVar)
from .theoretical_results import (BaseMLMCTheoreticalResults,
                                  ML2RTheoreticalResults,
                                  MLMCTheoreticalResults)
import typing
import numpy as np
import numpy.typing as npt
import math
import alm_model.utils as utils
import scipy.optimize as optim


class GeneralMLMCCalibrator(ABC):
    
    @abstractmethod
    def calibrate_R(self : typing.Self,
                    epsilon : float) -> int:
        pass
    
    @abstractmethod
    def calibrate_K(self : typing.Self,
                    epsilon : float,
                    R : int) -> float:
        pass
    
    @abstractmethod
    def calibrate_q(self : typing.Self,
                    R : int,
                    K : float
                    ) -> np.ndarray:
        pass
    
    @abstractmethod
    def calibrate_J(self : typing.Self,
                    epsilon : float,
                    R : int,
                    K : float,
                    q : np.ndarray
                    ) -> float:
        pass
    
    def calibrate(self : typing.Self,
                  epsilon : float) -> MLMCParameters:
        
        R = self.calibrate_R(epsilon)
        K = self.calibrate_K(epsilon, R)
        q = self.calibrate_q(R, K)
        J = self.calibrate_J(epsilon, R, K, q)
        
        return MLMCParameters(J, q, K, R)
    
    def calibration_csv(self : typing.Self,
                        epsilons : list[float],
                        output_file : Path) -> None:
        res_dict = {}
        for i in range(len(epsilons)):
            
            params = self.calibrate(epsilons[i])
            res_dict[epsilons[i]] = params

        export_dict_MLMCParameters(res_dict, output_file)   

class NestedMC1Calibrator(GeneralMLMCCalibrator):
    
    def __init__(self : typing.Self,
                 structural_constants : StructuralConstantsWE1,
                 params_nested : NestedFrameworkParameters):
        
        if structural_constants.alpha != 1:
            raise ValueError("alpha must be equal to 1 to use this estimator. (got {})".format(structural_constants.alpha))
        
        self.structural_constants = structural_constants
        self.params_nested = params_nested
        
    def calibrate_R(self : typing.Self,
                    epsilon : float
                    ) -> int:
        
        return 1
    
    def objective_function_optim_K(self: typing.Self,
                                   epsilon : float,
                                   K : float) -> float:
        tau = self.params_nested.tau 
        c1 = self.structural_constants.c1
        return (K + tau) / (epsilon**2 - (c1/K)**2)
    
    def get_K_min(self : typing.Self,
              epsilon : float) -> float:
        
        c1 = self.structural_constants.c1
        return abs(c1) / epsilon
    
    def calibrate_K(self : typing.Self,
                    epsilon : float,
                    R : int) -> int:
        
        c1 = self.structural_constants.c1
        tau = self.params_nested.tau

        #When tau is null the formula is specific
        if tau == 0:

            K_optim = abs(c1) * math.sqrt(3) / epsilon
        
        else:

            #Store to compute the constant only once
            tc = abs(c1) / tau

            #Each case implement its own formula
            if epsilon < tc:

                K_optim = (abs(c1) / epsilon) * 2 * math.cos(math.acos(epsilon * tau / abs(c1)) / 3)
            
            elif epsilon > tc:
                #Store to compute the constant only once
                sq = math.sqrt(tau**2 - (abs(c1) / epsilon)**2)

                K_optim = math.pow(abs(c1) / epsilon, 2/3) * (math.pow(tau + sq, 1/3) + math.pow(tau - sq, 1/3))
            
            else:
                
                K_optim = 3 * abs(c1)
        
        #Check wether we take upper or lower
        K_floor = math.floor(K_optim)
        val_floor = self.objective_function_optim_K(epsilon, K_floor)
        K_ceil = math.ceil(K_optim)
        val_ceil = self.objective_function_optim_K(epsilon, K_ceil)
        K_min = self.get_K_min(epsilon)
        
        if K_floor > K_min and val_floor < val_ceil:
            
            return int(K_floor)
        
        return int(K_ceil)
        
        
    def calibrate_q(self : typing.Self,
                    R : int,
                    K : float
                    ) -> np.ndarray:
        
        return np.array([1.])
        
    def calibrate_J(self : typing.Self,
                    epsilon : float,
                    R : int,
                    K : float,
                    q : np.ndarray) -> float:
        
        c1 = self.structural_constants.c1
        K = math.ceil(K)
        num = self.structural_constants.sigma_bar_1**2
        den = epsilon**2 - (c1 / K)**2

        return  num / den
    
    
class BaseOptimizedMLMCCalibrator(GeneralMLMCCalibrator):
    
    def __init__(self : typing.Self,
                 structural_constants : StructuralConstantsWE1Var,
                 params_nested : NestedFrameworkParameters,
                 mlmc_theoretical_results : BaseMLMCTheoreticalResults):
        
        self.structural_constants = structural_constants
        self.params_nested = params_nested
        self.mlmc_theoretical_results = mlmc_theoretical_results
    
    def get_weights(self : typing.Self, R : int) -> npt.NDArray[np.float64]:
        
        return self.mlmc_theoretical_results.get_weights(R)
    
    def unit_variance(self : typing.Self,
                      R : int,
                      q : np.ndarray,
                      K : float) -> float:

        res_acc = 0
        for r in range(1, R+1):
            
            sigma_bar = self.mlmc_theoretical_results.sigma_bar(r, K, R)
            res_acc += sigma_bar**2 / q[r-1]

        return res_acc

    def calibrate_J(self : typing.Self,
                    epsilon : float,
                    R : int,
                    K : float,
                    q : np.ndarray) -> float:

        mlmc_params = MLMCParameters(J=1, q=q, K=K, R=R)
        theo_bias = self.mlmc_theoretical_results.theoretical_bias(mlmc_params)
        return self.unit_variance(R, q, K) / (epsilon**2 - theo_bias**2)
    
    def calibrate_q(self : typing.Self,
                    R : float,
                    K : int) -> npt.NDArray[np.float64]:
        
        q = np.empty(R, dtype=np.float64)
        for r in range(1, R+1):
            
            Kr = math.ceil(K) * 2**(r-1)
            gamma_r = self.mlmc_theoretical_results.gamma(Kr)
            sigma_bar = self.mlmc_theoretical_results.sigma_bar(r, K, R)
            q[r-1] = sigma_bar / math.sqrt(gamma_r)

        mu = np.sum(q)
        return q / mu
    
    def effort_star(self : typing.Self,
                    K : float,
                    R : int) -> npt.NDArray[np.float64]:
                
        acc = 0

        for i in range(R):
            Kr = math.ceil(K) * 2**(i)
            sigma_bar_2 = self.mlmc_theoretical_results.sigma_bar(i+1, K, R)**2
            gamma = self.mlmc_theoretical_results.gamma(Kr)
            acc += math.sqrt(sigma_bar_2) * math.sqrt(gamma)

        return acc**2

    @abstractmethod
    def get_K_min(self : typing.Self,
                  epsilon : float,
                  R : int) -> float:
        pass

    def calibrate_K(self : typing.Self,
                epsilon : float,
                R : int) -> float:
        
        def f_to_min(x):
            mlmc_params = MLMCParameters(J=1, q=np.array([1.0]), K=x, R=R)
            theo_bias = self.mlmc_theoretical_results.theoretical_bias(mlmc_params)
            return self.effort_star(mlmc_params.K, mlmc_params.R) / (epsilon**2 - theo_bias**2)

        K_min = self.get_K_min(epsilon, R)
        return utils.unbounded_minimize_integer(f_to_min, K_min, K_min)[0]
    
    def get_R_max(self : typing.Self,
                  epsilon : float,
                  K_bar : int = 1) -> int:
        
        pass
    
    def calibrate_R(self : typing.Self,
                    epsilon : float) -> int:
        
        def f_to_min(x):
            R = math.ceil(x)
            K = self.calibrate_K(epsilon, R)
            q = self.calibrate_q(R, K)
            J = self.calibrate_J(epsilon, R, K, q)
            mlmc_params = MLMCParameters(J=J, q=q, K=K, R=R)
            phi = self.effort_star(mlmc_params.K, mlmc_params.R)
            theo_bias = self.mlmc_theoretical_results.theoretical_bias(mlmc_params)
            return phi / (epsilon**2 - theo_bias**2)
        
        R_min = 1
        R_max = self.get_R_max(epsilon)
        
        min_nb_levels = 0
        min_val = np.inf
        for r in range(R_min, R_max):
            
            cost = f_to_min(r)
            if cost < min_val:
                min_val = cost
                min_nb_levels = r
                
        return min_nb_levels
    
    def calibrate_cost(self : typing.Self,
                       cost : int) -> MLMCParameters:
        
        def f_to_root(epsilon):
            
            parms = self.calibrate(epsilon)
            return self.mlmc_theoretical_results.cost(parms) - cost
        
        eps_min = 1
        eps_max = 1 / (10e10)
        epsilon_star = optim.bisect(f_to_root, eps_min, eps_max)
        
        return self.calibrate(epsilon_star), epsilon_star

class OptimizedML2RCalibrator(BaseOptimizedMLMCCalibrator):
    
    structural_constants : StructuralConstantsWERVar
    
    def __init__(self : typing.Self,
                 structural_constants : StructuralConstantsWERVar,
                 params_nested : NestedFrameworkParameters,
                 mlmc_theoretical_results : ML2RTheoreticalResults):
        super().__init__(structural_constants, params_nested, mlmc_theoretical_results)
    
    def get_K_min(self : typing.Self,
                  epsilon : float,
                  R : int) -> float:
        
        c_R = self.structural_constants.c1 * self.structural_constants.a**(R-1)
        alpha = self.structural_constants.alpha
        
        res = c_R**(1/(R*alpha)) / (epsilon**(1/(alpha * R)) * 2**((R-1) / (2*alpha)))
        return math.floor(res) + 1
    
    def get_R_max(self : typing.Self,
                  epsilon : float,
                  K_bar : int = 1) -> int:
        
        c_tilde = self.structural_constants.a
        alpha = self.structural_constants.alpha
        
        A = 0.5 + math.log2(c_tilde**(1/alpha)/ K_bar)
        B = (2/alpha) * math.log2(math.sqrt(1 + 4*alpha) / epsilon)
        return math.ceil(A + math.sqrt(A**2 + B))

class OptimizedMLMCCalibrator(BaseOptimizedMLMCCalibrator):
    
    def __init__(self : typing.Self,
                 structural_constants : StructuralConstantsWE1Var,
                 params_nested : NestedFrameworkParameters,
                 mlmc_theoretical_results : MLMCTheoreticalResults):
        super().__init__(structural_constants, params_nested, mlmc_theoretical_results)

    def get_R_max(self : typing.Self,
                  epsilon : float,
                  K_bar : int = 1) -> int:
        
        c1 = self.structural_constants.c1
        alpha = self.structural_constants.alpha
        return math.ceil(1
                         + math.log2(c1**(1/alpha) / K_bar)
                         + math.log2(math.sqrt(1 + 2*alpha) / epsilon) / alpha)

    def get_K_min(self : typing.Self,
                  epsilon : float,
                  R : int) -> int:
        
        c1 = self.structural_constants.c1
        alpha = self.structural_constants.alpha
        K_tilde = abs(c1)**(1/alpha) / (epsilon**(1/alpha) * 2**(R-1))
        
        return math.floor(K_tilde) + 1

class ClosedML2RCalibrator(OptimizedML2RCalibrator):
    
    def __init__(self,
                 structural_constants,
                 params_nested,
                 mlmc_theoretical_results,
                 K_bar):
        self.K_bar = K_bar
        super().__init__(structural_constants, params_nested, mlmc_theoretical_results)
    
    def calibrate_J(self, epsilon, R, K, q):
        alpha = self.structural_constants.alpha
        beta = self.structural_constants.beta
        M = 1 + 1 / (2 * alpha * R)
        V1 = self.structural_constants.V1
        sigma_bar_1 = self.mlmc_theoretical_results.sigma_bar(1, K, R)
        weights = self.get_weights(R)
        
        acc = sigma_bar_1**2 / q[0]
        
        for r in range(2, R+1):
            
            acc += V1 * weights[r-1]**2 / (q[r-1] * K**(beta) * 2**((r-1)*beta))
        
        return M * acc / epsilon**2
    
    def calibrate_q(self, R, K):
        
        q = np.empty(R)
        
        V1 = self.structural_constants.V1
        sigma_bar_1 = self.mlmc_theoretical_results.sigma_bar(1, K, R)
        weights = self.get_weights(R)
        K_bar = self.K_bar
        beta = self.structural_constants.beta
        q[0] = sigma_bar_1
        
        for r in range(2, R+1):
            
            q[r-1] = math.sqrt(V1) * abs(weights[r-1]) / (K_bar**(0.5*beta) * 2**(0.5*(1+beta)*(r-1)))
        
        sum_q = q.sum()
        return q / sum_q
    
    def K_plus(self : typing.Self,
                epsilon : float,
                R : int) -> float:
        
        c_tilde = self.structural_constants.a
        alpha = self.structural_constants.alpha
        
        return (1 + 2 * alpha * R)**(1/(2*alpha*R)) * c_tilde**(1/alpha) / (epsilon**(1/(alpha*R)) * 2**(0.5*(R-1)))
    
    def calibrate_K(self, epsilon, R):
        K_plus = self.K_plus(epsilon, R)
        K_bar = self.K_bar
        
        return K_bar * math.ceil(K_plus / K_bar)
    
    def calibrate_R(self, epsilon):
        
        c_tilde = self.structural_constants.a
        K_bar = self.K_bar
        alpha = self.structural_constants.alpha
        
        A = 0.5 + math.log2(c_tilde**(1/alpha)/ K_bar)
        B = (2/alpha) * math.log2(math.sqrt(1 + 4*alpha) / epsilon)
        return math.ceil(A + math.sqrt(A**2 + B))

class ClosedMLMCCalibrator(OptimizedMLMCCalibrator):
    
    def __init__(self,
                 structural_constants,
                 params_nested,
                 mlmc_theoretical_results,
                 K_bar):
        self.K_bar = K_bar
        super().__init__(structural_constants, params_nested, mlmc_theoretical_results)
    
    def calibrate_J(self, epsilon, R, K, q):
        alpha = self.structural_constants.alpha
        beta = self.structural_constants.beta
        M = 1 + 1 / (2 * alpha)
        V1 = self.structural_constants.V1
        sigma_bar_1 = self.mlmc_theoretical_results.sigma_bar(1, K, R)
        weights = self.get_weights(R)
        
        acc = sigma_bar_1**2 / q[0]
        
        for r in range(2, R+1):
            
            acc += V1 * weights[r-1]**2 / (q[r-1] * K**(beta) * 2**((r-1)*beta))
        
        return M * acc / epsilon**2
    
    def calibrate_q(self, R, K):
        
        q = np.empty(R)
        
        V1 = self.structural_constants.V1
        sigma_bar_1 = self.mlmc_theoretical_results.sigma_bar(1, K, R)
        weights = self.get_weights(R)
        K_bar = self.K_bar
        beta = self.structural_constants.beta
        q[0] = sigma_bar_1
        
        for r in range(2, R+1):
            
            q[r-1] = math.sqrt(V1) * abs(weights[r-1]) / (K_bar**(0.5*beta) * 2**(0.5*(1+beta)*(r-1)))
        
        sum_q = q.sum()
        return q / sum_q
    
    def K_plus(self : typing.Self,
                epsilon : float,
                R : int) -> float:
        
        c1 = self.structural_constants.c1
        alpha = self.structural_constants.alpha
        
        return (1 + 2 * alpha)**(1/(2*alpha)) * abs(c1)**(1/alpha) / (epsilon**(1/(alpha)) * 2**(R-1))
    
    def calibrate_K(self, epsilon, R):
        K_plus = self.K_plus(epsilon, R)
        K_bar = self.K_bar
        
        return K_bar * math.ceil(K_plus / K_bar)
    
    def calibrate_R(self, epsilon):
        
        c1 = self.structural_constants.c1
        K_bar = self.K_bar
        alpha = self.structural_constants.alpha
        
        return math.ceil(1
                         + math.log2(abs(c1)**(1/alpha) / K_bar)
                         + math.log2(math.sqrt(1 + 2*alpha) / epsilon) / alpha)
