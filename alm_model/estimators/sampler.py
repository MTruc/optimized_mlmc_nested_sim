from abc import ABC, abstractmethod
from alm_model.nested.nested_framework import (NestedFramework,
                                               NestedFrameworkParameters)
from .mlmc_estimators import (sample_EK_gpu,
                              sample_delta_EK_anti_gpu,
                              calibrate_max_J_per_batch_nested_mc,
                              calibrate_max_J_per_batch_mlmc)
from .parameters import (MLMCParameters,
                         NestedMCParameters)
from .theoretical_results import MLMCTheoreticalResults
from cupyx.profiler import benchmark
import typing
import numpy as np
import cupy as cp
import math
import scipy.optimize as optim
import alm_model.utils as utils

class GeneralMLMCSample(ABC):
    
    @abstractmethod
    def evaluate(self : typing.Self,
                 u : float) -> float:
        pass
    
    @abstractmethod
    def root_find(self : typing.Self,
                  level : float) -> float:
        pass

class NestedMCSample():
    
    def __init__(self : typing.Self,
                 nested_framework_params : NestedFrameworkParameters,
                 EK_array : cp.ndarray | np.ndarray):
        
        self.nested_framework_params = nested_framework_params
        self.xp = cp.get_array_module(EK_array)
        self.EK_array = self.xp.sort(EK_array)
    
    def evaluate(self : typing.Self,
                 u : float) -> float:
        
        YK_array = self.nested_framework_params.payoff(self.EK_array, u)
        return self.xp.mean(YK_array)
    
    def root_find(self : typing.Self,
                  level : float) -> float:
        
        J = len(self.EK_array)
        idx = math.ceil(J * level)
        
        return self.EK_array[idx]

class MLMCSample():
    
    def __init__(self : typing.Self,
                 nested_framework_params : NestedFrameworkParameters,
                 EK_first_level : cp.ndarray | np.ndarray,
                 EK_upper_levels : list[tuple[cp.ndarray]] | list[tuple[np.ndarray]],
                 min_EK : float,
                 max_EK : float,
                 weights : np.ndarray):
        
        self.nested_framework_params = nested_framework_params
        self.xp = cp.get_array_module(EK_first_level)
        self.EK_first_level = EK_first_level
        self.EK_upper_levels = EK_upper_levels
        self.R = 1 + len(self.EK_upper_levels)
        self.min_EK = min_EK
        self.max_EK = max_EK
        self.weights = weights
    
    def evaluate(self : typing.Self,
                 u : float) -> float:
            #Evaluate function
            
        res = 0
        R = 1 + len(self.EK_upper_levels)

        #First level
        YK = self.nested_framework_params.payoff(self.EK_first_level, u)
        res += cp.mean(YK)

        #Upper levels
        for r in range(2, R+1):
            EKf = self.EK_upper_levels[r-2][0]
            EKc_1 = self.EK_upper_levels[r-2][1]
            EKc_2 = self.EK_upper_levels[r-2][2]
            YKf = self.nested_framework_params.payoff(EKf, u)
            YKc_1 = self.nested_framework_params.payoff(EKc_1, u)
            YKc_2 = self.nested_framework_params.payoff(EKc_2, u)
            delta_YK = YKf - 0.5 * (YKc_1 + YKc_2)
            res += self.weights[r-1] * cp.mean(delta_YK)

        return res
    
    def root_find(self : typing.Self,
                  level : float) -> float:
        
        def f_to_root(u):

            return self.evaluate(u) - level
    
        return optim.bisect(f_to_root, self.min_EK, self.max_EK)

class GeneralMLMCSampler():
    
    def __init__(self : typing.Self,
                nested_framework : NestedFramework,
                rng,
                gpu_memory_margin = 0.9):
        
        self.nested_framework = nested_framework
        self.rng = rng #cp generator
        
        print("Diagnostic GPU memory usage.")
        #Get GPU total memory
        device = cp.cuda.Device(0)
        total_memory_bytes = device.mem_info[1]
        gpu_available_memory_in_MB = total_memory_bytes / (1024 ** 2)
        self.gpu_available_memory_in_MB = gpu_available_memory_in_MB
        
        #Measure memory per unit
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        J_diagnostic = 2000
        K_diagnostic = 40
        sample_EK_gpu(self.nested_framework, J_diagnostic, K_diagnostic, self.rng)
        cp.cuda.Device().synchronize()
        gpu_mem_peak_mb = mempool.total_bytes() / 1024**2
        gpu_memory_peak_in_MB_per_unit = gpu_mem_peak_mb / (J_diagnostic * K_diagnostic)
        self.gpu_memory_peak_in_MB_per_unit = gpu_memory_peak_in_MB_per_unit
        self.gpu_memory_margin = gpu_memory_margin
        
        print("Diagnostic Time GPU Usage")
        bench = benchmark(sample_EK_gpu, (self.nested_framework, J_diagnostic, K_diagnostic, self.rng), n_repeat=25)
        avg_time = np.mean(bench.gpu_times) * 1.25
        avg_time_unit = avg_time / (J_diagnostic * K_diagnostic)
        self.avg_time_unit = avg_time_unit
    
    
    @abstractmethod
    def generate_sample(self : typing.Self,
                        mlmc_param : MLMCParameters) -> GeneralMLMCSample:
        pass
    
class NestedMCSamplerGPU(GeneralMLMCSampler):
        
    def generate_sample(self : typing.Self,
                        mc_params : NestedMCParameters) -> NestedMCSample:
        
        total_sizes = int(math.ceil(mc_params.J))
        K = int(math.ceil(mc_params.K))
        max_J = calibrate_max_J_per_batch_nested_mc(self.gpu_available_memory_in_MB,
                                                    self.gpu_memory_peak_in_MB_per_unit,
                                                    K,
                                                    self.gpu_memory_margin)
        sizes = utils.compute_batch_sizes(int(math.ceil(mc_params.J)), max_J)
        EK_array = cp.empty(total_sizes, dtype=cp.float64)
        
        cursor = 0
        for i in range(len(sizes)):

            size = sizes[i]
            EK_array[cursor:cursor+size] = sample_EK_gpu(self.nested_framework, size, K, self.rng) 
            cursor += size
        
        return NestedMCSample(self.nested_framework.params_nested, EK_array)

class BaseMLMCSamplerGPU(GeneralMLMCSampler):
    
    def __init__(self : typing.Self,
                 nested_framework : NestedFrameworkParameters,
                 rng,
                 mlmc_theo_res : MLMCTheoreticalResults, #TODO fix les type hint avec factorisation
                 gpu_memory_margin=0.9):
        self.mlmc_theo_res = mlmc_theo_res
        super().__init__(nested_framework, rng, gpu_memory_margin)
    
    def generate_sample(self : typing.Self,
                    mc_params : NestedMCParameters) -> MLMCSample:
        
        K = int(math.ceil(mc_params.K))
        R = mc_params.R
        max_J = calibrate_max_J_per_batch_mlmc(
            self.gpu_available_memory_in_MB,
            self.gpu_memory_peak_in_MB_per_unit,
            K,
            R,
            self.gpu_memory_margin
        )
        J_levels = [int(math.ceil(mc_params.J * mc_params.q[r])) for r in range(R)]
        sizes_levels = utils.compute_batch_sizes_mlmc(J_levels, max_J)
        
        EKs_levels = []
        min_EKs = cp.inf
        max_EKs = -cp.inf

        #First level
        sizes = sizes_levels[0]
        EKs_levels.append(cp.empty(J_levels[0]))
        cursor = 0

        for i in range(len(sizes)):
            
            size = sizes[i]
            #Sample
            # print("size = {:.2e}, K = {:.2e}, expected time : {} ms".format(size,
            #                                                                     K,
            #                                                                     self.avg_time_unit * size * K * 1000))
            # with gpu_timer():
            EKs_levels[0][cursor:cursor+size] = sample_EK_gpu(self.nested_framework, size, K, self.rng)
            cursor += size

        min_level = cp.min(EKs_levels[0])
        
        if min_level < min_EKs:
            min_EKs = min_level
        
        max_level = cp.max(EKs_levels[0])

        if max_level > max_EKs:
            max_EKs = max_level

        #Upper levels
        Kf = K
        for r in range(2, R+1):

            EKs_levels.append((cp.empty(J_levels[r-1]), cp.empty(J_levels[r-1]), cp.empty(J_levels[r-1])))
            sizes = sizes_levels[r-1]
            Kf = 2 * Kf
            cursor = 0
            for i in range(len(sizes)):
            
                size = sizes[i]
                
                #Sample
                Ek, Ek_1, Ek_2 = sample_delta_EK_anti_gpu(self.nested_framework, size, Kf, self.rng)
                EKs_levels[r-1][0][cursor:cursor+size] = Ek
                EKs_levels[r-1][1][cursor:cursor+size] = Ek_1
                EKs_levels[r-1][2][cursor:cursor+size] = Ek_2
                cursor += size

            min_level = min(cp.min(EKs_levels[r-1][0]),
                            cp.min(EKs_levels[r-1][1]),
                            cp.min(EKs_levels[r-1][2]))
        
            if min_level < min_EKs:
                min_EKs = min_level
            
            min_level = max(cp.max(EKs_levels[r-1][0]),
                            cp.max(EKs_levels[r-1][1]),
                            cp.max(EKs_levels[r-1][2]))

            if max_level > max_EKs:
                max_EKs = max_level
                
        return MLMCSample(self.nested_framework.params_nested,
                          EKs_levels[0],
                          EKs_levels[1:],
                          min_EKs,
                          max_EKs,
                          weights=self.mlmc_theo_res.get_weights(R))   
    
class ML2RSamplerGPU(BaseMLMCSamplerGPU):
    
    def __init__(self, nested_framework, rng, ml2r_theo_res, gpu_memory_margin=0.9):
        super().__init__(nested_framework, rng, ml2r_theo_res, gpu_memory_margin)

class MLMCSamplerGPU(BaseMLMCSamplerGPU):
    
    def __init__(self, nested_framework, rng, mlmc_theo_res, gpu_memory_margin=0.9):
        super().__init__(nested_framework, rng, mlmc_theo_res, gpu_memory_margin)
