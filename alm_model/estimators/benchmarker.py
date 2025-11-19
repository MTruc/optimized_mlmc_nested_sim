import typing
import pandas as pd
import numpy as np
import math
import time
import cupy as cp
from abc import ABC, abstractmethod
from pathlib import Path
from .theoretical_results import GeneralMLMCTheoreticalResults
from .sampler import GeneralMLMCSampler
from .parameters import MLMCParameters

class MLMCBenchmarker(ABC):
    
    def __init__(self : typing.Self,
                 parameters_df : pd.DataFrame):
        
        self.parameters_df = parameters_df
        
        #Parse a list of MLMC parameter from dataframe
        self.mlmc_parameters_list = []
        self.indexes = []
        
        for idx, row in parameters_df.iterrows():
            
            self.indexes.append(idx)
            J = row["J"]
            K = row["K"]
            R = int(row["R"])
            
            #Need to parse q as an array
            list_qs = [i for i in row.index if isinstance(i, str) and i.startswith("q")]
            list_q_vals = [row[i] for i in list_qs]
            
            #Get end zeros
            to_remove_indices = []
            for i in range(len(list_q_vals) - 1, -1, -1):
                if list_q_vals[i] == 0:
                    to_remove_indices.append(i)
                else:
                    break
            q = np.array([v for i, v in enumerate(list_q_vals) if i not in to_remove_indices],
                         dtype=np.float64)
            
            self.mlmc_parameters_list.append(MLMCParameters(J, q, K, R))
            self.benchmark = None
            
    @abstractmethod
    def run_benchmark(self : typing.Self) -> None:
        pass
    
    def export_benchmark(self : typing.Self,
                            output_file : Path) -> None:
        
        self.benchmark.to_csv(output_file, sep=";")
        

class MLMCTheoreticalBenchmarker(MLMCBenchmarker):
    
    def __init__(self : typing.Self,
                 parameters_df : pd.DataFrame,
                 mlmc_theoretical_results : GeneralMLMCTheoreticalResults):
        self.mlmc_theoretical_results = mlmc_theoretical_results
        super().__init__(parameters_df)
        
    def run_benchmark(self : typing.Self) -> None:
        
        abs_theo_bias = {}
        theo_var = {}
        theo_rmse = {}
        cost = {}
        
        for i in range(len(self.mlmc_parameters_list)):
            
            mlmc_params = self.mlmc_parameters_list[i]
            idx = self.indexes[i]
            
            abs_theo_bias[idx] = abs(self.mlmc_theoretical_results.theoretical_bias(mlmc_params))
            theo_var[idx] = self.mlmc_theoretical_results.theoretical_variance(mlmc_params)
            theo_rmse[idx] = math.sqrt(self.mlmc_theoretical_results.theoretical_mse(mlmc_params))
            cost[idx] = self.mlmc_theoretical_results.cost(mlmc_params)
        
        df = self.parameters_df.copy(deep=True)
        df["abs_theo_bias"] = abs_theo_bias
        df["theo_var"] = theo_var
        df["theo_rmse"] = theo_rmse
        df["cost"] = cost
        self.benchmark = df
        
class MLMCQuantileCdfEmpiricalBenchmarker(MLMCBenchmarker):
    
    def __init__(self : typing.Self,
                 parameters_df : pd.DataFrame,
                 sampler : GeneralMLMCSampler,
                 n_samples : int,
                 u : float,
                 quantile_level : float,
                 target_cdf_eval : float,
                 target_quantile : float):
        self.n_samples = n_samples
        self.sampler = sampler
        self.u = u
        self.quantile_level = quantile_level
        self.target_cdf_eval = target_cdf_eval
        self.target_quantile = target_quantile
        super().__init__(parameters_df)
        
    def run_benchmark(self : typing.Self) -> None:
        
        abs_cdf_emp_bias = {}
        abs_quantile_emp_bias = {}
        cdf_emp_var = {}
        quantile_emp_var = {}
        cdf_emp_rmse_upper = {}
        cdf_emp_rmse_lower = {}
        cdf_emp_rmse = {}
        quantile_emp_rmse_upper = {}
        quantile_emp_rmse_lower = {}
        quantile_emp_rmse = {}
        avg_time = {}
        
        for i in range(len(self.mlmc_parameters_list)):

            mlmc_params = self.mlmc_parameters_list[i]
            total_unit = 0
            for k in range(mlmc_params.R):
                
                total_unit += math.ceil(mlmc_params.J * mlmc_params.q[k]) * math.ceil(mlmc_params.K) * 2**(k)
                
            estimated_time = self.sampler.avg_time_unit * self.n_samples * total_unit
            msg_params = "Sample parameters : {} (Estimated time : {:.2f} s)".format(mlmc_params, estimated_time)
            print(msg_params)
            print("-"*len(msg_params))
            idx = self.indexes[i]
            quantile_estimations = np.empty(self.n_samples)
            cdf_estimations = np.empty(self.n_samples)
            
            cp.cuda.Stream.null.synchronize()
            start_time = time.perf_counter()
            
            for k in range(self.n_samples):
                
                sample = self.sampler.generate_sample(mlmc_params)
                cdf_estimations[k] = sample.evaluate(self.u)
                quantile_estimations[k] = sample.root_find(self.quantile_level)
            
            cp.cuda.Stream.null.synchronize()
            end_time = time.perf_counter()
            dt = end_time - start_time
            avg_dt = dt / self.n_samples
            print("Took {:.2f} s (avg : {:.2f} s)".format(dt, avg_dt))
            abs_cdf_emp_bias[idx] = abs(np.mean(cdf_estimations - self.target_cdf_eval))
            abs_quantile_emp_bias[idx] = abs(np.mean(quantile_estimations - self.target_quantile))
            cdf_emp_var[idx] = np.var(cdf_estimations)
            quantile_emp_var[idx] = np.var(quantile_estimations)
            avg_time[idx] = avg_dt
            
            squared_error = np.pow(cdf_estimations - self.target_cdf_eval, 2)
            ci = 1.96 * np.std(squared_error) / math.sqrt(self.n_samples)
            cdf_emp_rmse[idx] = math.sqrt(np.mean(squared_error))
            cdf_emp_rmse_upper[idx] = math.sqrt(np.mean(squared_error) + ci)
            cdf_emp_rmse_lower[idx] = math.sqrt(max(np.mean(squared_error) - ci, 0))
            
            squared_error = np.pow(quantile_estimations - self.target_quantile, 2)
            ci = 1.96 * np.std(squared_error) / math.sqrt(self.n_samples)
            quantile_emp_rmse[idx] = math.sqrt(np.mean(squared_error))
            quantile_emp_rmse_lower[idx] = math.sqrt(max(np.mean(squared_error) - ci, 0))
            quantile_emp_rmse_upper[idx] = math.sqrt(np.mean(squared_error) + ci)
            
            print("Results : ")
            print("RMSE Cdf : {:.2e} (min : {:.2e}, max : {:.2e})".format(cdf_emp_rmse[idx],
                                                              cdf_emp_rmse_lower[idx],
                                                              cdf_emp_rmse_upper[idx]))
            print("RMSE quantile : {:.2e} (min : {:.2e}, max : {:.2e})".format(quantile_emp_rmse[idx],
                                                              quantile_emp_rmse_lower[idx],
                                                              quantile_emp_rmse_upper[idx]))
            print("-"*len(msg_params))
            print("")
        
        df = self.parameters_df.copy(deep=True)
        df["abs_cdf_emp_bias"] = abs_cdf_emp_bias
        df["abs_quantile_emp_bias"] = abs_quantile_emp_bias
        df["cdf_emp_var"] = cdf_emp_var
        df["quantile_emp_var"] = quantile_emp_var
        df["cdf_emp_rmse"] = cdf_emp_rmse
        df["cdf_emp_rmse_upper"] = cdf_emp_rmse_upper
        df["cdf_emp_rmse_lower"] = cdf_emp_rmse_lower
        df["quantile_emp_rmse"] = quantile_emp_rmse
        df["quantile_emp_rmse_upper"] = quantile_emp_rmse_upper
        df["quantile_emp_rmse_lower"] = quantile_emp_rmse_lower
        df["avg_time"] = avg_time
        self.benchmark = df