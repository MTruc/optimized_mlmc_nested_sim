import sys
sys.path.append("./")

import json
from pathlib import Path
import alm_model.nested as nested
import alm_model.estimators as estim
import alm_model.utils as utils
import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

input_file = Path("./inputs/preprocessing_parameters.json")
with open(input_file, "r") as jsonfile:
    
    params = json.loads(jsonfile.read())

#GPU or CPU (always GPU actually)
print("Use GPU : True")

#Loaded framework
nested_framework_files = params["nested_framework_files"]
alm_nested_framework = nested.load_nested_alm_framework(nested_framework_files["nested_params_file"],
                                 nested_framework_files["alm_parameters_file"],
                                 nested_framework_files["stock_parameters_file"])

msg_loaded_nested_framework = "Loaded ALM nested framework"
print(msg_loaded_nested_framework)
print("="*len(msg_loaded_nested_framework))
print("s0 = {}, mu = {}, sigma = {}, r = {}".format(alm_nested_framework.model.stock_parameters.s0,
                                                    alm_nested_framework.model.stock_parameters.mu,
                                                    alm_nested_framework.model.stock_parameters.sigma,
                                                    alm_nested_framework.model.stock_parameters.r))
print("mr0 = {}, r_g = {}, ps_rate = {}, exit_rate = {}, horizon = {}".format(alm_nested_framework.model.alm_parameters.mr_0,
                                  alm_nested_framework.model.alm_parameters.min_guaranteed_rate,
                                  alm_nested_framework.model.alm_parameters.ps_rate,
                                  alm_nested_framework.model.alm_parameters.exit_rate,
                                  alm_nested_framework.model.alm_parameters.horizon))
print("Primary cost : {}, Secondary cost : {}, Tau : {} ".format(
    alm_nested_framework.params_nested.primary_cost,
    alm_nested_framework.params_nested.secondary_cost,
    alm_nested_framework.params_nested.tau))
print("="*len(msg_loaded_nested_framework))
print("")

#Estimators
n_test_cases = len(params["estimators"].keys())
msg_n_test_cases = "Loaded {} test cases".format(n_test_cases)
print(msg_n_test_cases)
print("="*len(msg_n_test_cases))

estimators_params = params["estimators"]
for key in estimators_params.keys():
    
    print("- {}".format(key))
    
print("="*len(msg_n_test_cases))
print("")

seed = 1908
rng = cp.random.default_rng(seed)#cp generator

print("Diagnostic GPU memory usage.")
#Get GPU total memory
device = cp.cuda.Device(0)
total_memory_bytes = device.mem_info[1]
gpu_available_memory_in_MB = total_memory_bytes / (1024 ** 2)
gpu_available_memory_in_MB = gpu_available_memory_in_MB

#Measure memory per unit
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
J_diagnostic = 2000
K_diagnostic = 40
estim.sample_EK_gpu(alm_nested_framework, J_diagnostic, K_diagnostic, rng)
cp.cuda.Device().synchronize()
gpu_mem_peak_mb = mempool.total_bytes() / 1024**2
gpu_memory_peak_in_MB_per_unit = gpu_mem_peak_mb / (J_diagnostic * K_diagnostic)
gpu_memory_peak_in_MB_per_unit = gpu_memory_peak_in_MB_per_unit

print("Diagnostic Time GPU Usage")
bench = benchmark(estim.sample_EK_gpu, (alm_nested_framework, J_diagnostic, K_diagnostic, rng), n_repeat=25)
avg_time = np.mean(bench.gpu_times) * 1.25
avg_time_unit = avg_time / (J_diagnostic * K_diagnostic)
avg_time_unit = avg_time_unit

i = 1
#Test all test cases
for key in estimators_params.keys():
    
    msg_test_name = "({} / {}) Running test case {}".format(i, n_test_cases, key)
    print(msg_test_name)
    print("="*len(msg_test_name))
        
    #Create output folder if needed for the estimator
    output_folder = Path("./outputs/{}/".format(key))
    output_folder.mkdir(exist_ok=True)
    curr_params = estimators_params[key]
    
    print("Steps :")
    steps = curr_params["steps"]
    enabled_steps = []
    for steps_name in steps.keys():
        
        if steps[steps_name]:
            enabled_steps.append(steps_name)
            print("- {}".format(steps_name))
    print("")
    
    #Test all steps enabled
    for step in enabled_steps:
        
        msg_step = "Step : {}".format(step)
        print(msg_step)
        print("="*len(msg_step))
        
        if step == "sampling" :
            
            res_sampling = {}
            if key == "all_structural_consts":
                
                rng = cp.random.default_rng(curr_params[step]["seed"])
                gpu_memory_margin = curr_params[step]["mem_margin"]
                threshold = curr_params[step]["threshold"]
                Kf = curr_params[step]["Kf"]
                J = curr_params[step]["J"]

                mean_D1 = {}
                var_D1 = {}
                mean_D1_non_anti = {}
                var_D1_non_anti = {}
                mean_D2 = {}
                var_D2 = {}
                N = {}
                for i in range(len(Kf)):
                    curr_J = J[i]
                    curr_Kf = Kf[i]
                    unit = curr_J * curr_Kf
                    expected_time = avg_time_unit * unit
                    msg_sampling = "Sampling J = {:.2e}, Kf = {:.2e} (expected time : {:.2f} s)".format(curr_J,
                                                                                              curr_Kf,
                                                                                              expected_time)
                    print(msg_sampling)
                    print("-"*len(msg_sampling))
                    max_J = estim.calibrate_max_J_per_batch_nested_mc(gpu_available_memory_in_MB,
                                            gpu_memory_peak_in_MB_per_unit,
                                            curr_Kf,
                                            gpu_memory_margin)
                    sizes = utils.compute_batch_sizes(int(math.ceil(curr_J)), max_J)
                    
                    D1_anti_res = cp.empty(curr_J)
                    D1_non_anti_res = cp.empty(curr_J)
                    D2_res = cp.empty(curr_J)
                    cursor = 0
                    cp.cuda.Stream.null.synchronize()
                    start_time = time.perf_counter()
                    for j in range(len(sizes)):
                        
                        size = sizes[j]
                        D1_anti, D1_non_anti, D2 = alm_nested_framework.sample_D1_both_types_D2_gpu(Kf[i], size, threshold, rng)
                        D1_anti_res[cursor:cursor+size] = D1_anti
                        D1_non_anti_res[cursor:cursor+size] = D1_non_anti
                        D2_res[cursor:cursor+size] = D2
                        cursor += size
                    cp.cuda.Stream.null.synchronize()
                    end_time = time.perf_counter()
                    dt = end_time - start_time
                    avg_dt = dt / len(sizes)
                    print("Took {:.2f} s (avg : {:.2f} s)".format(dt, avg_dt))
                    mean_D1[curr_Kf] = float(D1_anti_res.mean())
                    var_D1[curr_Kf] = float(D1_anti_res.var())
                    mean_D1_non_anti[curr_Kf] = float(D1_non_anti_res.mean())
                    var_D1_non_anti[curr_Kf] = float(D1_non_anti_res.var())
                    mean_D2[curr_Kf] = float(D2_res.mean())
                    var_D2[curr_Kf] = float(D2_res.var())
                    N[curr_Kf] = curr_J
                    print("mean D1 : {:.2e} +- {:.2e}".format(mean_D1[curr_Kf], 1.96 * math.sqrt(var_D1[curr_Kf] / len(D1_anti))))
                    print("mean D1 non anti : {:.2e} +- {:.2e}".format(mean_D1_non_anti[curr_Kf], 1.96 * math.sqrt(var_D1_non_anti[curr_Kf] / len(D1_non_anti))))
                    print("mean D2 : {:.2e} +- {:.2e}".format(mean_D2[curr_Kf], 1.96 * math.sqrt(var_D2[curr_Kf] / len(D2))))
                    print("-"*len(msg_sampling))
                    print("")
            df_res = pd.DataFrame({"Mean D1" : mean_D1,
                                "Var D1" : var_D1,
                                "Mean D1 non anti" : mean_D1_non_anti,
                                "Var D1 non anti" : var_D1_non_anti,
                                "Mean D2" : mean_D2,
                                "Var D2" : var_D2,
                                "N" : N})
            df_res.to_csv(output_folder / "all_struct_consts_stats.csv", sep=";")
            print(df_res)
            
        if step == "plot" :
            
            df_res = pd.read_csv(output_folder / "all_struct_consts_stats.csv", sep=";", index_col=0)
            
            print("Plot for c1.")
            c1_estim = curr_params[step]["c1"]

            df_res["approx_mean_D1"] = [c1_estim / K for K in df_res.index]
            ci_D1 = []
            for idx, row in df_res.iterrows():

                ci_D1.append(1.96 * math.sqrt(row["Var D1"] / row["N"]))
            
            df_res["ci_mean_D1"] = ci_D1

            plt.clf()
            plt.figure(figsize=(7, 4))
            plt.plot(df_res.index, df_res["Mean D1"], marker="o", label=r"Estimated $|\mathbb{E}[\Delta Y^A_{2K}]|$")
            plt.plot(df_res.index, df_res["approx_mean_D1"], marker="o", linestyle="--", label=r"Proxy for bias with $c_1$ = {:.2e}".format(c1_estim))
            plt.fill_between(df_res.index, df_res["Mean D1"] - df_res["ci_mean_D1"], df_res["Mean D1"] + df_res["ci_mean_D1"], alpha=0.3)
            plt.grid()
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"$2K$")
            plt.legend()

            plt.savefig(output_folder / "c1_plot.png", bbox_inches='tight')

            print("Plot for c2.")
            c2_estim = curr_params[step]["c2"]

            df_res["approx_mean_D2"] = [6*c2_estim / (K)**2 for K in df_res.index]
            ci_D2 = []
            for idx, row in df_res.iterrows():

                ci_D2.append(1.96 * math.sqrt(row["Var D2"] / row["N"]))
            
            df_res["ci_mean_D2"] = ci_D2
            df_res["Abs Mean D2"] = np.abs(df_res["Mean D2"])
            df_res["Abs Mean D2 upper"] = np.abs(df_res["Mean D2"] - ci_D2)
            df_res["Abs Mean D2 lower"] = np.abs(np.minimum(0, df_res["Mean D2"] + ci_D2))

            plt.clf()
            plt.figure(figsize=(7, 4))
            plt.plot(df_res.index, df_res["Abs Mean D2"], marker="o", label=r"Estimated $|\mathbb{E}[2Y_{4K} - 3Y_{2K} + Y_K]|$")
            plt.plot(df_res.index, df_res["approx_mean_D2"], marker="o", linestyle="--", label=r"Proxy with $c_2$ = {:.2e}".format(c2_estim))
            plt.fill_between(df_res.index, df_res["Abs Mean D2 lower"], df_res["Abs Mean D2 upper"], alpha=0.3)
            plt.grid()
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"$4K$")
            plt.legend()
            plt.savefig(output_folder / "c2_plot.png", bbox_inches='tight')


            print("Plot for V1.")
            V1_estim = curr_params[step]["V1_anti"]
            df_res["approx_var_anti"] = [V1_estim / math.sqrt(K) for K in df_res.index]

            V1_non_anti_estim = curr_params[step]["V1_non_anti"]
            df_res["approx_var_non_anti"] = [V1_non_anti_estim / math.sqrt(K) for K in df_res.index]

            plt.clf()
            plt.figure(figsize=(7, 4))
            plt.plot(df_res.index, df_res["Var D1"], color = "tab:blue", marker="o", label=r"Estimated $\mathrm{Var}[\Delta Y^A_{2K}]$")
            plt.plot(df_res.index, df_res["approx_var_anti"], color = "tab:blue", marker="o", linestyle="--", label=r"Proxy with $V^A_1$ = {:.2e}".format(V1_estim))
            plt.plot(df_res.index, df_res["Var D1 non anti"], color = "tab:orange",  marker="o", label=r"Estimated $\mathrm{Var}[\Delta Y^S_{2K}]$")
            plt.plot(df_res.index, df_res["approx_var_non_anti"], color = "tab:orange", marker="o", linestyle="--", label=r"Proxy with $V^S_1$ = {:.2e}".format(V1_non_anti_estim))
            plt.grid()
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"$2K$")
            plt.legend()
            plt.savefig(output_folder / "V1_plot.png", bbox_inches='tight')

            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

            # Upper row: two plots side by side
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])

            # Lower row: one plot centered (spanning both columns)
            ax3 = fig.add_subplot(gs[1, :])

            # Example data and plots
            ax1.plot(df_res.index, df_res["Mean D1"], marker="o", label=r"Estimated $|\mathbb{E}[\Delta Y^A_{2K}]|$")
            ax1.plot(df_res.index, df_res["approx_mean_D1"], marker="o", linestyle="--", label=r"Proxy for bias with $c_1$ = {:.2e}".format(c1_estim))
            ax1.fill_between(df_res.index, df_res["Mean D1"] - df_res["ci_mean_D1"], df_res["Mean D1"] + df_res["ci_mean_D1"], alpha=0.3)
            ax1.grid()
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_xlabel(r"$2K$")
            ax1.legend()

            ax2.plot(df_res.index, df_res["Abs Mean D2"], marker="o", label=r"Estimated $|\mathbb{E}[2Y_{4K} - 3Y_{2K} + Y_K]|$")
            ax2.plot(df_res.index, df_res["approx_mean_D2"], marker="o", linestyle="--", label=r"Proxy with $c_2$ = {:.2e}".format(c2_estim))
            ax2.fill_between(df_res.index, df_res["Abs Mean D2 lower"], df_res["Abs Mean D2 upper"], alpha=0.3)
            ax2.grid()
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel(r"$4K$")
            ax2.legend()

            ax3.plot(df_res.index, df_res["Var D1"], color = "tab:blue", marker="o", label=r"Estimated $\mathrm{Var}[\Delta Y^A_{2K}]$")
            ax3.plot(df_res.index, df_res["approx_var_anti"], color = "tab:blue", marker="o", linestyle="--", label=r"Proxy with $V^A_1$ = {:.2e}".format(V1_estim))
            ax3.plot(df_res.index, df_res["Var D1 non anti"], color = "tab:orange",  marker="o", label=r"Estimated $\mathrm{Var}[\Delta Y^S_{2K}]$")
            ax3.plot(df_res.index, df_res["approx_var_non_anti"], color = "tab:orange", marker="o", linestyle="--", label=r"Proxy with $V^S_1$ = {:.2e}".format(V1_non_anti_estim))
            ax3.grid()
            ax3.set_xscale("log")
            ax3.set_yscale("log")
            ax3.set_xlabel(r"$2K$")
            ax3.legend()

            fig.savefig(output_folder / "Figure_2.png", bbox_inches='tight')
            
        print("="*len(msg_step))
        print("")
    
    i += 1