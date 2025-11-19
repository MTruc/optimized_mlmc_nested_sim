import sys
sys.path.append("./")

import json
from pathlib import Path
import alm_model.nested as nested
import alm_model.estimators as estim
import pandas as pd
import cupy as cp
import alm_model.utils as utils
import matplotlib.pyplot as plt

main_input_file = Path("./inputs/main_parameters.json")
with open(main_input_file, "r") as jsonfile:
    
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

structural_consts = nested.load_structural_constants(nested_framework_files["structural_constants_file"])
msg_loaded_structural_constants = "Loaded structural constants"
print(msg_loaded_structural_constants)
print("="*len("Loaded structural constants"))
print("alpha = {}".format(structural_consts.alpha))
print("c1 = {}".format(structural_consts.c1))
print("sigma_bar_1 = {}".format(structural_consts.sigma_bar_1))
print("beta = {}".format(structural_consts.beta))
print("V1 = {}".format(structural_consts.V1))
print("a = {}".format(structural_consts.a))
print("="*len("Loaded structural constants"))
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
        
        if step == "calibration":
            
            if key == "nested_mc":
                
                calibrator = estim.NestedMC1Calibrator(structural_consts, alm_nested_framework.params_nested)
                print("Calibrating a nested MC with alpha = 1.")
            
            elif key == "optimized_ml2r":
                
                ml2r_theo_res = estim.ML2RTheoreticalResults(structural_consts,
                                                             alm_nested_framework.params_nested)
                calibrator = estim.OptimizedML2RCalibrator(structural_consts,
                                                           alm_nested_framework.params_nested,
                                                           ml2r_theo_res)
                print("Calibrating an optimized ML2R.")
            
            elif key == "closed_ml2r":
                
                print("Calibrating a closed ML2R.")
                ml2r_theo_res = estim.ML2RTheoreticalResults(structural_consts,
                                                alm_nested_framework.params_nested)
                
                K_bar = curr_params["calibration"]["K_bar"]
                calibrator = estim.ClosedML2RCalibrator(structural_consts,
                                            alm_nested_framework.params_nested,
                                            ml2r_theo_res,
                                            K_bar)
            elif key == "optimized_mlmc":
                
                print("Calibrating an optimized MLMC.")
                mlmc_theo_results = estim.MLMCTheoreticalResults(structural_consts,
                                                                     alm_nested_framework.params_nested)
                calibrator = estim.OptimizedMLMCCalibrator(structural_consts,
                                                           alm_nested_framework.params_nested,
                                                           mlmc_theo_results)
            elif key == "closed_mlmc":
                
                print("Calibrating a closed MLMC.")
                mlmc_theo_results = estim.MLMCTheoreticalResults(structural_consts,
                                                                 alm_nested_framework.params_nested)
                K_bar = curr_params["calibration"]["K_bar"]
                calibrator = estim.ClosedMLMCCalibrator(structural_consts,
                                                        alm_nested_framework.params_nested,
                                                        mlmc_theo_results,
                                                        K_bar)    
            else:
                raise ValueError("Unkown estimator. ({})".format(key))
                
            print("Calibrated parameters :")
            epsilons = curr_params["calibration"]["epsilon"]
            calibrator.calibration_csv(epsilons, output_folder / "calibration.csv")
            print(pd.read_csv(output_folder / "calibration.csv", sep=";", index_col=0))
            
        if step == "theoretical_benchmark":
            
            parameters_df = pd.read_csv(output_folder / "calibration.csv", sep=";", index_col=0)
            
            if key == "nested_mc":
                
                mlmc_theo_results = estim.NestedMCTheoreticalResults(structural_consts,
                                                                     alm_nested_framework.params_nested)
            
            elif key == "optimized_ml2r" or key == "closed_ml2r":
                
                mlmc_theo_results = estim.ML2RTheoreticalResults(structural_consts,
                                                             alm_nested_framework.params_nested)
            
            elif key == "optimized_mlmc" or key == "closed_mlmc":
                
                mlmc_theo_results = estim.MLMCTheoreticalResults(structural_consts,
                                                                 alm_nested_framework.params_nested)
            else:
                raise ValueError("Unkown estimator. ({})".format(key))
            
            theo_benchmarker = estim.MLMCTheoreticalBenchmarker(parameters_df, mlmc_theo_results)
            theo_benchmarker.run_benchmark()
            theo_benchmarker.export_benchmark(output_folder / "theo_benchmark.csv")
            print(pd.read_csv(output_folder / "theo_benchmark.csv", sep=";", index_col=0))
    
        if step == "sampling":
            
            parameters_df = pd.read_csv(output_folder / "calibration.csv", sep=";", index_col=0)
            seed = curr_params["sampling"]["seed"]
            gpu_memory_margin = curr_params["sampling"]["mem_margin"]
            rng = cp.random.default_rng(seed)
            if key == "nested_mc":
                sampler = estim.NestedMCSamplerGPU(alm_nested_framework,
                                         rng,
                                         gpu_memory_margin)
            
            elif key == "optimized_ml2r" or key == "closed_ml2r":
                
                mlmc_theo_results = estim.ML2RTheoreticalResults(structural_consts,
                                                             alm_nested_framework.params_nested)
                sampler = estim.ML2RSamplerGPU(alm_nested_framework,
                                               rng,
                                               mlmc_theo_results,
                                               gpu_memory_margin)
            
            elif key == "optimized_mlmc" or key == "closed_mlmc":
                
                mlmc_theo_results = estim.MLMCTheoreticalResults(structural_consts,
                                                                 alm_nested_framework.params_nested)
                sampler = estim.MLMCSamplerGPU(alm_nested_framework,
                                               rng,
                                               mlmc_theo_results,
                                               gpu_memory_margin)
            else:
                raise ValueError("Unkown estimator. ({})".format(key))
            
            n_samples = curr_params["sampling"]["n_samples"]
            u = curr_params["sampling"]["cdf_threshold"]
            quantile_level = curr_params["sampling"]["quantile_level"]
            target_cdf = curr_params["sampling"]["target_cdf"]
            target_quantile = curr_params["sampling"]["target_quantile"]
            emp_benchmarker = estim.MLMCQuantileCdfEmpiricalBenchmarker(parameters_df,
                                                                        sampler,
                                                                        n_samples,
                                                                        u,
                                                                        quantile_level,
                                                                        target_cdf,
                                                                        target_quantile)
            emp_benchmarker.run_benchmark()
            emp_benchmarker.export_benchmark(output_folder / "empirical_benchmark.csv")
            print(pd.read_csv(output_folder / "empirical_benchmark.csv", sep=";", index_col=0))
            
        if step == "plot":
            
            theo_bench = pd.read_csv(output_folder / "theo_benchmark.csv", sep=";", index_col=0)
            emp_bench = pd.read_csv(output_folder / "empirical_benchmark.csv", sep=";", index_col=0)
            
            print("Plotting {}".format(key))
            utils.plot_estimator(theo_bench, emp_bench, output_folder)
            
        print("="*len(msg_step))
        print("")
    
    i += 1

cdf_rmse = {}
cdf_rmse_upper = {}
cdf_rmse_lower = {}
times = {}
costs = {}
quantile_rmse = {}
quantile_rmse_upper = {}
quantile_rmse_lower = {}   
params_plot_all = params["plot_all_complexity"]
epsilons = params_plot_all["epsilon"]
if params_plot_all["enabled"]:
    
    msg_print_plot = "Ploting complexity of parameter :"
    print(msg_print_plot)
    print("="*len(msg_print_plot))
    for estim_name in params_plot_all["estimators_list"]:
        
        print("- {}".format(estim_name))
        output_file = Path("./outputs") / estim_name / "empirical_benchmark.csv"
        res = pd.read_csv(output_file, sep=";", index_col=0)
        res = res.loc[epsilons]
        output_file_theo = Path("./outputs") / estim_name / "theo_benchmark.csv"
        res_theo = pd.read_csv(output_file_theo, sep=";", index_col=0)
        res_theo = res_theo.loc[epsilons]
        cdf_rmse[estim_name] = res["cdf_emp_rmse"]
        cdf_rmse_upper[estim_name] = res["cdf_emp_rmse_upper"]
        cdf_rmse_lower[estim_name] = res["cdf_emp_rmse_lower"]
        costs[estim_name] = res_theo["cost"]
        times[estim_name] = res["avg_time"]
        quantile_rmse[estim_name] = res["quantile_emp_rmse"]
        quantile_rmse_upper[estim_name] = res["quantile_emp_rmse_upper"]
        quantile_rmse_lower[estim_name] = res["quantile_emp_rmse_lower"]
        
    #Plot cdf rmse vs cost
    plt.clf()
    plt.title("C.D.F evaluation")
    for key in cdf_rmse.keys():
        
        plt.plot(cdf_rmse[key], costs[key], marker="o", label=key)
        plt.fill_betweenx(costs[key],
                          cdf_rmse_lower[key],
                          cdf_rmse_upper[key],
                          alpha=0.2)
    
    plt.xscale("log")
    plt.xlabel("Empirical RMSE")
    plt.yscale("log")
    plt.ylabel("Computational cost")
    plt.legend()
    plt.grid()
    plt.savefig(Path("./outputs/cdf_all_rmse_cost.png"))
    
    #Plot cdf rmse vs time
    plt.clf()
    plt.title("C.D.F evaluation")
    for key in cdf_rmse.keys():
        
        plt.plot(cdf_rmse[key], times[key], marker="o", label=key)
        plt.fill_betweenx(times[key],
                          cdf_rmse_lower[key],
                          cdf_rmse_upper[key],
                          alpha=0.2)
    
    plt.xscale("log")
    plt.xlabel("Empirical RMSE")
    plt.yscale("log")
    plt.ylabel("Time (in s)")
    plt.legend()
    plt.grid()
    plt.savefig(Path("./outputs/cdf_all_rmse_time.png"))
    
    #Plot quantile rmse vs cost
    plt.clf()
    plt.title("Quantile evaluation")
    for key in quantile_rmse.keys():
        
        plt.plot(quantile_rmse[key], costs[key], marker="o", label=key)
        plt.fill_betweenx(costs[key],
                          quantile_rmse_lower[key],
                          quantile_rmse_upper[key],
                          alpha=0.2)
    
    plt.xscale("log")
    plt.xlabel("Empirical RMSE")
    plt.yscale("log")
    plt.ylabel("Computational cost")
    plt.legend()
    plt.grid()
    plt.savefig(Path("./outputs/quantile_all_rmse_cost.png"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    ax1.set_title("C.D.F evaluation")
    for key in cdf_rmse.keys():
        
        ax1.plot(cdf_rmse[key], costs[key], marker="o", label=key)
        ax1.fill_betweenx(costs[key],
                          cdf_rmse_lower[key],
                          cdf_rmse_upper[key],
                          alpha=0.2)
    
    ax1.set_xscale("log")
    ax1.set_xlabel("Empirical RMSE")
    ax1.set_yscale("log")
    ax1.set_ylabel("Computational cost")
    ax1.legend()
    ax1.grid()

    ax2.set_title("Quantile evaluation")
    for key in quantile_rmse.keys():
        
        ax2.plot(quantile_rmse[key], costs[key], marker="o", label=key)
        ax2.fill_betweenx(costs[key],
                          quantile_rmse_lower[key],
                          quantile_rmse_upper[key],
                          alpha=0.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Empirical RMSE")
    ax2.set_yscale("log")
    ax2.set_ylabel("Computational cost")
    ax2.legend()
    ax2.grid()

    fig.savefig("./outputs/Figure_3.png")
    
    #Plot quantile rmse vs time
    plt.clf()
    plt.title("Quantile evaluation")
    for key in quantile_rmse.keys():
        
        plt.plot(quantile_rmse[key], times[key], marker="o", label=key)
        plt.fill_betweenx(times[key],
                          quantile_rmse_lower[key],
                          quantile_rmse_upper[key],
                          alpha=0.2)
    
    plt.xscale("log")
    plt.xlabel("Empirical RMSE")
    plt.yscale("log")
    plt.ylabel("Time (in s)")
    plt.legend()
    plt.grid()
    plt.savefig(Path("./outputs/quantile_all_rmse_time.png"))
    print("="*len(msg_print_plot))
    print("")

params_tau_sensi = params["sensibility_tau"]
msg_tau_sensi = "Tau sensibility"
print(msg_tau_sensi)
print("="*len(msg_tau_sensi))
print("Loaded steps :")
enabled_steps = []
for step in params_tau_sensi["steps"].keys():
    
    if params_tau_sensi["steps"][step]:
        print("- {}".format(step))
        enabled_steps.append(step)
print("")

for estim_name in params_tau_sensi["estimators"].keys():
    
    for step in enabled_steps:

        print("Step : {}".format(step))
        if step == "calibration":
            params_res = {}
            cost_res = {}
            epsilon_star_res = {}
            params_step = params_tau_sensi["estimators"][estim_name]["calibration"]
            tau_list = params_step["tau"]
            for tau in tau_list:
                print("tau : {}".format(tau))
                secondary_cost = alm_nested_framework.params_nested.secondary_cost
                primary_cost = tau * secondary_cost
                alm_nested_framework.set_costs(primary_cost, secondary_cost)
            
                if estim_name == "optimized_ml2r":
                    
                    ml2r_theo_res = estim.ML2RTheoreticalResults(structural_consts,
                                                                alm_nested_framework.params_nested)
                    calibrator = estim.OptimizedML2RCalibrator(structural_consts,
                                                            alm_nested_framework.params_nested,
                                                            ml2r_theo_res)
                    print("Calibrating an optimized ML2R.")
                    
                elif estim_name == "closed_ml2r":

                    K_bar = params_step["K_bar"]
                    ml2r_theo_res = estim.ML2RTheoreticalResults(structural_consts,
                                                                alm_nested_framework.params_nested)
                    calibrator = estim.ClosedML2RCalibrator(structural_consts,
                                                            alm_nested_framework.params_nested,
                                                            ml2r_theo_res,
                                                            K_bar)
                    print("Calibrating a closed ML2R.")
                
                max_cost = params_step["max_cost"]
                mlmc_parms, epsilon_star = calibrator.calibrate_cost(max_cost)
                params_res[tau] = mlmc_parms
                cost_res[tau] = calibrator.mlmc_theoretical_results.cost(mlmc_parms)
                epsilon_star_res[tau] = epsilon_star
            
            estim.export_dict_MLMCParameters(params_res, Path("./outputs/tau") / "{}_calibration.csv".format(estim_name))
            df_params = pd.read_csv(Path("./outputs/tau") / "{}_calibration.csv".format(estim_name), sep=";", index_col=0)
            df_params["Cost"] = cost_res
            df_params["Epsilon"] = epsilon_star_res
            df_params.to_csv(Path("./outputs/tau") / "{}_calibration.csv".format(estim_name), sep=";")
        
        if step == "sampling":
            params_step = params_tau_sensi["estimators"][estim_name]["sampling"]
            df_params = pd.read_csv(Path("./outputs/tau") / "{}_calibration.csv".format(estim_name), sep=";", index_col=0)
            seed = params_step["seed"]
            gpu_memory_margin = params_step["mem_margin"]
            rng = cp.random.default_rng(seed)
            if estim_name == "optimized_ml2r" or estim_name == "closed_ml2r":
                
                mlmc_theo_results = estim.ML2RTheoreticalResults(structural_consts,
                                                             alm_nested_framework.params_nested)
                sampler = estim.ML2RSamplerGPU(alm_nested_framework,
                                               rng,
                                               mlmc_theo_results,
                                               gpu_memory_margin)
            else:
                raise ValueError("Unkown estimator. ({})".format(key))
            
            n_samples = params_step["n_samples"]
            u = params_step["cdf_threshold"]
            quantile_level = params_step["quantile_level"]
            target_cdf = params_step["target_cdf"]
            target_quantile = params_step["target_quantile"]
            emp_benchmarker = estim.MLMCQuantileCdfEmpiricalBenchmarker(df_params,
                                                                        sampler,
                                                                        n_samples,
                                                                        u,
                                                                        quantile_level,
                                                                        target_cdf,
                                                                        target_quantile)
            emp_benchmarker.run_benchmark()
            output_file = Path("./outputs/tau") / "{}_empirical_benchmark.csv".format(estim_name)
            emp_benchmarker.export_benchmark(output_file)
            print(pd.read_csv(output_file, sep=";", index_col=0))
        
params_step = params_tau_sensi["plot"]
for plot_case in params_step.keys():
    
    if plot_case == "closed_ml2r_vs_optimized_ml2r":
        
        bench_closed_ml2r = pd.read_csv(Path("./outputs/tau") / "closed_ml2r_empirical_benchmark.csv", sep=";", index_col=0)
        bench_optim_ml2r = pd.read_csv(Path("./outputs/tau") / "optimized_ml2r_empirical_benchmark.csv", sep=";", index_col=0)
        mean_efficiency = bench_closed_ml2r["cdf_emp_rmse"] / bench_optim_ml2r["cdf_emp_rmse"]
        upper_efficiency = bench_closed_ml2r["cdf_emp_rmse_upper"] / bench_optim_ml2r["cdf_emp_rmse_lower"]
        lower_efficiency = bench_closed_ml2r["cdf_emp_rmse_lower"] / bench_optim_ml2r["cdf_emp_rmse_upper"]
        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.plot(bench_closed_ml2r.index, mean_efficiency, marker="o")
        plt.fill_between(bench_closed_ml2r.index, lower_efficiency, upper_efficiency, alpha=0.2)
        plt.grid()
        plt.xlabel(r"$\tau$")
        plt.ylabel("Efficiency")
        plt.savefig(Path("./outputs/tau/closed_ml2r_vs_optimized_ml2r.png"))