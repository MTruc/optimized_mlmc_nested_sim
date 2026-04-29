import sys
sys.path.append("./")

import json
from pathlib import Path
import alm_model.nested as nested
import alm_model.estimators as estim
import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import math
import pandas as pd

## Script constants
seed = 1908
u_ref = 252.75873881492203
nb_trials = 250
calibration_budgets = [10_000, 100_000, 1_000_000]
c1_ref = -0.025
V1_ref = 0.01
c2_ref = 0.05
sigma_1_ref = math.sqrt(0.005 * 0.995)

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

rng = cp.random.default_rng(seed)

def preprocessing_constants_fn(J, k, rng, u_ref_val=None):

    if u_ref_val is not None:
        return alm_nested_framework.preprocessing_constants(J, k, rng, u_ref=u_ref_val)
    return alm_nested_framework.preprocessing_constants(J, k, rng)


def compute_metrics(estimates, reference_value):
    arr = np.array([np.nan if x is None else float(x) for x in estimates], dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean_val = float(np.nanmean(arr))
    bias_val = mean_val - reference_value
    var_val = float(np.nanvar(arr))
    squared_errors = (arr - reference_value)**2
    mse_val = np.mean(squared_errors)
    std_squared_errors = np.std(squared_errors)
    ci_radius_mse = 1.96 * std_squared_errors / math.sqrt(len(arr))
    rmse_val = math.sqrt(mse_val)
    rmse_lower = math.sqrt(mse_val - ci_radius_mse if mse_val > ci_radius_mse else 0)
    rmse_upper = math.sqrt(mse_val + ci_radius_mse)
    rmse_pct_val = 100 * rmse_val / abs(reference_value) if reference_value != 0 else np.nan
    rmse_pct_val_lower = 100 * rmse_lower / abs(reference_value) if reference_value != 0 else np.nan
    rmse_pct_val_upper = 100 * rmse_upper / abs(reference_value) if reference_value != 0 else np.nan
    return mean_val, bias_val, var_val, mse_val, rmse_val, rmse_lower, rmse_upper, rmse_pct_val, rmse_pct_val_lower, rmse_pct_val_upper


def run_preprocessing_benchmark(benchmark_name, estimator_fn, with_u_ref=False):
    metrics_rows = []
    print("\n=== Running benchmark: {} ===".format(benchmark_name))

    for budget in calibration_budgets:

        K_star = int(math.floor(budget**(1/3)))
        K = 2**int(math.log2(K_star))
        k = int(math.log2(K))
        J = int(math.floor(budget/K))
        print("Calibration budget : {}, J = {}, K = {}".format(J * K, J, K))

        c1_list = []
        V1_list = []
        c2_list = []
        sigma_bar_1_list = []
        u_estim_list = []

        for i in range(nb_trials):
            print("\rBudget {} - Trial {}/{}".format(J * K, i + 1, nb_trials), end="", flush=True)
            if with_u_ref:
                res = estimator_fn(J, k, rng, u_ref_val=u_ref)
            else:
                res = estimator_fn(J, k, rng)
            c1_list.append(res[0])
            V1_list.append(res[1])
            c2_list.append(res[2])
            sigma_bar_1_list.append(res[3])
            u_estim_list.append(res[4])

        print("")
            
        c1_estimates = np.array(c1_list)
        V1_estimates = np.array(V1_list)
        c2_estimates = np.array([np.nan if x is None else float(x) for x in c2_list], dtype=float)
        sigma_bar_1_estimates = np.array(sigma_bar_1_list)
        u_estim_estimates = np.array(u_estim_list)

        c1_mean, c1_bias, c1_var, c1_mse, c1_rmse, c1_rmse_lower, c1_rmse_upper, c1_rmse_pct, c1_rmse_pct_lower, c1_rmse_pct_upper = compute_metrics(c1_estimates, c1_ref)
        V1_mean, V1_bias, V1_var, V1_mse, V1_rmse, V1_rmse_lower, V1_rmse_upper, V1_rmse_pct, V1_rmse_pct_lower, V1_rmse_pct_upper = compute_metrics(V1_estimates, V1_ref)
        c2_mean, c2_bias, c2_var, c2_mse, c2_rmse, c2_rmse_lower, c2_rmse_upper, c2_rmse_pct, c2_rmse_pct_lower, c2_rmse_pct_upper = compute_metrics(c2_estimates, c2_ref)
        sigma_bar_1_mean, sigma_bar_1_bias, sigma_bar_1_var, sigma_bar_1_mse, sigma_bar_1_rmse, sigma_bar_1_rmse_lower, sigma_bar_1_rmse_upper, sigma_bar_1_rmse_pct, sigma_bar_1_rmse_pct_lower, sigma_bar_1_rmse_pct_upper = compute_metrics(sigma_bar_1_estimates, sigma_1_ref)
        u_estim_mean, u_estim_bias, u_estim_var, u_estim_mse, u_estim_rmse, u_estim_rmse_lower, u_estim_rmse_upper, u_estim_rmse_pct, u_estim_rmse_pct_lower, u_estim_rmse_pct_upper = compute_metrics(u_estim_estimates, u_ref)

        print("c1 estimate mean : {}, std : {} (min : {}, max : {})".format(c1_mean, np.std(c1_estimates), np.min(c1_estimates), np.max(c1_estimates)))
        print("V1 estimate mean : {}, std : {} (min : {}, max : {})".format(V1_mean, np.std(V1_estimates), np.min(V1_estimates), np.max(V1_estimates)))
        print("c2 estimate mean : {}, std : {} (min : {}, max : {})".format(c2_mean, np.nanstd(c2_estimates), np.nanmin(c2_estimates), np.nanmax(c2_estimates)))
        print("sigma_bar_1 estimate mean : {}, std : {} (min : {}, max : {})".format(sigma_bar_1_mean, np.std(sigma_bar_1_estimates), np.min(sigma_bar_1_estimates), np.max(sigma_bar_1_estimates)))
        print("u_estim estimate mean : {}, std : {} (min : {}, max : {})".format(u_estim_mean, np.std(u_estim_estimates), np.min(u_estim_estimates), np.max(u_estim_estimates)))

        print("\n--- Bias / Variance / MSE vs reference [{}] ---".format(benchmark_name))
        print("c1:         bias = {:.6e}, variance = {:.6e}, MSE = {:.6e}, RMSE = {:.6e} ({:.6e} - {:.6e}), RMSE PCT {:.6e} % ({:.6e}% - {:.6e}%)".format(c1_bias, c1_var, c1_mse, c1_rmse, c1_rmse_lower, c1_rmse_upper, c1_rmse_pct, c1_rmse_pct_lower, c1_rmse_pct_upper))
        print("V1:         bias = {:.6e}, variance = {:.6e}, MSE = {:.6e}, RMSE = {:.6e} ({:.6e} - {:.6e}), RMSE PCT {:.6e} % ({:.6e}% - {:.6e}%)".format(V1_bias, V1_var, V1_mse, V1_rmse, V1_rmse_lower, V1_rmse_upper, V1_rmse_pct, V1_rmse_pct_lower, V1_rmse_pct_upper))
        print("c2:         bias = {:.6e}, variance = {:.6e}, MSE = {:.6e}, RMSE = {:.6e} ({:.6e} - {:.6e}), RMSE PCT {:.6e} % ({:.6e}% - {:.6e}%)".format(c2_bias, c2_var, c2_mse, c2_rmse, c2_rmse_lower, c2_rmse_upper, c2_rmse_pct, c2_rmse_pct_lower, c2_rmse_pct_upper))
        print("sigma_bar_1: bias = {:.6e}, variance = {:.6e}, MSE = {:.6e}, RMSE = {:.6e} ({:.6e} - {:.6e}), RMSE PCT {:.6e} % ({:.6e}% - {:.6e}%)".format(sigma_bar_1_bias, sigma_bar_1_var, sigma_bar_1_mse, sigma_bar_1_rmse, sigma_bar_1_rmse_lower, sigma_bar_1_rmse_upper, sigma_bar_1_rmse_pct, sigma_bar_1_rmse_pct_lower, sigma_bar_1_rmse_pct_upper))
        print("u_estim:    bias = {:.6e}, variance = {:.6e}, MSE = {:.6e}, RMSE = {:.6e} ({:.6e} - {:.6e}), RMSE PCT {:.6e} % ({:.6e}% - {:.6e}%)".format(u_estim_bias, u_estim_var, u_estim_mse, u_estim_rmse, u_estim_rmse_lower, u_estim_rmse_upper, u_estim_rmse_pct, u_estim_rmse_pct_lower, u_estim_rmse_pct_upper))

        metrics_rows.extend([
            {
                "benchmark_name": benchmark_name,
                "with_u_ref": with_u_ref,
                "u_ref": u_ref if with_u_ref else np.nan,
                "k_method": k,
                "calibration_budget_target": budget,
                "calibration_budget_effective": J * 2**k,
                "J": J,
                "K": 2**k,
                "nb_trials": nb_trials,
                "metric_name": "c1",
                "mean_estimate": c1_mean,
                "reference_value": c1_ref,
                "bias": c1_bias,
                "variance": c1_var,
                "mse": c1_mse,
                "rmse": c1_rmse,
                "rmse_lower": c1_rmse_lower,
                "rmse_upper": c1_rmse_upper,
                "rmse_pct": c1_rmse_pct,
                "rmse_pct_lower": c1_rmse_pct_lower,
                "rmse_pct_upper": c1_rmse_pct_upper,
            },
            {
                "benchmark_name": benchmark_name,
                "with_u_ref": with_u_ref,
                "u_ref": u_ref if with_u_ref else np.nan,
                "k_method": k,
                "calibration_budget_target": budget,
                "calibration_budget_effective": J * 2**k,
                "J": J,
                "K": 2**k,
                "nb_trials": nb_trials,
                "metric_name": "V1",
                "mean_estimate": V1_mean,
                "reference_value": V1_ref,
                "bias": V1_bias,
                "variance": V1_var,
                "mse": V1_mse,
                "rmse": V1_rmse,
                "rmse_lower": V1_rmse_lower,
                "rmse_upper": V1_rmse_upper,
                "rmse_pct": V1_rmse_pct,
                "rmse_pct_lower": V1_rmse_pct_lower,
                "rmse_pct_upper": V1_rmse_pct_upper,
            },
            {
                "benchmark_name": benchmark_name,
                "with_u_ref": with_u_ref,
                "u_ref": u_ref if with_u_ref else np.nan,
                "k_method": k,
                "calibration_budget_target": budget,
                "calibration_budget_effective": J * 2**k,
                "J": J,
                "K": 2**k,
                "nb_trials": nb_trials,
                "metric_name": "c2",
                "mean_estimate": c2_mean,
                "reference_value": c2_ref,
                "bias": c2_bias,
                "variance": c2_var,
                "mse": c2_mse,
                "rmse": c2_rmse,
                "rmse_lower": c2_rmse_lower,
                "rmse_upper": c2_rmse_upper,
                "rmse_pct": c2_rmse_pct,
                "rmse_pct_lower": c2_rmse_pct_lower,
                "rmse_pct_upper": c2_rmse_pct_upper,
            },
            {
                "benchmark_name": benchmark_name,
                "with_u_ref": with_u_ref,
                "u_ref": u_ref if with_u_ref else np.nan,
                "k_method": k,
                "calibration_budget_target": budget,
                "calibration_budget_effective": J * 2**k,
                "J": J,
                "K": 2**k,
                "nb_trials": nb_trials,
                "metric_name": "sigma_bar_1",
                "mean_estimate": sigma_bar_1_mean,
                "reference_value": sigma_1_ref,
                "bias": sigma_bar_1_bias,
                "variance": sigma_bar_1_var,
                "mse": sigma_bar_1_mse,
                "rmse": sigma_bar_1_rmse,
                "rmse_lower": sigma_bar_1_rmse_lower,
                "rmse_upper": sigma_bar_1_rmse_upper,
                "rmse_pct": sigma_bar_1_rmse_pct,
                "rmse_pct_lower" : sigma_bar_1_rmse_pct_lower,
                "rmse_pct_upper" : sigma_bar_1_rmse_pct_upper
            },
            {
                "benchmark_name": benchmark_name,
                "with_u_ref": with_u_ref,
                "u_ref": u_ref if with_u_ref else np.nan,
                "k_method": k,
                "calibration_budget_target": budget,
                "calibration_budget_effective": J * 2**k,
                "J": J,
                "K": 2**k,
                "nb_trials": nb_trials,
                "metric_name": "u_estim",
                "mean_estimate": u_estim_mean,
                "reference_value": u_ref,
                "bias": u_estim_bias,
                "variance": u_estim_var,
                "mse": u_estim_mse,
                "rmse": u_estim_rmse,
                "rmse_lower": u_estim_rmse_lower,
                "rmse_upper": u_estim_rmse_upper,
                "rmse_pct": u_estim_rmse_pct,
                "rmse_pct_lower": u_estim_rmse_pct_lower,
                "rmse_pct_upper": u_estim_rmse_pct_upper,
            },
        ])

    return metrics_rows


metrics_rows = []
metrics_rows.extend(run_preprocessing_benchmark("quantile", estimator_fn=preprocessing_constants_fn, with_u_ref=False))
metrics_rows.extend(run_preprocessing_benchmark("probability", estimator_fn=preprocessing_constants_fn, with_u_ref=True))

metrics_df = pd.DataFrame(metrics_rows)
output_dir = Path("./outputs/practical_preprocessing")
output_dir.mkdir(parents=True, exist_ok=True)
output_metrics_file = output_dir / "practical_preprocessing_benchmark.csv"
metrics_df.to_csv(output_metrics_file, index=False, sep=";")
print("\nSaved metrics CSV to {} ({} rows)".format(output_metrics_file, len(metrics_df)))