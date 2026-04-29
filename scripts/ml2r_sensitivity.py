import sys
sys.path.append("./")

import json
from pathlib import Path
import alm_model.nested as nested
import alm_model.estimators as estim
import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import pandas as pd
import math
from alm_model.nested.structural_consts import StructuralConstantsWERVar


seed = 1908
rng = cp.random.default_rng(seed)
gpu_memory_margin = 0.5
sampling_budgets = [int(10**6), int(10**7), int(10**8)]
preprocessing_ratio = 0.01
nb_trials = 250
ref_quantile = 252.75873881492203
ref_u = 0.995
input_file = Path("./inputs/preprocessing_parameters.json")

# Quantile reference benchmark constants
c1_ref = 0.025
V1_anti_ref = 0.01
sigma_bar_1_ref = math.sqrt(0.005 * 0.995)
a_ref = 2

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

def _to_scalar(value):
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.item())

    return float(np.sum(arr))


def _compute_bias_var_rmse(values, reference):
    arr = np.asarray(values, dtype=float)
    bias = float(np.mean(arr) - reference)
    var = float(np.var(arr))
    squared_errors = (arr - reference) ** 2
    mse = float(np.mean(squared_errors))
    mse_ci_radius = 1.96 * np.sqrt(np.var(squared_errors) / arr.size)
    rmse = float(np.sqrt(mse))
    rmse_lower = float(np.sqrt(max(0, mse - mse_ci_radius)))
    rmse_upper = float(np.sqrt(mse + mse_ci_radius))
    return bias, var, rmse, rmse_lower, rmse_upper


results = []

for sampling_budget in sampling_budgets:
    preprocessing_budget = int(preprocessing_ratio * sampling_budget)
    print("\n" + "=" * 80)
    print(f"Sampling budget: {sampling_budget}, Preprocessing budget: {preprocessing_budget}")
    print("=" * 80)

    structural_consts = StructuralConstantsWERVar(alpha=1, c1=c1_ref, sigma_bar_1=sigma_bar_1_ref, beta=0.5, V1=V1_anti_ref, a=a_ref)
    ml2r_theo_res = estim.ML2RTheoreticalResults(structural_consts, alm_nested_framework.params_nested)
    calibrator = estim.OptimizedML2RCalibrator(structural_consts, alm_nested_framework.params_nested, ml2r_theo_res)
    params, eps = calibrator.calibrate_cost(sampling_budget)
    J_ref = _to_scalar(params.J)
    K_ref = _to_scalar(params.K)
    R_ref = _to_scalar(params.R)
    print(f"[Reference][B={sampling_budget}] Calibrated parameters: J={J_ref}, K={K_ref}, R={R_ref}, eps={eps:.6f}")
    sampler = estim.ML2RSamplerGPU(alm_nested_framework, rng, ml2r_theo_res, gpu_memory_margin, time_diagnostic=False)

    errors_ref = []
    for i in range(nb_trials):
        sample = sampler.generate_sample(params)
        val = sample.root_find(ref_u)
        error = abs(val - ref_quantile)
        print(f"[Reference][B={sampling_budget}] Trial {i+1}/{nb_trials} - Estimated quantile: {val:.6f}, Error: {error:.6f}")
        errors_ref.append(error)

    squared_errors_ref = np.array(errors_ref)**2
    mse_ref = np.mean(squared_errors_ref)
    mse_ref_ci_radius = 1.96 * np.sqrt(np.var(squared_errors_ref) / nb_trials)
    rmse_ref = np.sqrt(mse_ref)
    rmse_ref_lower = np.sqrt(max(0, mse_ref - mse_ref_ci_radius))
    rmse_ref_upper = np.sqrt(mse_ref + mse_ref_ci_radius)
    rmse_ref_pct = (rmse_ref / ref_quantile) * 100
    rmse_ref_pct_lower = (rmse_ref_lower / ref_quantile) * 100
    rmse_ref_pct_upper = (rmse_ref_upper / ref_quantile) * 100
    print(f"[Reference][B={sampling_budget}] RMSE: {rmse_ref:.6f}, RMSE (%): {rmse_ref_pct:.4f}%")

    # Quantile benchmark with preprocessing noise
    errors_preproc = []
    J_preproc_list = []
    K_preproc_list = []
    R_preproc_list = []
    for i in range(nb_trials):
        K_star = int(preprocessing_budget**(1/3))
        k_method = max(1, int(math.floor(math.log2(K_star))))
        K_method = 2**k_method
        J = int(preprocessing_budget / K_method)

        res = alm_nested_framework.preprocessing_constants(J, k_method, rng)
        c1 = abs(res[0])
        V1 = res[1]
        sigma_bar_1 = res[3]

        structural_consts_local = StructuralConstantsWERVar(alpha=1, c1=c1, sigma_bar_1=sigma_bar_1, beta=0.5, V1=V1, a=a_ref)

        print("[With preprocessing noise][B={}] Trial {}/{} - Preprocessing done (k={}, J={}, K={})".format(
            sampling_budget, i+1, nb_trials, k_method, J, K_method
        ))

        ml2r_theo_res_local = estim.ML2RTheoreticalResults(structural_consts_local, alm_nested_framework.params_nested)
        calibrator_local = estim.OptimizedML2RCalibrator(structural_consts_local, alm_nested_framework.params_nested, ml2r_theo_res_local)
        params, eps = calibrator_local.calibrate_cost(sampling_budget)
        J_preproc_list.append(_to_scalar(params.J))
        K_preproc_list.append(_to_scalar(params.K))
        R_preproc_list.append(_to_scalar(params.R))

        sampler = estim.ML2RSamplerGPU(alm_nested_framework, rng, ml2r_theo_res_local, gpu_memory_margin, time_diagnostic=False)
        sample = sampler.generate_sample(params)
        val = sample.root_find(ref_u)
        error = abs(val - ref_quantile)
        errors_preproc.append(error)
        print(f"[With preprocessing noise][B={sampling_budget}] Trial {i+1}/{nb_trials} - Calibrated parameters: J={params.J}, K={params.K}, R={params.R}, eps={eps:.6f}, error: {error:.6f}")


    squared_errors_preproc = np.array(errors_preproc)**2
    mse_preproc = np.mean(squared_errors_preproc)
    mse_preproc_ci_radius = 1.96 * np.sqrt(np.var(squared_errors_preproc) / nb_trials)
    rmse_preproc = np.sqrt(mse_preproc)
    rmse_preproc_lower = np.sqrt(max(0, mse_preproc - mse_preproc_ci_radius))
    rmse_preproc_upper = np.sqrt(mse_preproc + mse_preproc_ci_radius)
    rmse_preproc_pct = (rmse_preproc / ref_quantile) * 100
    rmse_preproc_pct_lower = (rmse_preproc_lower / ref_quantile) * 100
    rmse_preproc_pct_upper = (rmse_preproc_upper / ref_quantile) * 100
    rmse_gap = rmse_preproc - rmse_ref
    rmse_gap_pct_points = rmse_preproc_pct - rmse_ref_pct

    J_bias, J_var, J_rmse, J_rmse_lower, J_rmse_upper = _compute_bias_var_rmse(J_preproc_list, J_ref)
    J_rmse_pct = (J_rmse / J_ref) * 100
    J_rmse_lower_pct = (J_rmse_lower / J_ref) * 100
    J_rmse_upper_pct = (J_rmse_upper / J_ref) * 100
    K_bias, K_var, K_rmse, K_rmse_lower, K_rmse_upper = _compute_bias_var_rmse(K_preproc_list, K_ref)
    K_rmse_pct = (K_rmse / K_ref) * 100
    K_rmse_lower_pct = (K_rmse_lower / K_ref) * 100
    K_rmse_upper_pct = (K_rmse_upper / K_ref) * 100
    R_bias, R_var, R_rmse, R_rmse_lower, R_rmse_upper = _compute_bias_var_rmse(R_preproc_list, R_ref)
    R_rmse_pct = (R_rmse / R_ref) * 100
    R_rmse_lower_pct = (R_rmse_lower / R_ref) * 100
    R_rmse_upper_pct = (R_rmse_upper / R_ref) * 100

    print(f"[With preprocessing noise][B={sampling_budget}] RMSE: {rmse_preproc:.6f}, RMSE (%): {rmse_preproc_pct:.4f}%")
    print(f"[With preprocessing noise][B={sampling_budget}] J RMSE vs ref (%): {J_rmse_pct:.2f}, K RMSE vs ref (%): {K_rmse_pct:.2f}, R RMSE vs ref (%): {R_rmse_pct:.2f}")
    print(f"[Delta RMSE][B={sampling_budget}] RMSE gap: {rmse_gap:.6f}, RMSE pct-point gap: {rmse_gap_pct_points:.4f}")

    results.append(
        {
            "sampling_budget": sampling_budget,
            "preprocessing_budget": preprocessing_budget,
            "reference_rmse": rmse_ref,
            "reference_rmse_pct": rmse_ref_pct,
            "reference_J": J_ref,
            "reference_K": K_ref,
            "reference_R": R_ref,
            "with_preprocessing_noise_rmse": rmse_preproc,
            "with_preprocessing_noise_rmse_pct": rmse_preproc_pct,
            "rmse_gap_preproc_minus_ref": rmse_gap,
            "rmse_pct_point_gap_preproc_minus_ref": rmse_gap_pct_points,
            "J_bias_preproc_vs_ref": J_bias,
            "J_variance_preproc": J_var,
            "J_rmse_preproc_vs_ref": J_rmse,
            "J_rmse_preproc_lower": J_rmse_lower,
            "J_rmse_preproc_upper": J_rmse_upper,
            "J_rmse_preproc_vs_ref_pct": J_rmse_pct,
            "J_rmse_preproc_lower_pct": J_rmse_lower_pct,
            "J_rmse_preproc_upper_pct": J_rmse_upper_pct,
            "K_bias_preproc_vs_ref": K_bias,
            "K_variance_preproc": K_var,
            "K_rmse_preproc_vs_ref": K_rmse,
            "K_rmse_preproc_lower": K_rmse_lower,
            "K_rmse_preproc_upper": K_rmse_upper,
            "K_rmse_preproc_vs_ref_pct": K_rmse_pct,
            "K_rmse_preproc_lower_pct": K_rmse_lower_pct,
            "K_rmse_preproc_upper_pct": K_rmse_upper_pct,
            "R_bias_preproc_vs_ref": R_bias,
            "R_variance_preproc": R_var,
            "R_rmse_preproc_vs_ref": R_rmse,
            "R_rmse_preproc_lower": R_rmse_lower,
            "R_rmse_preproc_upper": R_rmse_upper,
            "R_rmse_preproc_vs_ref_pct": R_rmse_pct,
            "R_rmse_preproc_lower_pct": R_rmse_lower_pct,
            "R_rmse_preproc_upper_pct": R_rmse_upper_pct,
            "with_preprocessing_noise_rmse_lower": rmse_preproc_lower,
            "with_preprocessing_noise_rmse_upper": rmse_preproc_upper,
            "with_preprocessing_noise_rmse_pct_lower": rmse_preproc_pct_lower,
            "with_preprocessing_noise_rmse_pct_upper": rmse_preproc_pct_upper,
        }
    )

print("\nSummary by sampling budget")
for row in results:
    print(
        "B={sampling_budget} | Ref RMSE={reference_rmse:.6f} ({reference_rmse_pct:.4f}%) | "
        "Preproc RMSE={with_preprocessing_noise_rmse:.6f} ({with_preprocessing_noise_rmse_pct:.4f}%)".format(**row)
    )

results_df = pd.DataFrame(results)
output_dir = Path("./outputs/ml2r_sensitivity")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "benchmarks.csv"
results_df.to_csv(output_file, index=False)
print(f"\nSaved benchmark results to {output_file} ({len(results_df)} rows)")
