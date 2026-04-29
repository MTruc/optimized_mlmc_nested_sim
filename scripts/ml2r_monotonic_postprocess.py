import sys
sys.path.append("./")

import importlib
import json
from pathlib import Path
import alm_model.nested as nested
import alm_model.estimators as estim
import cupy as cp
import numpy as np
import math
import pandas as pd
from alm_model.nested.structural_consts import StructuralConstantsWERVar

c1 = 0.025
V1_anti = 0.01
a = 2
sigma_bar_1 = math.sqrt(0.005 * 0.995)
budget = 10**7
q_ref = 252.75873881492225
rng = cp.random.default_rng(seed=25042026)
threshold = 0.995
n_trials = 1000
n_clostest_target = 200

input_file = Path("./inputs/preprocessing_parameters.json")
with open(input_file, "r") as jsonfile:
    
    config_params = json.loads(jsonfile.read())
print("[Step 1/8] Parameters loaded")

#GPU or CPU (always GPU actually)
print("Use GPU : True")

#Loaded framework
nested_framework_files = config_params["nested_framework_files"]
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
print("[Step 2/8] Nested framework ready")

structural_consts = StructuralConstantsWERVar(alpha=1, c1=c1, sigma_bar_1=sigma_bar_1, beta=0.5, V1=V1_anti, a=a)
print("[Step 3/8] Structural constants built")
ml2r_theo_res = estim.ML2RTheoreticalResults(structural_consts, alm_nested_framework.params_nested)
calibrator = estim.OptimizedML2RCalibrator(structural_consts, alm_nested_framework.params_nested, ml2r_theo_res)
ml2r_params, eps = calibrator.calibrate_cost(budget)
print("[Step 4/8] Calibration done")
print(f"Calibration epsilon: {eps}")
monotone_method = config_params.get("monotone_method", "cummax")

output_root = Path("./outputs/ml2r_monotonic_postprocess")
output_root.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_root}")


def make_monotone_cdf(v_grid, f_hat, method):
    v_grid_np = np.asarray(v_grid, dtype=float)
    f_hat_np = np.clip(np.asarray(f_hat, dtype=float), 0.0, 1.0)

    if method == "cummax":
        return np.maximum.accumulate(f_hat_np)

    if method == "isotonic":
        isotonic_module_spec = importlib.util.find_spec("sklearn.isotonic")
        if isotonic_module_spec is None:
            raise ImportError(
                "monotone_method='isotonic' requires scikit-learn. "
                "Install scikit-learn or switch monotone_method to 'cummax'."
            )

        isotonic_module = importlib.import_module("sklearn.isotonic")
        isotonic_regression_cls = getattr(isotonic_module, "IsotonicRegression")

        order = np.argsort(v_grid_np)
        v_grid_sorted = v_grid_np[order]
        f_hat_sorted = f_hat_np[order]
        iso = isotonic_regression_cls(y_min=0.0, y_max=1.0, increasing=True)
        f_iso_sorted = iso.fit_transform(v_grid_sorted, f_hat_sorted)

        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(order.size)
        return f_iso_sorted[inverse_order]

    raise ValueError(f"Unknown monotone_method: {method}")


def analyze_sample(sample, quantile_level, n_closest_target, threshold, monotone_method, detailed_logs=False):
    q_first = float(sample.root_find(quantile_level))
    if detailed_logs:
        print("[Step 5/8] Sample generated and q_first computed")
        print(f"q_first: {q_first}")

    # Build one flat array with first level + all 3 arrays for each upper level.
    all_parts = [cp.ravel(sample.EK_first_level)]
    for level_arrays in sample.EK_upper_levels:
        for arr in level_arrays:
            all_parts.append(cp.ravel(arr))

    final_sorted_array = cp.sort(cp.concatenate(all_parts))
    if detailed_logs:
        print("[Step 6/8] final_sorted_array built and sorted")
        print(f"final_sorted_array size: {final_sorted_array.size}")

    n_closest = min(n_closest_target, int(final_sorted_array.size))
    closest_idx = cp.argsort(cp.abs(final_sorted_array - q_first))[:n_closest]
    closest_values = cp.sort(final_sorted_array[closest_idx])
    if detailed_logs:
        print("[Step 7/8] closest_values selected around q_first")
        print(f"n_closest used: {n_closest}")

    evaluated_closest_values_list = []
    running_max_eval = -math.inf
    for u in cp.asnumpy(closest_values):
        curr_eval = float(sample.evaluate(float(u)))
        evaluated_closest_values_list.append(curr_eval)

    evaluated_closest_values_np = make_monotone_cdf(
        v_grid=cp.asnumpy(closest_values),
        f_hat=evaluated_closest_values_list,
        method=monotone_method,
    )
    evaluated_closest_values = cp.asarray(evaluated_closest_values_np)

    if detailed_logs:
        print("[Step 8/8] evaluated_closest_values computed")
        print(f"closest_values shape: {closest_values.shape}")
        print(f"evaluated_closest_values shape: {evaluated_closest_values.shape}")
        print(f"Monotone method applied to evaluated_closest_values: {monotone_method}")

    mask = evaluated_closest_values > threshold
    if bool(cp.any(mask)):
        first_idx = int(cp.where(mask)[0][0])
        closest_value_above_threshold = float(cp.asnumpy(closest_values[first_idx]))
        if detailed_logs:
            print(f"closest_value_above_threshold: {closest_value_above_threshold}")
    else:
        closest_value_above_threshold = None
        if detailed_logs:
            print("No closest_values element has evaluated value > 0.995")

    return q_first, closest_value_above_threshold


sampler = estim.ML2RSamplerGPU(alm_nested_framework, rng, ml2r_theo_res, 0.5)

benchmark_methods = ["cummax", "isotonic"]
n_trials = int(config_params.get("rmse_trials", n_trials))
print_every = 1
results = []

for method in benchmark_methods:
    print(f"\n=== Method: {method} ===")

    sample = sampler.generate_sample(ml2r_params)
    q_first, closest_value_above_threshold = analyze_sample(
        sample=sample,
        quantile_level=threshold,
        n_closest_target=n_clostest_target,
        threshold=threshold,
        monotone_method=method,
        detailed_logs=True,
    )

    print("=== Final comparison ===")
    if closest_value_above_threshold is not None:
        abs_diff = abs(closest_value_above_threshold - q_first)
        rel_diff = abs_diff / abs(q_first) if q_first != 0 else math.nan
        print(f"closest_value_above_threshold: {closest_value_above_threshold}")
        print(f"q_first: {q_first}")
        print(f"absolute difference: {abs_diff}")
        print(f"relative difference: {rel_diff}")
    else:
        abs_diff = math.nan
        rel_diff = math.nan
        print("Comparison skipped: no value above threshold was found.")

    print("=== RMSE benchmark loop ===")
    sum_sq_q_first = 0.0
    sum_sq_closest = 0.0
    count_closest = 0

    for i in range(1, n_trials + 1):
        sample_i = sampler.generate_sample(ml2r_params)
        q_first_i, closest_i = analyze_sample(
            sample=sample_i,
            quantile_level=0.995,
            n_closest_target=40,
            threshold=threshold,
            monotone_method=method,
            detailed_logs=False,
        )
        sum_sq_q_first += (q_first_i - q_ref) ** 2

        if closest_i is not None:
            sum_sq_closest += (closest_i - q_ref) ** 2
            count_closest += 1

        if i % print_every == 0 or i == n_trials:
            rmse_q_first_running = math.sqrt(sum_sq_q_first / i)
            rmse_closest_running = math.sqrt(sum_sq_closest / count_closest) if count_closest > 0 else math.nan
            print(
                "[{}][RMSE progress] {}/{} | RMSE(q_first, q_ref)={:.6f} | RMSE(closest, q_ref)={:.6f} | valid_closest={}".format(
                    method,
                    i,
                    n_trials,
                    rmse_q_first_running,
                    rmse_closest_running,
                    count_closest,
                ),
                end="\r",
                flush=True,
            )

    print()

    rmse_q_first = math.sqrt(sum_sq_q_first / n_trials)
    rmse_closest = math.sqrt(sum_sq_closest / count_closest) if count_closest > 0 else math.nan

    print("=== RMSE final ===")
    print(f"method: {method}")
    print(f"n_trials: {n_trials}")
    print(f"RMSE(q_first, q_ref): {rmse_q_first}")
    print(f"RMSE(closest_value_above_threshold, q_ref): {rmse_closest}")
    print(f"valid_closest_trials: {count_closest}/{n_trials}")

    results.append(
        {
            "method": method,
            "budget": budget,
            "threshold": threshold,
            "n_trials": n_trials,
            "q_ref": q_ref,
            "q_first_one_sample": q_first,
            "closest_value_above_threshold_one_sample": closest_value_above_threshold,
            "abs_diff_one_sample": abs_diff,
            "rel_diff_one_sample": rel_diff,
            "rmse_q_first": rmse_q_first,
            "rmse_closest": rmse_closest,
            "valid_closest_trials": count_closest,
        }
    )

results_df = pd.DataFrame(results)
output_file = output_root / "benchmark_methods.csv"
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")