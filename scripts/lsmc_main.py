import sys
sys.path.append("./")

import json
import cupy as cp
import math
import numpy as np
import pandas as pd
from pathlib import Path
import alm_model.nested as nested
import matplotlib.pyplot as plt
from lsmc import LSMCSampler, LSMCCalibrator1D, predict_monomial_1d, save_monomial_model_to_csv
from alm_model.nested.structural_consts import StructuralConstantsWERVar
import alm_model.estimators as estim

INPUT_FILE = Path("./inputs/preprocessing_parameters.json")
BUDGETS = [10**4, 10**5, 10**6, 10**7, 10**8]
N_TRIALS = 250
REF_QUANTILE = 252.75873881492225
QUANTILE_LEVEL = 0.995
N_QUANTILE_SAMPLES = 500000
SEED = 24042026
MAX_TERMS = 20

LSMC_CPU_LINALG = True
OUTPUT_DIR = Path("./outputs/lsmc_ml2r")
OUTPUT_CSV = OUTPUT_DIR / "lsmc.csv"
OUTPUT_POLY_DIR = OUTPUT_DIR / "polynomials"
OUTPUT_FINAL_PLOT = OUTPUT_DIR / "rmse_loglog.png"

def load_nested_framework(input_file):

    with open(input_file, "r", encoding="utf-8") as jsonfile:
        params = json.loads(jsonfile.read())

    nested_framework_files = params["nested_framework_files"]
    alm_nested_framework = nested.load_nested_alm_framework(
        nested_framework_files["nested_params_file"],
        nested_framework_files["alm_parameters_file"],
        nested_framework_files["stock_parameters_file"],
    )
    return alm_nested_framework

def evaluate_budget_ml2r(alm_nested_framework, calib_budget, n_trials, ref_quantile, quantile_level, n_quantile_samples, seed):

    # Quantile reference benchmark
    c1 = 0.025
    V1_anti = 0.01
    sigma_bar_1 = math.sqrt(0.005 * 0.995)
    a = 2
    gpu_memory_margin = 0.5
    rng = cp.random.default_rng(seed=seed)

    structural_consts = StructuralConstantsWERVar(alpha=1, c1=c1, sigma_bar_1=sigma_bar_1, beta=0.5, V1=V1_anti, a=a)
    ml2r_theo_res = estim.ML2RTheoreticalResults(structural_consts, alm_nested_framework.params_nested)
    calibrator = estim.OptimizedML2RCalibrator(structural_consts, alm_nested_framework.params_nested, ml2r_theo_res)
    params, eps = calibrator.calibrate_cost(calib_budget)
    sampler = estim.ML2RSamplerGPU(alm_nested_framework, rng, ml2r_theo_res, gpu_memory_margin)

    quantile_errors = []
    proba_errors = []
    for i in range(n_trials):
        sample = sampler.generate_sample(params)
        val_quantile = sample.root_find(quantile_level)
        val_proba = sample.evaluate(ref_quantile)
        error_quantile = abs(val_quantile - ref_quantile)
        error_proba = abs(val_proba - quantile_level)
        print(
            "\r  Trials budget {}: {}/{} ({:.1f}%)".format(
                calib_budget, i+1, n_trials, 100.0 * (i+1) / n_trials
            ),
            end="",
            flush=True,
        )
        quantile_errors.append(float(error_quantile))
        proba_errors.append(float(error_proba))

    print("")
    quantile_errors = np.array(quantile_errors, dtype=float)
    squared_quantile_errors = np.square(quantile_errors)
    quantile_mse = float(np.mean(squared_quantile_errors))
    quantile_mse_ci_radius = 1.96 * float(np.std(squared_quantile_errors, ddof=1) / np.sqrt(n_trials))
    quantile_rmse = float(np.sqrt(quantile_mse))
    quantile_rmse_lower = float(np.sqrt(max(0.0, quantile_mse - quantile_mse_ci_radius)))
    quantile_rmse_upper = float(np.sqrt(quantile_mse + quantile_mse_ci_radius))
    quantile_bias = float(np.mean(quantile_errors))
    quantile_variance = float(np.var(quantile_errors, ddof=0))

    probability_errors = np.array(proba_errors, dtype=float)
    squared_probability_errors = np.square(probability_errors)
    probability_mse = float(np.mean(squared_probability_errors))
    probability_mse_ci_radius = 1.96 * float(np.std(squared_probability_errors, ddof=1) / np.sqrt(n_trials))
    probability_rmse = float(np.sqrt(np.mean(np.square(probability_errors))))
    probability_rmse_lower = float(np.sqrt(max(0.0, probability_mse - probability_mse_ci_radius)))
    probability_rmse_upper = float(np.sqrt(probability_mse + probability_mse_ci_radius))
    probability_bias = float(np.mean(probability_errors))
    probability_variance = float(np.var(probability_errors, ddof=0))

    return {
        "calib_budget_requested": int(calib_budget),
        "J": params.J,
        "q": params.q,
        "K" : params.K,
        "R": params.R,
        "total_budget_used": ml2r_theo_res.cost(params),
        "n_trials": int(n_trials),
        "quantile_level": float(quantile_level),
        "n_quantile_samples": int(n_quantile_samples),
        "ref_quantile": float(ref_quantile),
        "quantile_rmse": quantile_rmse,
        "quantile_rmse_lower": quantile_rmse_lower,
        "quantile_rmse_upper": quantile_rmse_upper,
        "quantile_rmse_pct": 100.0 * quantile_rmse / abs(ref_quantile),
        "quantile_rmse_pct_lower": 100.0 * quantile_rmse_lower / abs(ref_quantile),
        "quantile_rmse_pct_upper": 100.0 * quantile_rmse_upper / abs(ref_quantile),
        "quantile_bias": quantile_bias,
        "quantile_bias_pct": 100.0 * quantile_bias / abs(ref_quantile),
        "quantile_variance": quantile_variance,
        "probability_rmse": probability_rmse,
        "probability_rmse_lower": probability_rmse_lower,
        "probability_rmse_upper": probability_rmse_upper,
        "probability_rmse_pct": 100.0 * probability_rmse / abs(quantile_level),
        "probability_rmse_pct_lower": 100.0 * probability_rmse_lower / abs(quantile_level),
        "probability_rmse_pct_upper": 100.0 * probability_rmse_upper / abs(quantile_level),
        "probability_bias": probability_bias,
        "probability_bias_pct": 100.0 * probability_bias / abs(quantile_level),
        "probability_variance": probability_variance
    }

def evaluate_budget_lsmc(
    lsmc_sampler,
    alm_nested_framework,
    calib_budget,
    n_trials,
    ref_quantile,
    quantile_level,
    n_quantile_samples,
    seed,
    max_terms,
    output_poly_dir,
    use_cpu_linalg=False,
    stock_lower_bound=40.0,
    stock_upper_bound=180.0
):

    if calib_budget < 2:
        raise ValueError("Each calibration budget must be >= 2.")

    k_inner = 2
    j_train = 2 ** int(math.log2(calib_budget / 2))
    m = int(math.log2(j_train))
    l_bound = cp.array([stock_lower_bound])
    u_bound = cp.array([stock_upper_bound])
    var_names = alm_nested_framework.get_risk_factors_name()

    proxy_quantiles = []
    proxy_probabilities = []
    rng = cp.random.default_rng(seed=seed)
    example_poly_csv = None
    for trial_idx in range(n_trials):
        trial_num = trial_idx + 1
        progress_pct = 100.0 * trial_num / n_trials
        print(
            "\r  Trials budget {}: {}/{} ({:.1f}%)".format(
                calib_budget, trial_num, n_trials, progress_pct
            ),
            end="",
            flush=True,
        )

        df_train = lsmc_sampler.sample_dataset_sobol(m, l_bound, u_bound, rng, k_inner, J_max=10**5)
        x_calib = cp.array(df_train.iloc[:, :-1].values)
        y_calib = cp.array(df_train.iloc[:, -1].values)

        if use_cpu_linalg:
            x_calib_cpu = cp.asnumpy(x_calib)
            y_calib_cpu = cp.asnumpy(y_calib)
            calib1D = LSMCCalibrator1D(x_calib_cpu[:, 0], y_calib_cpu, max_nb_terms=max_terms, var_name=var_names[0], normalize_input=True)
        else:
            calib1D = LSMCCalibrator1D(x_calib[:, 0], y_calib, max_nb_terms=max_terms, var_name=var_names[0], normalize_input=True)
        poly2 = calib1D.fast_calibrate_proxy()

        if trial_idx == 0:
            example_poly_csv = output_poly_dir / "poly_budget_{}.csv".format(calib_budget)
            save_monomial_model_to_csv(poly2, example_poly_csv)
            s1 = np.linspace(stock_lower_bound, stock_upper_bound, 1000)
            model = alm_nested_framework.model
            stock_path = np.empty((1000, 2), dtype=np.float64)
            stock_path[:, 0] = model.stock_parameters.s0
            stock_path[:, 1] = s1
            loss = model.own_fund_loss_cond_stock(stock_path)

            rf = cp.array(s1)
            poly_values = predict_monomial_1d(rf, poly2)
            poly_values = cp.asnumpy(poly_values)

            plt.clf()
            plt.plot(s1, loss, label="True own-fund loss")
            plt.plot(s1, poly_values, label="LSMC proxy")
            plt.xlabel("Stock value at t=1")
            plt.ylabel("Own-fund loss")
            plt.legend()
            plt.savefig(output_poly_dir / "poly_budget_{}.png".format(calib_budget))

        rf_quant = alm_nested_framework.sample_risk_factors(n_quantile_samples, rng)

        if use_cpu_linalg:
            rf_quant_cpu = cp.asnumpy(rf_quant[:, 0])
            poly_values = predict_monomial_1d(rf_quant_cpu, poly2)
        else:
            poly_values = predict_monomial_1d(rf_quant[:, 0], poly2)

        xp = cp.get_array_module(poly_values)
        ind = xp.where(poly_values <= REF_QUANTILE, 1.0, 0.0)
        proxy_proba = float(xp.mean(ind))
        proxy_probabilities.append(proxy_proba)
        proxy_quantile = xp.quantile(poly_values, q=quantile_level)
        proxy_quantiles.append(float(proxy_quantile))

    print("")

    proxy_quantiles = np.asarray(proxy_quantiles, dtype=float)
    quantile_errors = proxy_quantiles - ref_quantile
    print(quantile_errors)
    squared_quantile_errors = np.square(quantile_errors)
    quantile_mse = float(np.mean(squared_quantile_errors))
    quantile_mse_ci_radius = 1.96 * float(np.std(squared_quantile_errors, ddof=1) / np.sqrt(n_trials))
    quantile_rmse = float(np.sqrt(quantile_mse))
    quantile_rmse_lower = float(np.sqrt(max(0.0, quantile_mse - quantile_mse_ci_radius)))
    quantile_rmse_upper = float(np.sqrt(quantile_mse + quantile_mse_ci_radius))
    quantile_bias = float(np.mean(quantile_errors))
    quantile_variance = float(np.var(proxy_quantiles, ddof=0))

    proxy_probabilities = np.asarray(proxy_probabilities, dtype=float)
    probability_errors = proxy_probabilities - quantile_level
    squared_probability_errors = np.square(probability_errors)
    probability_mse = float(np.mean(squared_probability_errors))
    probability_mse_ci_radius = 1.96 * float(np.std(squared_probability_errors, ddof=1) / np.sqrt(n_trials))
    probability_rmse = float(np.sqrt(np.mean(np.square(probability_errors))))
    probability_rmse_lower = float(np.sqrt(max(0.0, probability_mse - probability_mse_ci_radius)))
    probability_rmse_upper = float(np.sqrt(probability_mse + probability_mse_ci_radius))
    probability_bias = float(np.mean(probability_errors))
    probability_variance = float(np.var(proxy_probabilities, ddof=0))

    return {
        "calib_budget_requested": int(calib_budget),
        "J_train": int(j_train),
        "K": int(k_inner),
        "total_budget_used": int(j_train * k_inner),
        "n_trials": int(n_trials),
        "quantile_level": float(quantile_level),
        "n_quantile_samples": int(n_quantile_samples),
        "ref_quantile": float(ref_quantile),
        "proxy_quantile_mean": float(np.mean(proxy_quantiles)),
        "proxy_quantile_std": float(np.std(proxy_quantiles, ddof=0)),
        "quantile_rmse": quantile_rmse,
        "quantile_rmse_lower": quantile_rmse_lower,
        "quantile_rmse_upper": quantile_rmse_upper,
        "quantile_rmse_pct": 100.0 * quantile_rmse / abs(ref_quantile),
        "quantile_rmse_pct_lower": 100.0 * quantile_rmse_lower / abs(ref_quantile),
        "quantile_rmse_pct_upper": 100.0 * quantile_rmse_upper / abs(ref_quantile),
        "quantile_bias": quantile_bias,
        "quantile_bias_pct": 100.0 * quantile_bias / abs(ref_quantile),
        "quantile_variance": quantile_variance,
        "proxy_probability_mean": float(np.mean(proxy_probabilities)),
        "proxy_probability_std": float(np.std(proxy_probabilities, ddof=0)),
        "probability_rmse": probability_rmse,
        "probability_rmse_lower": probability_rmse_lower,
        "probability_rmse_upper": probability_rmse_upper,
        "probability_rmse_pct": 100.0 * probability_rmse / abs(quantile_level),
        "probability_rmse_pct_lower": 100.0 * probability_rmse_lower / abs(quantile_level),
        "probability_rmse_pct_upper": 100.0 * probability_rmse_upper / abs(quantile_level),
        "probability_bias": probability_bias,
        "probability_bias_pct": 100.0 * probability_bias / abs(quantile_level),
        "probability_variance": probability_variance,
        "example_poly_csv": str(example_poly_csv),
    }


def main():

    alm_nested_framework = load_nested_framework(INPUT_FILE)
    lsmc_sampler = LSMCSampler(alm_nested_framework)

    sample = alm_nested_framework.sample_risk_factors(500_000, rng=cp.random.default_rng(seed=SEED)).reshape(-1)
    stock_lower_bound = float(cp.quantile(sample, q=0.001))
    stock_upper_bound = float(cp.quantile(sample, q=0.999))
    print("Stock lower bound (0.1% quantile) : {:.2f}".format(stock_lower_bound))
    print("Stock upper bound (99.9% quantile) : {:.2f}".format(stock_upper_bound))
    print("Use GPU : True")
    print("Budgets : {}".format(BUDGETS))
    print("Trials per budget : {}".format(N_TRIALS))

    rows_lsmc = []
    rows_ml2r = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_POLY_DIR.mkdir(parents=True, exist_ok=True)
    for calib_budget in BUDGETS:
        print("Running budget {}...".format(calib_budget))
        row_lsmc = evaluate_budget_lsmc(
            lsmc_sampler=lsmc_sampler,
            alm_nested_framework=alm_nested_framework,
            calib_budget=calib_budget,
            n_trials=N_TRIALS,
            ref_quantile=REF_QUANTILE,
            quantile_level=QUANTILE_LEVEL,
            n_quantile_samples=N_QUANTILE_SAMPLES,
            seed=SEED+392,
            max_terms=MAX_TERMS,
            output_poly_dir=OUTPUT_POLY_DIR,
            use_cpu_linalg=LSMC_CPU_LINALG,
            stock_lower_bound=stock_lower_bound,
            stock_upper_bound=stock_upper_bound
        )
        rows_lsmc.append(row_lsmc)

        print(
            "LSMC:  Quantile RMSE = {:.6f} ({:.4f}%), Bias = {:.6f}, Variance = {:.6f}".format(
                row_lsmc["quantile_rmse"],
                row_lsmc["quantile_rmse_pct"],
                row_lsmc["quantile_bias"],
                row_lsmc["quantile_variance"],
            )
        )
        print(
            "LSMC: Probability RMSE = {:.6f} ({:.4f}%), Bias = {:.6f}, Variance = {:.6f}".format(
                row_lsmc["probability_rmse"],
                row_lsmc["probability_rmse_pct"],
                row_lsmc["probability_bias"],
                row_lsmc["probability_variance"],
            )
        )

        row_ml2r = evaluate_budget_ml2r(alm_nested_framework, calib_budget, N_TRIALS, REF_QUANTILE, QUANTILE_LEVEL, N_QUANTILE_SAMPLES, SEED+392)
        rows_ml2r.append(row_ml2r)

        print(
            "ML2R:  Quantile RMSE = {:.6f} ({:.4f}%), Bias = {:.6f}, Variance = {:.6f}".format(
                row_ml2r["quantile_rmse"],
                row_ml2r["quantile_rmse_pct"],
                row_ml2r["quantile_bias"],
                row_ml2r["quantile_variance"],
            )
        )
        print(
            "ML2R: Probability RMSE = {:.6f} ({:.4f}%), Bias = {:.6f}, Variance = {:.6f}".format(
                row_ml2r["probability_rmse"],
                row_ml2r["probability_rmse_pct"],
                row_ml2r["probability_bias"],
                row_ml2r["probability_variance"],
            )
        )
    df_results_lsmc = pd.DataFrame(rows_lsmc)
    df_results_lsmc = df_results_lsmc.sort_values("calib_budget_requested").reset_index(drop=True)
    df_results_lsmc.to_csv(OUTPUT_CSV, index=False)
    print("LSMC results exported to {}".format(OUTPUT_CSV))

    df_results_ml2r = pd.DataFrame(rows_ml2r)
    df_results_ml2r = df_results_ml2r.sort_values("calib_budget_requested").reset_index(drop=True)
    df_results_ml2r.to_csv(OUTPUT_DIR / "ml2r.csv", index=False)
    print("ML2R results exported to {}".format(OUTPUT_DIR / "ml2r.csv"))

    # Final log-log RMSE curves with confidence intervals.
    fig, (ax_proba, ax_quantile) = plt.subplots(1, 2, figsize=(12, 6))

    x_lsmc = df_results_lsmc["calib_budget_requested"].to_numpy(dtype=float)
    x_ml2r = df_results_ml2r["calib_budget_requested"].to_numpy(dtype=float)

    proba_lsmc = df_results_lsmc["probability_rmse"].to_numpy(dtype=float)
    proba_lsmc_low = np.maximum(df_results_lsmc["probability_rmse_lower"].to_numpy(dtype=float), 1e-16)
    proba_lsmc_up = np.maximum(df_results_lsmc["probability_rmse_upper"].to_numpy(dtype=float), 1e-16)
    proba_ml2r = df_results_ml2r["probability_rmse"].to_numpy(dtype=float)
    proba_ml2r_low = np.maximum(df_results_ml2r["probability_rmse_lower"].to_numpy(dtype=float), 1e-16)
    proba_ml2r_up = np.maximum(df_results_ml2r["probability_rmse_upper"].to_numpy(dtype=float), 1e-16)

    quant_lsmc = df_results_lsmc["quantile_rmse"].to_numpy(dtype=float)
    quant_lsmc_low = np.maximum(df_results_lsmc["quantile_rmse_lower"].to_numpy(dtype=float), 1e-16)
    quant_lsmc_up = np.maximum(df_results_lsmc["quantile_rmse_upper"].to_numpy(dtype=float), 1e-16)
    quant_ml2r = df_results_ml2r["quantile_rmse"].to_numpy(dtype=float)
    quant_ml2r_low = np.maximum(df_results_ml2r["quantile_rmse_lower"].to_numpy(dtype=float), 1e-16)
    quant_ml2r_up = np.maximum(df_results_ml2r["quantile_rmse_upper"].to_numpy(dtype=float), 1e-16)

    ax_proba.plot(x_lsmc, proba_lsmc, marker="o", label="LSMC")
    ax_proba.fill_between(x_lsmc, proba_lsmc_low, proba_lsmc_up, alpha=0.2)
    ax_proba.plot(x_ml2r, proba_ml2r, marker="o", label="Optimized ML2R")
    ax_proba.fill_between(x_ml2r, proba_ml2r_low, proba_ml2r_up, alpha=0.2)
    ax_proba.set_title("C.D.F evaluation")
    ax_proba.set_xscale("log")
    ax_proba.set_yscale("log")
    ax_proba.set_xlabel("Computational Cost")
    ax_proba.set_ylabel("Empirical RMSE")
    ax_proba.grid()
    ax_proba.legend()

    ax_quantile.plot(x_lsmc, quant_lsmc, marker="o", label="LSMC")
    ax_quantile.fill_between(x_lsmc, quant_lsmc_low, quant_lsmc_up, alpha=0.2)
    ax_quantile.plot(x_ml2r, quant_ml2r, marker="o", label="Optimized ML2R")
    ax_quantile.fill_between(x_ml2r, quant_ml2r_low, quant_ml2r_up, alpha=0.2)
    ax_quantile.set_title("Quantile evaluation")
    ax_quantile.set_xscale("log")
    ax_quantile.set_yscale("log")
    ax_quantile.set_xlabel("Computational Cost")
    ax_quantile.set_ylabel("Empirical RMSE")
    ax_quantile.grid()
    ax_quantile.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_FINAL_PLOT)
    print("Final RMSE plot exported to {}".format(OUTPUT_FINAL_PLOT))


if __name__ == "__main__":
    main()