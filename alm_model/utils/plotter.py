import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_estimator(theoretical_benchmark : pd.DataFrame,
                   empirical_benchmark : pd.DataFrame,
                   output_folder : Path):
    
    index = theoretical_benchmark.index
    
    #Plot for var c.d.f
    plt.clf()
    plt.title("Evalauating c.d.f")
    plt.plot(index, empirical_benchmark["cdf_emp_var"], marker="o", label="Empirical Variance")
    plt.plot(index, theoretical_benchmark["theo_var"], marker="o", linestyle="--", label="Theoretical Variance")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Target RMSE  $\epsilon$")
    plt.grid()
    plt.legend()
    plt.savefig(Path(output_folder / "cdf_variance.png"), bbox_inches='tight')
    
    #Plot for var quantile
    plt.clf()
    plt.title("Evaluating quantile")
    plt.plot(index, empirical_benchmark["quantile_emp_var"], marker="o", label="Empirical Variance")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Target RMSE  $\epsilon$")
    plt.grid()
    plt.legend()
    plt.savefig(Path(output_folder / "quantile_variance.png"), bbox_inches='tight')
    
    #Plot abs bias for c.d.f
    plt.clf()
    plt.title("Evalauating c.d.f")
    plt.plot(index, empirical_benchmark["abs_cdf_emp_bias"], marker="o", label="Empirical Absolute Bias")
    plt.plot(index, theoretical_benchmark["abs_theo_bias"], marker="o", linestyle="--", label="Theoretical Absolute Bias")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Target RMSE  $\epsilon$")
    plt.grid()
    plt.legend()
    plt.savefig(Path(output_folder / "cdf_bias.png"), bbox_inches='tight')
    
    #Plot abs bias for quantile
    plt.clf()
    plt.title("Evalauating quantile")
    plt.plot(index, empirical_benchmark["abs_quantile_emp_bias"], marker="o", label="Empirical Absolute Bias")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Target RMSE  $\epsilon$")
    plt.grid()
    plt.legend()
    plt.savefig(Path(output_folder / "quantile_bias.png"), bbox_inches='tight')

    #Plot cdf rmse
    plt.clf()
    plt.title("Evalauating c.d.f")
    plt.plot(index, empirical_benchmark["cdf_emp_rmse"], marker="o", label="Empirical RMSE")
    plt.fill_between(index,
                     empirical_benchmark["cdf_emp_rmse_upper"],
                     empirical_benchmark["cdf_emp_rmse_lower"],
                     alpha = 0.2)
    plt.plot(index, theoretical_benchmark["theo_rmse"], marker="o", linestyle="--", label="Theoretical RMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Target RMSE  $\epsilon$")
    plt.grid()
    plt.legend()
    plt.savefig(Path(output_folder / "cdf_rmse.png"), bbox_inches='tight')
    
    #Plot quantime rmse
    plt.clf()
    plt.title("Evalauating quantile")
    plt.plot(index, empirical_benchmark["quantile_emp_rmse"], marker="o", label="Empirical RMSE")
    plt.fill_between(index,
                     empirical_benchmark["quantile_emp_rmse_upper"],
                     empirical_benchmark["quantile_emp_rmse_lower"],
                     alpha = 0.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Target RMSE  $\epsilon$")
    plt.grid()
    plt.legend()
    plt.savefig(Path(output_folder / "quantile_rmse.png"), bbox_inches='tight')


    #Plot complexity c.d.f
    plt.clf()
    plt.title("Evalauating c.d.f")
    plt.plot(empirical_benchmark["cdf_emp_rmse"], theoretical_benchmark["cost"], marker="o", label="Empirical Complexity")
    plt.fill_betweenx(theoretical_benchmark["cost"],
                      empirical_benchmark["cdf_emp_rmse_lower"],
                      empirical_benchmark["cdf_emp_rmse_upper"],
                      alpha=0.2)
    plt.plot(theoretical_benchmark["theo_rmse"], theoretical_benchmark["cost"], marker="o", linestyle="--", label="Theoretical Complexity")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("RMSE")
    plt.ylabel("Cost")
    plt.grid()
    plt.legend()
    plt.savefig(Path(output_folder / "cdf_complexity.png"), bbox_inches='tight')
    
    #Plot complexity quantile
    plt.clf()
    plt.title("Evalauating quantile")
    plt.plot(empirical_benchmark["quantile_emp_rmse"], theoretical_benchmark["cost"], marker="o", label="Empirical Complexity")
    plt.fill_betweenx(theoretical_benchmark["cost"],
                      empirical_benchmark["quantile_emp_rmse_lower"],
                      empirical_benchmark["quantile_emp_rmse_upper"],
                      alpha=0.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("RMSE")
    plt.ylabel("Cost")
    plt.grid()
    plt.legend()
    plt.savefig(Path(output_folder / "quantile_complexity.png"), bbox_inches='tight')