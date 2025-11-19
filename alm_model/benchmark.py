import typing
import pandas as pd
import numpy as np
import math
import numpy.typing as npt
from pathlib import Path
import alm_model.utils as utils

class BenchmarkResult:

    def __init__(self : typing.Self):    
        self.field_names = ["var", "abs_bias", "rmse", "rmse_upper", "rmse_lower"]
        
        for name in self.field_names:

            setattr(self, name, {})
    
    def add_entry(self : typing.Self,
                  idx : typing.Hashable,
                  values : dict[str, float]) -> None:

        for name in self.field_names:

            d = getattr(self, name)
            d[idx] = values[name]

    def to_dataframe(self : typing.Self) -> pd.DataFrame:

        return pd.DataFrame({name : getattr(self, name) for name in self.field_names})
    
class BenchmarkAnalyst:
    
    def __init__(self : typing.Self, name : str, target : float):
        
        self.name = name
        self.target = target
        self.bench_res = BenchmarkResult()
    
    def process_estimations(self : typing.Self,
                            idx : typing.Hashable,
                            estimations : npt.ArrayLike) -> None:

        target = self.target
        d = {}
        d["var"] = np.var(estimations)
        d["abs_bias"] = np.abs(np.mean(estimations - target))
        squared_error = np.pow(estimations - target, 2)
        mse = np.mean(squared_error)
        d["rmse"] = np.sqrt(mse)

        ci_mse = 1.96 * np.std(squared_error) / math.sqrt(len(squared_error))
        d["rmse_upper"] = np.sqrt(mse + ci_mse)
        d["rmse_lower"] = np.sqrt(max(mse - ci_mse, 0))

        self.bench_res.add_entry(idx, d)
        
    def export_res(self : typing.Self,
                   output_folder : Path) -> pd.DataFrame:
        
        output_file = output_folder / "{}_benchmark.csv".format(self.name)
        return self.bench_res.to_dataframe().to_csv(output_file, sep=";")
    
def analyze_cdf_quantile_benchmark(files : list[Path],
                                   output_folder : Path,
                                   target_cdf : float,
                                   target_quantile : float,
                                   indexes : list[float]) -> None:
    
    cdf_bench_anlyst = BenchmarkAnalyst("cdf", target_cdf)
    quantile_bench_analyst = BenchmarkAnalyst("quantile", target_quantile)
    analysts = [cdf_bench_anlyst, quantile_bench_analyst]
    data_labels = ["cdf_eval_samples", "quantile_eval_samples"]

    for i in range(len(files)):

        data = np.load(files[i])
        idx = indexes[i]
        
        for i in range(len(analysts)):
            analyst = analysts[i]
            label = data_labels[i]
            samples = data[label]
            analyst.process_estimations(idx, samples)

    for analyst in analysts:
        analyst.export_res(output_folder)
        
def analyze_cdf_quantile_benchmark_from_json(json_file : Path, output_folder : Path) -> None:
    
    params = utils.load_json_file(json_file)

    files = params["files"]
    target_cdf = params["target_cdf"]
    target_quantile = params["target_quantile"]
    epsilons = params["epsilon"]
    
    analyze_cdf_quantile_benchmark(files,
                                   output_folder,
                                   target_cdf,
                                   target_quantile,
                                   epsilons)