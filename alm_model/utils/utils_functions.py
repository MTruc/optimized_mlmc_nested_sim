import typing
import json
import cupy as cp
from pathlib import Path

def compute_batch_sizes(n_total : int,
                        n_max_parallel : int) -> int:
    sizes = []
    for i in range(0, n_total, n_max_parallel):
        batch_size = min(n_max_parallel, n_total - i)
        sizes.append(batch_size)

    return sizes

def compute_batch_sizes_mlmc(n_totals,
                            n_max_parallels):
    sizes_level = []
    for i in range(len(n_totals)):

        sizes_level.append(compute_batch_sizes(n_totals[i], n_max_parallels[i]))

    return sizes_level

def load_json_file(json_file : Path) -> dict:

    with open(json_file, "r") as jsonfile:

        params = json.loads(jsonfile.read())

    return params