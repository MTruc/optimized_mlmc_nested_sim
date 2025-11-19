from pathlib import Path
import json
from alm_model.alm import (load_alm_parameters)
from alm_model.nested import (AlmFramework,
                              StructuralConstantsWERVar,
                              load_structural_constants,
                              load_nested_framework_parameters)
from alm_model.stock import load_stock_parameters

def load_all_parameters(parent_folder : Path) -> dict[str, any]:

    framework_parameters_file = parent_folder / "alm_nested_framework_params.json"
    framework_parameters = load_nested_framework_parameters(framework_parameters_file)

    params_alm_file = parent_folder / "alm_parameters.json"
    params_alm = load_alm_parameters(params_alm_file)

    params_stock_file = parent_folder / "stock_parameters.json"
    params_stock = load_stock_parameters(params_stock_file)

    structural_consts_file = parent_folder / "structural_constants.json"
    structural_consts = load_structural_constants(structural_consts_file)
    
    return {
        "framework_parameters": framework_parameters,
        "params_alm": params_alm,
        "params_stock": params_stock,
        "structural_consts": structural_consts,
    }

def initialize_framework(parent_folder : Path) -> tuple[AlmFramework, StructuralConstantsWERVar]:

    params = load_all_parameters(parent_folder)
    nested_framework = AlmFramework(
        params["framework_parameters"],
        params["params_alm"],
        params["params_stock"]
    )

    return nested_framework, params["structural_consts"]