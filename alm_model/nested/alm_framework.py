from alm_model.nested.nested_framework import NestedFramework, NestedFrameworkParameters, load_nested_framework_parameters
from alm_model.model import Model
from alm_model.stock import StockParameters, load_stock_parameters
from alm_model.alm import AlmParameters, load_alm_parameters
import typing
import numpy as np
import cupy as cp

class AlmFramework(NestedFramework):

    def __init__(self : typing.Self,
                 params_nested : NestedFrameworkParameters,
                 params_alm : AlmParameters,
                 params_stock : StockParameters):
        
        self.model = Model(params_stock, params_alm)
        super().__init__(params_nested)

    def sample_F_matrix(self, K, J, rng):

        gaussian_samples_outers = rng.standard_normal(size=(J,), dtype=np.float64)
        gaussian_samples_inners = rng.standard_normal(size=(J, K, self.model.alm_parameters.horizon - 1), dtype=np.float64)
        
        return self.compute_F_matrix(gaussian_samples_outers, gaussian_samples_inners)
    
    def sample_F_matrix_gpu(self, K, J, rng):

        gaussian_samples_outers = rng.standard_normal(size=(J,), dtype=cp.float64)
        gaussian_samples_inners = rng.standard_normal(size=(J, K, self.model.alm_parameters.horizon - 1), dtype=cp.float64)
        
        return self.compute_F_matrix_gpu(gaussian_samples_outers, gaussian_samples_inners)

    def compute_F_matrix(self, gaussian_samples_outers, gaussian_samples_inners):

        outer_stock_paths = np.empty((gaussian_samples_outers.shape[0], 2), dtype=np.float64)
        outer_stock_paths[:, 0] = self.model.stock_parameters.s0
        t1 = 1
        outer_stock_paths[:, 1] = self.model.stock_model.compute_t1(t1, gaussian_samples_outers)
        mr, phi = self.model.alm_model.project_mr_phi(outer_stock_paths)
        mr = mr.repeat(gaussian_samples_inners.shape[1])
        phi = phi.repeat(gaussian_samples_inners.shape[1])

        outer_stock_paths = outer_stock_paths[:, 1].repeat(gaussian_samples_inners.shape[1]).reshape(-1, 1)
        inner_stock_paths = self.model.stock_model.compute_conditional(outer_stock_paths, gaussian_samples_inners.reshape(-1, gaussian_samples_inners.shape[2]))
        terminal_value = self.model.alm_model.project_terminal_assets(t1, phi, mr, inner_stock_paths)

        r = self.model.stock_parameters.r
        of_0 = self.model.own_fund_0
        T = self.model.alm_parameters.horizon - 1
        return (of_0 - np.exp( -r * T) * terminal_value).reshape((gaussian_samples_inners.shape[0], gaussian_samples_inners.shape[1]))
    
    def compute_F_matrix_gpu(self : typing.Self,
                             gaussian_samples_outers : cp.ndarray,
                             gaussian_samples_inners : cp.ndarray) -> cp.ndarray:
        
        outer_stock_paths = cp.empty((gaussian_samples_outers.shape[0], 2), dtype=cp.float64)
        outer_stock_paths[:, 0] = self.model.stock_parameters.s0
        t1 = 1
        outer_stock_paths[:, 1] = self.model.stock_model.compute_t1_gpu(t1, gaussian_samples_outers)
        mr, phi = self.model.alm_model.project_mr_phi_gpu(outer_stock_paths)
        mr = mr.repeat(gaussian_samples_inners.shape[1])
        phi = phi.repeat(gaussian_samples_inners.shape[1])
        
        outer_stock_paths = outer_stock_paths[:, 1].repeat(gaussian_samples_inners.shape[1]).reshape(-1, 1)
        inner_stock_paths = self.model.stock_model.compute_conditional_gpu(outer_stock_paths, gaussian_samples_inners.reshape(-1, gaussian_samples_inners.shape[2]))
        terminal_value = self.model.alm_model.project_terminal_assets_gpu(t1, phi, mr, inner_stock_paths)

        r = self.model.stock_parameters.r
        of_0 = self.model.own_fund_0
        T = self.model.alm_parameters.horizon - 1
        return (of_0 - np.exp( -r * T) * terminal_value).reshape((gaussian_samples_inners.shape[0], gaussian_samples_inners.shape[1]))
    
def load_nested_alm_framework(nested_framework_parameters_file,
                              alm_parameters_file,
                              stock_parameters_file):
    
    framework_parameters = load_nested_framework_parameters(nested_framework_parameters_file)
    params_alm = load_alm_parameters(alm_parameters_file)
    params_stock = load_stock_parameters(stock_parameters_file)
    
    return AlmFramework(framework_parameters,
                        params_alm,
                        params_stock)