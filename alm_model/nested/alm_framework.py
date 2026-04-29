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
    
    def compute_risk_factors(self : typing.Self,
                             gaussian_samples_outer : np.ndarray | cp.ndarray
                             ) -> np.ndarray | cp.ndarray:
        
        t1 = 1
        return self.model.stock_model.compute_t1(t1, gaussian_samples_outer)
    
    def get_risk_factors_name(self : typing.Self):
        
        return ["S1"]
    
    def sample_risk_factors(self : typing.Self,
                            J : int,
                            rng : np.random.Generator | cp.random.BitGenerator
                            ) -> np.ndarray | cp.ndarray:
        
        gaussian_samples_outer = rng.standard_normal(J)
        return self.compute_risk_factors(gaussian_samples_outer).reshape(-1, 1)
    
    def compute_E2_antithetic(self : typing.Self,
                              risk_factors : np.ndarray | cp.ndarray,
                              gaussian_samples : np.ndarray | cp.ndarray):
        J = risk_factors.shape[0]
        T = self.model.alm_parameters.horizon
        xp = cp.get_array_module(risk_factors)
        gaussian_samples_inners = xp.empty((J, 2, T - 1))
        gaussian_samples_inners[:, 0, :] = gaussian_samples
        gaussian_samples_inners[:, 1, :] = -gaussian_samples
        F_mat = self.compute_F_mat_from_risk_factors(risk_factors,
                                                    gaussian_samples_inners)
        
        return xp.mean(F_mat, axis=1)
    
    def sample_E2_antithetic(self : typing.Self,
                             risk_factors : np.ndarray | cp.ndarray,
                             rng : np.random.Generator | cp.random.BitGenerator
                             ) -> np.ndarray | cp.ndarray:
        J = len(risk_factors)
        T = self.model.alm_parameters.horizon - 1
        gaussian_samples = rng.standard_normal((J, T))
        return self.compute_E2_antithetic(risk_factors,
                                          gaussian_samples)
    
    def sample_Y2_antithetic(self : typing.Self,
                            risk_factors : np.ndarray | cp.ndarray,
                            rng : np.random.Generator | cp.random.BitGenerator
                            )-> np.ndarray | cp.ndarray:
        
        J = len(risk_factors)
        T = self.model.alm_parameters.horizon - 1
        gaussian_samples = rng.standard_normal((J, T))
        xp = cp.get_array_module(risk_factors)
        gaussian_samples_inners = xp.empty((J, 2, T))
        gaussian_samples_inners[:, 0, :] = gaussian_samples
        gaussian_samples_inners[:, 1, :] = -gaussian_samples
        F_mat = self.compute_F_mat_from_risk_factors(risk_factors,
                                                    gaussian_samples_inners)
        
        return F_mat
    
    def compute_F_mat_from_risk_factors(self : typing.Self,
                                        risk_factors : np.ndarray | cp.ndarray,
                                        gaussian_samples_inners : np.ndarray | cp.ndarray
                                        ) -> np.ndarray | cp.ndarray:
        """
        Computes the matrix of projected future own funds (F-matrix) given a set of outer risk factor scenarios
        and inner Gaussian samples, for use in nested simulation frameworks.

        The function operates as follows:
        - For each outer scenario (risk_factors : stock prices at t=1), constructs stock price paths.
        - Projects mathematical reserve (mr) and number of stock shares (phi) parameters for the ALM model.
        - For each inner scenario (gaussian_samples_inners), simulates conditional stock price paths.
        - Computes the projected terminal value of assets.
        - Returns the discounted difference between initial own funds and projected terminal asset values,
        reshaped into a matrix of shape (num_outer_scenarios, num_inner_scenarios).

        Args:
            self (typing.Self): Instance of the class.
            risk_factors (np.ndarray | cp.ndarray): Array of shape (num_outer_scenarios, num_risk_factors)
                representing sampled outer risk factor scenarios (stock prices at t=1).
            gaussian_samples_inners (np.ndarray | cp.ndarray): Array of shape (num_outer_scenarios, num_inner_scenarios, projection_horizon-1)
                representing conditional inner simulations (e.g., Gaussian noise for stock model paths).

        Returns:
            np.ndarray | cp.ndarray: Array of shape (num_outer_scenarios, num_inner_scenarios) containing
                discounted projected own funds for each outer-inner scenario pair.
        """
        
        t1 = 1
        r = self.model.stock_parameters.r
        of_0 = self.model.own_fund_0
        T = self.model.alm_parameters.horizon - 1
        xp = cp.get_array_module(risk_factors)
        
        outer_stock_paths = xp.empty((risk_factors.shape[0], 2), dtype=xp.float64)
        outer_stock_paths[:, 0] = self.model.stock_parameters.s0
        outer_stock_paths[:, 1] = risk_factors[:, 0]
        
        mr, phi = self.model.alm_model.project_mr_phi(outer_stock_paths)
        mr = mr.repeat(gaussian_samples_inners.shape[1])
        phi = phi.repeat(gaussian_samples_inners.shape[1])

        outer_stock_paths = outer_stock_paths[:, 1].repeat(gaussian_samples_inners.shape[1]).reshape(-1, 1)
        inner_stock_paths = self.model.stock_model.compute_conditional(outer_stock_paths, gaussian_samples_inners.reshape(-1, gaussian_samples_inners.shape[2]))
        terminal_value = self.model.alm_model.project_terminal_assets(t1, phi, mr, inner_stock_paths)
        
        return (of_0 - xp.exp( -r * T) * terminal_value).reshape((gaussian_samples_inners.shape[0], gaussian_samples_inners.shape[1]))
    

    def sample_F_mat_from_risk_factors(self : typing.Self,
                                       K : int,
                                       risk_factors : np.ndarray | cp.ndarray,
                                       rng : np.random.Generator | cp.random.BitGenerator
                                       )-> np.ndarray | cp.ndarray:
        
        J = len(risk_factors)
        T = self.model.alm_parameters.horizon - 1
        gaussian_samples_inners = rng.standard_normal((J, K, T))
        F_mat = self.compute_F_mat_from_risk_factors(risk_factors,
                                                    gaussian_samples_inners)
        
        return F_mat
    


def load_nested_alm_framework(nested_framework_parameters_file,
                              alm_parameters_file,
                              stock_parameters_file):
    
    framework_parameters = load_nested_framework_parameters(nested_framework_parameters_file)
    params_alm = load_alm_parameters(alm_parameters_file)
    params_stock = load_stock_parameters(stock_parameters_file)
    
    return AlmFramework(framework_parameters,
                        params_alm,
                        params_stock)