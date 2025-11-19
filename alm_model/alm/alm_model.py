import typing
import alm_model.utils as utils
import numpy as np
import numpy.typing as npt
import cupy as cp
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class AlmParameters():

    #Horizon of ALM portfolio, must be greater or equal to 1
    horizon : int

    #Minimum guaranteed rate served to policholders each year
    min_guaranteed_rate : float

    #Profit sharing rate served to policyholders each year
    ps_rate : float

    #Percentage of policyholders exiting their contract each year, must be nonnegative
    exit_rate : float

    #Mathematical reserve at inception of the portfolio, must be nonnegative
    mr_0 : float

    def __post_init__(self : typing.Self):

        #Check that horizon is greater or equal to 1
        if self.horizon < 1:
            raise ValueError("Expected horizon to be greater or equal to 1 (got {})".format(self.horizon))
        
        #Check that exit rate is nonnegative
        if self.exit_rate < 0:
            raise ValueError("Expected exit_rate to be nonnegative (got {})".format(self.exit_rate))
        
        #Check that mr_0 is nonnegative
        if self.mr_0 < 0:
            raise ValueError("Expected mr_0 to be nonnegative (got {})".format(self.mr_0))

def load_alm_parameters(path : Path) -> AlmParameters:
    
    with open(path, "r") as file:
        loaded_params = json.loads(file.read())

    horizon = loaded_params["horizon"]
    min_guaranteed_rate = loaded_params["min_guaranteed_rate"]
    ps_rate = loaded_params["ps_rate"]
    exit_rate = loaded_params["exit_rate"]
    mr_0 = loaded_params["mr_0"]

    return AlmParameters(horizon,
                         min_guaranteed_rate,
                         ps_rate,
                         exit_rate,
                         mr_0)

class AlmModel:

    def __init__(self, parameters : AlmParameters):

        self.parameters = parameters

    def project_terminal_assets(self : typing.Self, t : int, curr_phi : float, curr_mr : float, stock_paths : npt.NDArray[np.float64]):

        rg = self.parameters.min_guaranteed_rate
        gamma = self.parameters.ps_rate
        p = self.parameters.exit_rate
        T = self.parameters.horizon
        phi = curr_phi
        mr = curr_mr

        #From t+1  to T - 1
        for i in range(t+1, T):
            dt = i - t
            rs = np.maximum(gamma * np.log(stock_paths[:, dt] / stock_paths[:, dt-1]), rg)
            tilde_mr = mr * (1 + rs)
            delta_phi = tilde_mr * p / stock_paths[:, dt]
            phi = phi - delta_phi
            mr = tilde_mr * (1 - p)
        
        #Terminal time t = T
        rs = np.maximum(gamma * np.log(stock_paths[:, T-t] / stock_paths[:, T-t-1]), rg)
        tilde_mr = mr * (1 + rs)
        delta_phi = tilde_mr / stock_paths[:, T-t]
        phi = phi - delta_phi

        return phi * stock_paths[:, T-t]
    
    def project_terminal_assets_gpu(self : typing.Self,
                                    t : int,
                                    curr_phi : float,
                                    curr_mr : float,
                                    stock_paths : cp.ndarray) -> cp.ndarray:

        rg = self.parameters.min_guaranteed_rate
        gamma = self.parameters.ps_rate
        p = self.parameters.exit_rate
        T = self.parameters.horizon
        phi = curr_phi
        mr = curr_mr

        #From t+1  to T - 1
        for i in range(t+1, T):
            dt = i - t
            rs = cp.maximum(gamma * cp.log(stock_paths[:, dt] / stock_paths[:, dt-1]), rg)
            tilde_mr = mr * (1 + rs)
            delta_phi = tilde_mr * p / stock_paths[:, dt]
            phi = phi - delta_phi
            mr = tilde_mr * (1 - p)
        
        #Terminal time t = T
        rs = cp.maximum(gamma * cp.log(stock_paths[:, T-t] / stock_paths[:, T-t-1]), rg)
        tilde_mr = mr * (1 + rs)
        delta_phi = tilde_mr / stock_paths[:, T-t]
        phi = phi - delta_phi

        return phi * stock_paths[:, T-t]
    
    def project_mr_phi(self : typing.Self, stock_paths : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        
        final_time = stock_paths.shape[1]
        rg = self.parameters.min_guaranteed_rate
        gamma = self.parameters.ps_rate
        p = self.parameters.exit_rate
        T = self.parameters.horizon
        mr = np.full(shape=(stock_paths.shape[0]), fill_value=self.parameters.mr_0)
        phi = mr / stock_paths[:, 0]

        for i in range(1, final_time):

            rs = np.maximum(gamma * np.log(stock_paths[:, i] / stock_paths[:, i-1]), rg)
            tilde_mr = mr * (1 + rs)
            delta_phi = tilde_mr * p / stock_paths[:, i]
            phi = phi - delta_phi
            mr = tilde_mr * (1 - p)

        return mr, phi
    
    def project_mr_phi_gpu(self : typing.Self,
                           stock_paths : cp.ndarray) -> cp.ndarray:
        
        final_time = stock_paths.shape[1]
        rg = self.parameters.min_guaranteed_rate
        gamma = self.parameters.ps_rate
        p = self.parameters.exit_rate
        T = self.parameters.horizon
        mr = cp.full(shape=(stock_paths.shape[0]), fill_value=self.parameters.mr_0)
        phi = mr / stock_paths[:, 0]

        for i in range(1, final_time):

            rs = cp.maximum(gamma * np.log(stock_paths[:, i] / stock_paths[:, i-1]), rg)
            tilde_mr = mr * (1 + rs)
            delta_phi = tilde_mr * p / stock_paths[:, i]
            phi = phi - delta_phi
            mr = tilde_mr * (1 - p)

        return mr, phi