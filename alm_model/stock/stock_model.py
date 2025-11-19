import math
import numpy as np
import numpy.typing as npt
import typing
import scipy.stats as stats
import cupy as cp
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class StockParameters:
    
    #Stock value at inception, must be > 0
    s0 : float

    #Black-Scholes drift of the stock
    mu : float

    #Black-Scholes vol of the stock, must be nonnegative
    sigma : float

    #Black-Scholes risk free rate
    r : float

    def __post_init__(self : typing.Self):

        if self.s0 <= 0.0:
            raise ValueError("Expected s0 to be > 0 (got {}).".format(self.s0))
        
        if self.sigma < 0.0:
            raise ValueError("Expected sigma to be >= 0 (got {}).".format(self.sigma))
        
def load_stock_parameters(file_path : "Path") -> StockParameters:

    with open(file_path, "r") as file:
        loaded_parameters = json.loads(file.read())
    s0 = loaded_parameters["s0"]
    mu = loaded_parameters["mu"]
    sigma = loaded_parameters["sigma"]
    r = loaded_parameters["r"]
    return StockParameters(s0, mu, sigma, r)

class StockModel():

    def __init__(self, parameters : StockParameters):
        
        self.parameters = parameters
    
    def compute_conditional(self : typing.Self, s_previous : npt.NDArray[np.float64], gaussian_samples : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        #Unpacking parameters
        r = self.parameters.r
        s0 = self.parameters.s0
        sigma = self.parameters.sigma
        t1 = s_previous.shape[1]
        T = t1 + gaussian_samples.shape[1]
        N = s_previous.shape[0]

        stock_paths = np.empty((N, T))
        stock_paths[:, :t1] = s_previous
        drift = r - 0.5*sigma**2
        stock_paths[:, t1:] = np.exp(drift + sigma * gaussian_samples)
        stock_paths[:, (t1-1):] = np.cumprod(stock_paths[:, (t1-1):], axis=1)

        return stock_paths
    
    def compute_conditional_gpu(self : typing.Self,
                                s_previous : cp.ndarray,
                                gaussian_samples : cp.ndarray) -> cp.ndarray:

        #Unpacking parameters
        r = self.parameters.r
        s0 = self.parameters.s0
        sigma = self.parameters.sigma
        t1 = s_previous.shape[1]
        T = t1 + gaussian_samples.shape[1]
        N = s_previous.shape[0]

        stock_paths = cp.empty((N, T))
        stock_paths[:, :t1] = s_previous
        drift = r - 0.5*sigma**2
        stock_paths[:, t1:] = cp.exp(drift + sigma * gaussian_samples)
        stock_paths[:, (t1-1):] = cp.cumprod(stock_paths[:, (t1-1):], axis=1)

        return stock_paths

    def sample_conditional(self : typing.Self, s_previous : npt.NDArray[np.float64], dt : int, rng : np.random.Generator) -> npt.NDArray[np.float64]:

        return self.compute_conditional(s_previous, rng.standard_normal((s_previous.shape[0], dt), dtype=np.float64))
    
    def compute_t1(self, t1 : int, gaussian_sample : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        #Unpacking parameters
        mu = self.parameters.mu
        s0 = self.parameters.s0
        sigma = self.parameters.sigma

        #Compute useful constant
        drift = mu - 0.5 * sigma * sigma
        
        return s0 * np.exp(drift * t1 + math.sqrt(t1) * sigma * gaussian_sample)

    def compute_t1_gpu(self, t1 : int, gaussian_sample : cp.ndarray) -> cp.ndarray:

        #Unpacking parameters
        mu = self.parameters.mu
        s0 = self.parameters.s0
        sigma = self.parameters.sigma

        #Compute useful constant
        drift = mu - 0.5 * sigma * sigma
        
        return s0 * cp.exp(drift * t1 + math.sqrt(t1) * sigma * gaussian_sample)

    def sample_t1(self, t1 : int, N : int, rng : np.random.Generator) -> npt.NDArray[np.float64]:
        
        return self.compute_t1(t1, rng.standard_normal(N))
    
    def quantile_t1(self : typing.Self, t1 : int, prob : float) -> float:

        mu = self.parameters.mu
        s0 = self.parameters.s0
        sigma = self.parameters.sigma

        drift = mu - 0.5 * sigma * sigma
        q = stats.norm.ppf(prob)

        return s0 * np.exp(drift * t1 + math.sqrt(t1) * sigma * q)