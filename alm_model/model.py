import alm_model.utils as utils
import alm_model.stock as stock
import alm_model.alm as alm
import typing
import numpy as np
import scipy.stats as stats
import math
import numpy.typing as npt

class Model:

    def __init__(self : typing.Self, stock_parameters :stock.StockParameters, alm_parameters : alm.AlmParameters):

        self.stock_parameters = stock_parameters
        self.alm_parameters = alm_parameters
        self.stock_model = stock.StockModel(stock_parameters)
        self.alm_model = alm.AlmModel(alm_parameters)

        stock_paths = np.array([[self.stock_parameters.s0]], dtype=np.float64)
        mr, phi = self.alm_model.project_mr_phi(stock_paths)
        mr = mr[0]
        phi = phi[0]
        self.own_fund_0 = self.portfolio_value(0, mr, phi, self.stock_parameters.s0)

    def portfolio_value(self : typing.Self, t : int, mr_t : float, phi_t : float, s_t : float) -> float:
        
        p = self.alm_parameters.exit_rate
        T = self.alm_parameters.horizon
        r_g = self.alm_parameters.min_guaranteed_rate
        gamma = self.alm_parameters.ps_rate
        sigma = self.stock_parameters.sigma
        r = self.stock_parameters.r

        d = (r - 0.5 * sigma**2 - r_g / gamma) / sigma
        z = 1 + r_g + gamma * sigma * (stats.norm.pdf(d) + d * stats.norm.cdf(d))

        us = np.arange(t+1, T)
        acc = np.power(1 - p, us - t - 1) * np.power(z, us - t) * np.exp(-r*(us - t))

        return phi_t * s_t - mr_t * (p * acc.sum() + math.exp(-r*(T - t)) * (1 - p)**(T - t - 1) * z**(T - t))
    
    def portfolio_value_cond_stock(self : typing.Self, stock_paths : npt.NDArray[np.float64]):

        t = stock_paths.shape[1] - 1
        mr, phi = self.alm_model.project_mr_phi(stock_paths)

        return self.portfolio_value(t, mr, phi, stock_paths[:, -1])
    
    def own_fund_loss_cond_stock(self : typing.Self, stock_paths : npt.NDArray[np.float64]):

        return self.own_fund_0 - self.portfolio_value_cond_stock(stock_paths)
    
    def own_fund_loss_distribution(self : typing.Self, nb_samples : int, rng : np.random.Generator):

        stock_paths = np.empty((nb_samples, 2), dtype=np.float64)
        s1 = self.stock_model.sample_t1(1, nb_samples, rng)
        stock_paths[:, 0] = self.stock_parameters.s0
        stock_paths[:, 1] = s1
        return self.own_fund_loss_cond_stock(stock_paths)
    
    def quantile_own_fund_loss(self : typing.Self, prob : float):

        s1_quant = self.stock_model.quantile_t1(1, prob)
        stock_paths = np.array([[self.stock_parameters.s0, s1_quant]], dtype=np.float64)
        mr, phi = self.alm_model.project_mr_phi(stock_paths)
        return self.own_fund_loss_cond_stock(stock_paths)[0]