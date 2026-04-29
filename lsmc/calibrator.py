import typing
import numpy as np
import cupy as cp
import lsmc
import math
import pandas as pd

def monomial_design_1d(x, p_max, normalize=True):

    xp = cp.get_array_module(x)
    x_avg = xp.mean(x)
    x_std = xp.std(x)
    if normalize:
        z = (x - x_avg) / (x_std) 
    else:
        z = x

    X = xp.empty((len(x), p_max + 1))
    X[:, 0] = 1.0
    for p in range(1, p_max+1):
        X[:, p] = X[:, p-1] * z

    return X, x_avg, x_std

def forward_stepwise_monomial_1d(x, y, p_max=20, tol=0.0):
    model = select_monomial_degree_1d(x, y, p_max=p_max)

    aic = model["aic"]
    xp = cp.get_array_module(x)
    p = 0
    while p < p_max:
        if aic[p + 1] < aic[p] - tol:
            p += 1
        else:
            break

    X, xmin, xmax = monomial_design_1d(x, p)
    coef = xp.linalg.lstsq(X, y, rcond=None)[0]

    return {
        "degree": p,
        "coef": coef,
        "xmin": xmin,
        "xmax": xmax,
        "aic": aic,
    }

def select_monomial_degree_1d(X_calib, y_calib, p_max, normalize=True):

    xp = cp.get_array_module(X_calib)
    X_design, xmin, xmax = monomial_design_1d(X_calib, p_max, normalize=normalize)
    
    Q, R = xp.linalg.qr(X_design, mode="reduced")
    qty = Q.T @ y_calib
    y_norm2 = y_calib.T @ y_calib
    explained = xp.cumsum(qty**2)
    rss = xp.maximum(y_norm2 - explained, 1e-300)

    degrees = xp.arange(p_max + 1)
    k_params = degrees + 1

    n = len(y_calib)
    aic = n * (xp.log(2.0 * xp.pi * rss / n) + 1.0) + 2.0 * k_params

    p_star = int(xp.argmin(aic))
    # coef = xp.linalg.lstsq(X_design[:, :p_star + 1], y_calib, rcond=None)[0]
    coef = None

    return {
        "degree": p_star,
        "coef": coef,
        "xmin": xmin,
        "xmax": xmax,
        "aic": aic,
        "rss": rss,
    }

def predict_monomial_1d(x_new, model):
    """
    Predicts from the selected monomial polynomial model.
    """

    xp = cp.get_array_module(x_new)
    x_avg = model["xmin"]
    x_std = model["xmax"]
    p = model["degree"]

    z = (x_new - x_avg) / (x_std) 

    X_new = xp.empty((len(x_new), p + 1))
    X_new[:, 0] = 1.0
    for k in range(1, p + 1):
        X_new[:, k] = X_new[:, k - 1] * z

    return X_new @ xp.array(model["coef"])

class LSMCCalibrator1D:

    def __init__(self, X_calib, y_calib, max_nb_terms, var_name="x", normalize_input=False):

        if X_calib.ndim != 1:
            raise ValueError("X_calib must be a 1D array")
        
        if y_calib.ndim != 1:
            raise ValueError("y_calib must be 1D array")
        
        if len(X_calib) != len(y_calib):
            raise ValueError("X_calib and y_calib must have the same length")
        
        self.X_calib = X_calib
        self.y_calib = y_calib
        self.max_nb_terms = max_nb_terms

    def fast_calibrate_proxy(self):

        model = forward_stepwise_monomial_1d(self.X_calib, self.y_calib, p_max=self.max_nb_terms - 1)

        return model

def save_monomial_model_to_csv(model, path):
    """
    Save a 1D monomial polynomial model to CSV.

    Output format:
        ;S1;coef
        0;0;c0
        1;1;c1
        ...
    
    where S1 is the monomial degree.
    """
    coef = model["coef"]
    xp = cp.get_array_module(coef)

    if xp is cp:
        coef = cp.asnumpy(coef)
    degree = model["degree"]

    df = pd.DataFrame({
        "S1": list(range(degree + 1)),
        "coef": coef
    })

    df.to_csv(path, sep=";", index=True)