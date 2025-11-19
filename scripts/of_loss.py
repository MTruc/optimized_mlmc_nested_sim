import sys
sys.path.append("./")

import alm_model.model as mod
import alm_model.stock as stock
import alm_model.alm as alm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

stock_parameters_file = Path("./inputs/stock_parameters.json")
stock_params = stock.load_stock_parameters(stock_parameters_file)
alm_parameters_file = Path("./inputs/alm_parameters.json")
alm_params = alm.load_alm_parameters(alm_parameters_file)

model = mod.Model(stock_params, alm_params)

seed = 9832
nb_samples = 1_000_000
rng = np.random.default_rng(seed)
samples = model.own_fund_loss_distribution(nb_samples, rng)
np.save("./outputs/of_loss_samples.npy", samples)
print(samples)

model = mod.Model(stock_params, alm_params)
s1 = np.linspace(40, 180, 1000)
stock_path = np.empty((1000, 2), dtype=np.float64)
stock_path[:, 0] = stock_params.s0
stock_path[:, 1] = s1
loss = model.own_fund_loss_cond_stock(stock_path)
np.savez("./outputs/own_fund_loss_cond.npz", loss=loss, s1=s1)
print(loss)

loaded_array = np.load("./outputs/own_fund_loss_cond.npz")
loss = loaded_array["loss"]
s1 = loaded_array["s1"]

plt.clf()
plt.plot(s1, loss)
plt.xlabel("Stock value at t=1")
plt.ylabel("Own-fund loss")
plt.grid()
plt.savefig("./outputs/own_fund_loss_cond.png")

samples = np.load("./outputs/of_loss_samples.npy")
kde = gaussian_kde(samples)

xmin, xmax = -200, 300
xgrid = np.linspace(xmin, xmax, 1000)
kde_values = kde(xgrid)

plt.plot(xgrid, kde_values, label='Gaussian Kernel Density Estimation')
plt.hist(samples, bins=30, density=True, alpha=0.5, label='Histogram', range=(xmin, xmax))
plt.xlabel('Own fund loss')
plt.ylabel('Density')
plt.legend()
plt.savefig("./outputs/of_loss_distrib_plot.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))

ax1.plot(xgrid, kde_values, label='Gaussian Kernel Density Estimation')
ax1.hist(samples, bins=30, density=True, alpha=0.5, label='Histogram', range=(xmin, xmax))
ax1.set_xlabel('Own fund loss')
ax1.set_ylabel('Density')
ax1.legend()

ax2.plot(s1, loss)
ax2.set_xlabel("Stock value at t=1")
ax2.set_ylabel("Own-fund loss")
ax2.grid()

fig.savefig("./outputs/Figure_1.png")