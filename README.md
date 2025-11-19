# Numerical expriments : Optimized Multi-level Monte Carlo and Antithetic sampling for nested simulations

This is the repository associated to the numerical experiment of the paper Boumezoued et. al. (2025) available at https://arxiv.org/abs/2510.18995. The objective of the numerical experiment is to compare the performance of the new multi-level parametrization of traditional Multi-level Monte Carlo and Weighted Multi-level Monte Carlo presented in the paper to the state of the art.

**Warning :**
The experiment in the paper are relatively heavy in computations, by default the files are configured to run on a machine equipped with a NVDIA GPU. Depending on the machine hardware the scripts may take several hours or days to run.

## Getting Started
The main scripts to rerun the experiment are :
- `./scripts/of_loss.py`
- `./preprocessing.py`
- `./scripts/main.py`

generally exectued in that order. We will get into the specifics of each scripts in later sections. In order to run the scripts we recommend using a `conda` environement with accessibility to the `conda-forge` channel. The depencies are as follows :

dependencies:
  - python=3.12
  - numpy
  - scipy
  - matplotlib
  - pandas
  - ipykernel
  - cuda-nvcc
  - cuda-version=12.8
  - cupy

They can be found in the `environment.yml` file. The environement can be installed using the following command in a terminal :

```
conda env create -f environment.yml
```

**Warning :**
The environement file is configured for a machine using cuda 12.8, please modify the version of cuda in `environment.yml` with the appropriate one for your machine. You can check the cuda version available on your machine with the following command in a terminal :

```
 nvidia-smi
```

## Script `of_loss.py`

This script produces the density of $L_1$ and plot the function $\psi$ (**Figure 1** in the paper). Parameters of the ALM model are located in `./inputs/stock_parameters.json` and `./inputs/alm_parameters.json`. The outputs can be found in `./outputs/`.

## Script `preprocessing.py`

This script produces estimates of structural constants of the model (**Figure 2** in the paper). Parameters of the preprocessing are located in `./inputs/preprocessing_parameters.json`. The ouputs can be found in `./outputs/all_structural_consts`.

## Script `main.py`

This script produces the main numerical experiments, namely the benchmark and comparison of all estimators (**Figure 3** and **Figure 4** in the paper). Parameters of the experiments can be found in `./inputs/main_parameters.json`. The ouputs of each individual benchmark can be found in their respective folder in ``./outputs``.

**Warning :**
By default in `./inputs/main_parameters.json` all the different benchmarks are enabled in the file, this can be pretty long to run. You can easily deactivate benchmarks you are not interested in in this file.