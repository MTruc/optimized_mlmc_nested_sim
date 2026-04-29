import typing
import alm_model.nested as nested
import numpy as np
import numpy.typing as npt
import cupy as cp
import pandas as pd
import scipy.stats as stats


class LSMCSampler:

    nested_framework: nested.NestedFramework
    rf_names: list[str]

    def __init__(self: typing.Self, nested_framework: nested.NestedFramework):

        self.nested_framework = nested_framework
        self.rf_names = self.nested_framework.get_risk_factors_name()
        self.rf_dim = len(self.rf_names)

    def sample_E2_anti(
        self: typing.Self,
        risk_factors: np.ndarray | cp.ndarray,
        rng: np.random.Generator | cp.random.BitGenerator,
    ) -> np.ndarray | cp.ndarray:

        e2_anti = self.nested_framework.sample_E2_antithetic(risk_factors, rng)
        e2_anti = e2_anti.get() if hasattr(e2_anti, "get") else e2_anti  # type: ignore
        return e2_anti

    def construct_dataset(self: typing.Self, rf_points: np.ndarray | cp.ndarray):
        pass

    def sample_dataset_base(
        self: typing.Self,
        J: int,
        risk_factors: np.ndarray | cp.ndarray,
        rng: np.random.Generator | cp.random.BitGenerator,
        K=2,
        antithetic=True,
        J_max=None,
    ):

        xp = cp.get_array_module(risk_factors)
        rf_name = self.nested_framework.get_risk_factors_name()

        if not isinstance(rng, np.random.Generator):
            risk_factors = cp.asarray(risk_factors)

        # Si J_max est spécifié, traiter par batch
        if J_max is not None and J > J_max:
            data_list = []
            for i in range(0, J, J_max):
                batch_end = min(i + J_max, J)
                batch_size = batch_end - i
                rf_batch = risk_factors[i:batch_end]
                
                # Generate dataset for this batch
                if K == 2 and antithetic:
                    ek_batch = self.sample_E2_anti(rf_batch, rng)
                else:
                    ek_batch = xp.mean(
                        self.nested_framework.sample_F_mat_from_risk_factors(
                            K, rf_batch, rng
                        ),
                        axis=1,
                    )
                
                # Format batch data
                rf_batch_cpu = rf_batch.get() if hasattr(rf_batch, "get") else rf_batch  # type: ignore
                batch_data = np.empty((batch_size, len(rf_name) + 1))
                batch_data[:, : len(rf_name)] = rf_batch_cpu
                batch_data[:, -1] = cp.asnumpy(ek_batch)
                data_list.append(batch_data)
            
            # Concatenate all batches
            data = np.vstack(data_list)
        else:
            # Generate dataset (non-batched)
            if K == 2 and antithetic:
                ek = self.sample_E2_anti(risk_factors, rng)
            else:
                ek = xp.mean(
                    self.nested_framework.sample_F_mat_from_risk_factors(
                        K, risk_factors, rng
                    ),
                    axis=1,
                )

            # Format dataset
            risk_factors = risk_factors.get() if hasattr(risk_factors, "get") else risk_factors  # type: ignore
            data = np.empty((J, len(rf_name) + 1))
            data[:, : len(rf_name)] = risk_factors
            data[:, -1] = cp.asnumpy(ek)
        
        rf_name.append("Y")
        return pd.DataFrame(data, columns=rf_name)

    def sample_dataset(
        self: typing.Self, J: int, rng: np.random.Generator | cp.random.BitGenerator, J_max=None
    ):

        risk_factors = self.nested_framework.sample_risk_factors(J, rng)
        return self.sample_dataset_base(J, risk_factors, rng, J_max=J_max)

    def sample_dataset_sobol(
        self: typing.Self,
        m: int,
        l_bound: npt.NDArray[np.float64],
        u_bound: npt.NDArray[np.float64],
        rng: np.random.Generator | cp.random.Generator,
        K=2,
        antithetic=True,
        scramble=False,
        seed=None,
        J_max=None
    ):

        d = len(self.nested_framework.get_risk_factors_name())
        J = 2 ** (m)
        sobol_engine = stats.qmc.Sobol(d, scramble=scramble, rng=np.random.default_rng(seed) if scramble else None)
        risk_factors = (u_bound - l_bound) * cp.asarray(sobol_engine.random_base2(m)) + l_bound

        return self.sample_dataset_base(J, risk_factors, rng, K, antithetic, J_max=J_max)

    def sample_full_sobol_dataset(
        self: typing.Self,
        m: int,
        l_bound: npt.NDArray[np.float64],
        u_bound: npt.NDArray[np.float64],
        K: int,
        rng: np.random.Generator | cp.random.Generator,
        scramble: bool,
    ) -> pd.DataFrame:
        """
        Generate a full Sobol dataset of risk factors and payoffs using the nested framework.

        This method generates Sobol quasi-random samples of risk factors within the bounds [l_bound[i], u_bound[i]],
        evaluates payoffs using the nested framework for each sample, and returns a pandas DataFrame with
        risk factors and payoffs.

        The returned DataFrame contains one column for each risk factor and one column per payoff.
        The number of samples is 2**m, and the number of risk factor columns matches self.rf_names.
        The payoff columns are named "Payoff 1", "Payoff 2", ..., "Payoff K".

        Args:
            m (int): Power for the number of Sobol samples to generate, i.e., number of samples is 2**m.
                    Must be non-negative.
            l_bound (npt.NDArray[np.float64]): Lower bounds for the risk factor sampling interval.
            u_bound (npt.NDArray[np.float64]): Upper bounds for the risk factor sampling interval. Must be strictly greater than l_bound.
            K (int): Number of payoffs to compute for each sample. Must be positive.
            rng (np.random.Generator | cp.random.Generator): Random number generator for backend computations.
                Determines whether numpy or cupy is used for calculations.
            scramble (bool): If True, applies scrambling to the Sobol sequence.
                If False, generates a standard (unscrambled) Sobol sequence.

        Raises:
            ValueError: If m is negative.
            ValueError: If l_bound is not strictly less than u_bound (component wise).
            ValueError: If K is not positive.

        Returns:
            pd.DataFrame: Dataset with shape (2**m, rf_dim + K), where rf_dim is the number of risk factors.
                The columns are risk factor names followed by "Payoff 1", ..., "Payoff K".
                Each row corresponds to a Sobol sample and its computed payoffs.
        """

        if m < 0:
            raise ValueError("m ({}) must be positive.".format(m))

        if (l_bound >= u_bound).all():
            raise ValueError(
                "l_bound ({}) must be strictly lower than u_bound ({}).".format(
                    l_bound, u_bound
                )
            )

        if K <= 0:
            raise ValueError("K ({}) must be greater or equal to 1.".format(K))

        sobol_engine = stats.qmc.Sobol(self.rf_dim, scramble=scramble)
        rf_points = (u_bound - l_bound) * sobol_engine.random_base2(m) + l_bound

        # Convert to cupy for computations
        if not isinstance(rng, np.random.Generator):

            rf_points = cp.asarray(rf_points)

        payoffs = self.nested_framework.sample_F_mat_from_risk_factors(
            K, rf_points, rng
        )
        rf_names = self.rf_names

        # Convert back to numpy for outputs
        if not isinstance(rng, np.random.Generator):

            rf_points = rf_points.get()  # type: ignore
            payoffs = payoffs.get()  # type: ignore

        return construct_dataset(rf_points, rf_names, payoffs, average_payoff=False)


def construct_dataset(
    rf_points: np.ndarray,
    rf_names: list[str],
    payoffs: np.ndarray,
    average_payoff: bool,
) -> pd.DataFrame:
    """
    Construct a pandas DataFrame from reference points and payoffs, with options for average or individual payoffs.

    The function checks for consistency between the shapes of the input arrays and the list of reference names.
    It then creates a DataFrame where each reference name becomes a column containing the corresponding values
    from rf_points. The payoff columns are added as either the average of payoffs (if average_payoff is True)
    or as separate columns for each payoff (if average_payoff is False).

    Args:
        rf_points (np.ndarray): A 2D numpy array of shape (n_samples, n_features) representing the reference points.
        rf_names (list[str]): List of length n_features with the names of the reference point dimensions (columns).
        payoffs (np.ndarray): A 2D numpy array of shape (n_samples, n_payoffs) representing the payoffs.
        average_payoff (bool): If True, adds a single column "Payoff Avg" with the mean payoff per sample.
                               If False, adds one column per payoff, named "Payoff 1", "Payoff 2", etc.

    Raises:
        ValueError: If rf_points is not a 2D array.
        ValueError: If payoffs is not a 2D array.
        ValueError: If the number of samples in rf_points and payoffs do not match.
        ValueError: If the number of reference names does not match the number of columns in rf_points.

    Returns:
        pd.DataFrame: A DataFrame with columns for each reference name and payoff (either average or individual).
                      Each row corresponds to a sample.
    """

    if rf_points.ndim != 2:
        raise ValueError(
            "rf_points must be a 2D array (ndim {})".format(rf_points.ndim)
        )

    if payoffs.ndim != 2:
        raise ValueError("payoffs must be a 2D array (ndim {})".format(payoffs.ndim))

    if rf_points.shape[0] != payoffs.shape[0]:
        error_str = (
            "rf_point.shape[0] ({}) must be equal to payoffs.shape[0] ({})".format(
                rf_points.shape[0], payoffs.shape[0]
            )
        )
        raise ValueError(error_str)

    if rf_points.shape[1] != len(rf_names):
        error_str = (
            "rf_points.shape[1] ({}) must be equal to len(rf_names) ({})".format(
                rf_points.shape[1], len(rf_names)
            )
        )
        raise ValueError(error_str)

    data = {}

    for i in range(len(rf_names)):

        name = rf_names[i]
        data[name] = rf_points[:, i]

    if average_payoff:

        name = "Payoff Avg"
        data[name] = np.mean(payoffs, axis=1)

    else:

        for i in range(payoffs.shape[1]):

            name = "Payoff {}".format(i + 1)
            data[name] = payoffs[:, i]

    return pd.DataFrame(data)
