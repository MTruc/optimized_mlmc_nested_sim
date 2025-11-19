from alm_model.nested import AlmFramework
import math
import cupy as cp

def sample_EK_gpu(framework : AlmFramework,
                            nb_samples : int,
                            K : int,
                            rng : cp.random.Generator):
    
    F_mat = framework.sample_F_matrix_gpu(K, nb_samples, rng)
    return framework.compute_EK_gpu(F_mat)

def sample_delta_EK_anti_gpu(framework : AlmFramework,
                             nb_samples : int,
                             Kf : int,
                             rng : cp.random.Generator):
    
    F_mat = framework.sample_F_matrix_gpu(Kf, nb_samples, rng)
    return framework.compute_EK_anti_gpu(F_mat)
    
def calibrate_max_J_per_batch_nested_mc(free_mem : float,
                                        peak_per_unit : float,
                                        K : int,
                                        margin=0.9):
    """Compute the maximum J per batch for a nested mc."""

    mem_per_J = peak_per_unit * K
    return math.floor(free_mem * margin / mem_per_J)

def calibrate_max_J_per_batch_mlmc(free_mem : float,
                                        peak_per_unit : float,
                                        K : int,
                                        R : int,
                                        margin=0.9):
    """Compute the maximum J per batch for a nested mc."""

    max_Js = []
    for r in range(R):
        mem_per_J = peak_per_unit * K * 2**r
        max_Js.append(math.floor(free_mem * margin / mem_per_J))

    return max_Js