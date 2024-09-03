import itertools
import numpy as np
import mpi_map

def compute_eri(i, n):
    # Mimic the function to compute ERIs
    np.random.seed(i)
    return np.random.rand(n,n,n)

def get_j_task(i, dm):
    n = dm.shape[0]
    return np.einsum('jkl,j->kl', compute_eri(i, n), dm[:,i])

def get_j(dm):
    n = dm.shape[0]
    output = mpi_map.map(get_j_task, range(n), itertools.repeat(dm, n))
    jmat = sum(output)
    return jmat
