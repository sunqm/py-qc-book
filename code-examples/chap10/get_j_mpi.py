'''
Execute this script with the command:

    mpirun -n 8 python get_j_mpi.py
'''

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

einsum = np.einsum

rank = comm.Get_rank()
size = comm.Get_size()

def broadcast(arr):
    if rank == 0:
        comm.bcast((arr.shape, arr.dtype.char))
    else:
        shape, dtype = comm.bcast(None)
        arr = np.empty(shape, dtype=dtype)
    # Relies on automatic MPI datatype discovery.
    # Array data will be written into the buffer directly.
    comm.Bcast([arr, arr.dtype.char])
    return arr

def compute_eri(i, n):
    # Mimic the function to compute ERIs
    np.random.seed(i)
    return np.random.rand(n,n,n)

def get_j(dm=None):
    dm = broadcast(dm)
    try:
        n = dm.shape[0]
        output = np.zeros((n, n))
        for i in range(0, n, size):
            output += einsum('jkl,j->kl', compute_eri(i, n), dm[:,i])
    except Exception:
        comm.Abort(1)

    jmat = np.zeros((n, n))
    req = comm.Iallreduce(output, jmat) # Communication in background
    req.wait()
    return output

if rank == 0:
    n = 25
    dm = np.random.rand(n, n)
    output = get_j(dm)
    print(output.shape)
else:
    get_j()
