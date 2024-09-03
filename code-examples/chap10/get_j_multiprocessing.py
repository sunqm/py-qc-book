from multiprocessing import Array, Lock, Process
from multiprocessing.shared_memory import SharedMemory
import numpy as np

def compute_eri(k, n):
    # Mimic the function to compute ERIs
    np.random.seed(k)
    return np.random.rand(n,n,n)

def get_j_task(k, dm, output):
    n = dm.shape[0]
    output[k] = np.einsum('ijl,ji->l', compute_eri(k, n), dm)

def get_j(dm, output):
    output[:] = 0.
    ps = [Process(target=get_j_task, args=(k, dm, output)) for k in range(n)]
    [p.start() for p in ps]
    [p.join() for p in ps]
    return output

if __name__ == '__main__':
    n = 50
    dm = np.identity(n)

    shm = Array(dm.dtype.char, dm.size)
    output = np.ndarray(dtype=dm.dtype, shape=dm.shape, buffer=shm.get_obj())
    print(get_j(dm, output).sum())

    shm = SharedMemory(create=True, size=dm.nbytes)
    shm.unlink()
    output = np.ndarray(dtype=dm.dtype, shape=dm.shape, buffer=shm.buf)
    print(get_j(dm, output).sum())

    output = np.memmap('/dev/shm/output', dtype=dm.dtype, shape=dm.shape, mode='w+')
    print(get_j(dm, output).sum())
