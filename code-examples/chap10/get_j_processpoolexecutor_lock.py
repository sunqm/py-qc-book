import os
import tempfile
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def compute_eri(i, n):
    # Mimic the function to compute ERIs
    np.random.seed(i)
    return np.random.rand(n,n,n)

def get_j_task(i, dm, output_attr):
    filename, dtype, shape, lock = output_attr
    output = np.memmap(filename, shape=shape, dtype=dtype, mode='r+')
    n = dm.shape[0]
    jmat = np.einsum('jkl,j->kl', compute_eri(i, n), dm[:,i])
    with lock:
        output += jmat

def get_j(dm):
    mmap_file = tempfile.mktemp()
    output = np.memmap(mmap_file, dtype=dm.dtype, shape=dm.shape, mode='w+')
    with Manager() as mgr, ProcessPoolExecutor(max_workers=4) as scheduler:
        lock = mgr.Lock()
        output_attr = (mmap_file, output.dtype, output.shape, lock)
        for _ in scheduler.map(get_j_task, range(n), [dm]*n, [output_attr]*n):
            pass
    output = output.copy()
    os.remove(mmap_file)
    return output

if __name__ == '__main__':
    n = 50
    dm = np.identity(n)
    print(get_j(dm).sum())
