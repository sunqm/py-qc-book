from concurrent.futures import ProcessPoolExecutor
import numpy as np

def compute_eri(k, n):
    # Mimic the function to compute ERIs
    np.random.seed(k)
    return np.random.rand(n,n,n)

def get_j_task(k, dm, output_attr):
    filename, dtype, shape = output_attr
    output = np.memmap(filename, shape=shape, dtype=dtype, mode='r+')
    n = dm.shape[0]
    output[k] = np.einsum('ijl,ji->l', compute_eri(k, n), dm)

def get_j(dm):
    n = dm.shape[0]
    output = np.memmap('/dev/shm/output', dtype=dm.dtype, shape=dm.shape, mode='w+')
    output[:] = 0.
    output_attr = ('/dev/shm/output', output.dtype, output.shape)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(get_j_task, i, dm, output_attr) for i in range(n)]
        [f.result() for f in futures]
    return output

if __name__ == '__main__':
    n = 50
    dm = np.identity(n)
    print(get_j(dm).sum())
