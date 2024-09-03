import itertools
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue
import numpy as np

q = SimpleQueue()

def compute_eri(i, n):
    # Mimic the function to compute ERIs
    np.random.seed(i)
    return np.random.rand(n,n,n)

def get_j_task(i, dm):
    n = dm.shape[0]
    jmat = np.einsum('jkl,j->kl', compute_eri(i,n), dm[:,i])
    q.put(jmat)

def reduction(n):
    output = np.zeros((n,n))
    for i in range(n):
        output += q.get()
    return output

if __name__ == '__main__':
    n = 25
    dm = np.identity(n)
    with ThreadPoolExecutor(max_workers=8) as scheduler:
        fut = scheduler.submit(reduction, n)
        mapped = scheduler.map(get_j_task, range(n), itertools.repeat(dm, n))
    print(fut.result().shape)
