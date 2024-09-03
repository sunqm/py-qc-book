import numpy as np
import cupy as cp
from numba import cuda

@cuda.jit(fastmath=True)
def naive_kernel(output, eri, dm):
    n = dm.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    output[k,l] += eri[i,j,k,l] * dm[j,i]

@cuda.jit(fastmath=True)
def kernel_v1(output, eri, dm):
    n = dm.shape[0]
    ij = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if ij > n * n: return
    i, j = divmod(ij, n)
    dm_ji = dm[j,i]
    for k in range(n):
        for l in range(n):
            cuda.atomic.add(output, (k,l), eri[i,j,k,l] * dm_ji)

BLOCK = 16

@cuda.jit(fastmath=True)
def kernel_v3(output, eri, dm):
    n = dm.shape[0]
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    k, l = cuda.grid(2)
    dm_t = cuda.shared.array((BLOCK, BLOCK), dtype='f8')

    s = 0.
    for i0 in range(0, n, BLOCK):
        for j0 in range(0, n, BLOCK):
            cuda.syncthreads()
            dm_t[tx,ty] = dm[j0+ty,i0+tx]
            cuda.syncthreads()
            for i in range(BLOCK):
                for j in range(BLOCK):
                    s += eri[i0+i,j0+j,k,l] * dm_t[i,j]
    output[k,l] = s

def get_j(eri, dm, kernel=kernel_v1):
    n = dm.shape[0]
    output = np.zeros_like(dm)
    blocks = ((n*n+255) // 256,)
    threads = (256,)
    kernel[blocks, threads](output, eri, dm)
    return output

def get_j_v3(eri, dm):
    n = dm.shape[0]
    assert n % BLOCK == 0
    output = np.empty_like(dm)
    blocks = ((n+BLOCK-1)//BLOCK, (n+BLOCK-1)//BLOCK)
    threads = (BLOCK, BLOCK)
    kernel_v3[blocks, threads](output, eri, dm)
    return output

if __name__ == '__main__':
    n = 64
    dm = cp.random.rand(n,n)
    eri = cp.random.rand(n,n,n,n)
    print(get_j(eri, dm).sum())
    print(get_j(eri, dm, naive_kernel).sum())
    print(get_j_v3(eri, dm).sum())
