import numpy as np
import cupy as cp
# PyCUDA automatic initialization
# https://documen.tician.de/pycuda/util.html#module-pycuda.autoinit
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule

pycuda_mod = SourceModule('''
__global__ void _kernel_v1(double *output, double *eri, double *dm, int n)
{
    size_t ij = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nn = n * n;
    if (ij > nn) return;
    size_t i = ij / n;
    size_t j = ij % n;
    size_t kl;
    double dm_ji = dm[j*n+i];
    for (kl = 0; kl < nn; ++kl) {
        atomicAdd(&output[kl], eri[(i*n+j)*nn+kl] * dm_ji);
    }
}

__global__ void _kernel_v2(double *output, double *eri, double *dm, int n)
{
    size_t kl = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nn = n * n;
    if (kl > nn) return;
    size_t i, j;
    double s = 0.;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            s += eri[(i*n+j)*nn+kl] * dm[j*n+i];
        }
    }
    output[kl] = s;
}
''')

kernel_v1 = pycuda_mod.get_function('_kernel_v1')
kernel_v2 = pycuda_mod.get_function('_kernel_v2')

def get_j(eri, dm, kernel=kernel_v2):
    n = dm.shape[0]
    eri = cp.asarray(eri)
    dm = cp.asarray(dm)
    output = cp.zeros_like(dm)
    blocks = ((n*n+255) // 256, 1, 1)
    threads = (256, 1, 1)
    kernel(output, eri, dm, np.int32(n), block=blocks, grid=threads)
    return output


if __name__ == '__main__':
    n = 50
    np.random.seed(1)
    eri = np.random.rand(n,n,n,n)
    dm = np.random.rand(n,n)
    output = get_j(eri, dm)
    print(output.sum())

    output = get_j(eri, dm, kernel_v1)
    print(output.sum())
