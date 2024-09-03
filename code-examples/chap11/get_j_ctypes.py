import ctypes
import numpy as np
import cupy as cp

gpu_ext = ctypes.CDLL('./libget_j.so')

def empty_mapped(shape, dtype=float, order='C'):
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    mem = cp.cuda.PinnedMemoryPointer(
        cp.cuda.PinnedMemory(nbytes, cp.cuda.runtime.hostAllocMapped), 0)
    out = np.ndarray(shape, dtype=dtype, buffer=mem, order=order)
    return out

def get_j(eri, dm, kernel='kernel_v2'):
    n = dm.shape[0]
    if 'v3' in kernel:
        assert n % 16 == 0
    dm = cp.asarray(dm)
    output = cp.zeros_like(dm)
    f = getattr(gpu_ext, kernel)
    if 'mapped' in kernel:
        f(ctypes.c_void_p(output.data.ptr), eri.ctypes,
          ctypes.c_void_p(dm.data.ptr), ctypes.c_int(n))
    else:
        eri = cp.asarray(eri)
        f(ctypes.c_void_p(output.data.ptr),
          ctypes.c_void_p(eri.data.ptr),
          ctypes.c_void_p(dm.data.ptr), ctypes.c_int(n))
    return output

if __name__ == '__main__':
    n = 64
    np.random.seed(1)
    eri = np.random.rand(n, n, n, n)
    dm = np.random.rand(n, n)
    output = get_j(eri, dm) # warmup

    output = get_j(eri, dm, kernel='kernel_v1')
    print(output.sum())

    output = get_j(eri, dm, kernel='kernel_v2')
    print(output.sum())

    output = get_j(eri, dm, kernel='kernel_v3')
    print(output.sum())

    rng = np.random.default_rng(seed=1)
    eri = empty_mapped((n, n, n, n))
    rng.random((n, n, n, n), out=eri)

    output = get_j(eri, dm, kernel='kernel_v1_mapped')
    print(output.sum())

    output = get_j(eri, dm, kernel='kernel_v2_mapped')
    print(output.sum())

    output = get_j(eri, dm, kernel='kernel_v3_mapped')
    print(output.sum())
