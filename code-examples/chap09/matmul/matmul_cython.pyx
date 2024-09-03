#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
#cython: cdivision=True
#cython: language_level=3
#distutils: extra_compile_args=-fopenmp -O3 -march=native -funroll-loops -fvect-cost-model=unlimited -ffast-math
#distutils: extra_link_args=-fopenmp

import numpy as np
from cython.parallel import prange

def matmul_nn_tiling(double[:,::1] a, double[:,::1] b):
    cdef int arow = a.shape[0]
    cdef int acol = a.shape[1]
    cdef int brow = b.shape[0]
    cdef int bcol = b.shape[1]
    assert acol == brow
    _c = np.zeros((arow, bcol))
    cdef double[:, ::1] c = _c
    cdef int block_size = 200
    cdef int i0, i1, j0, j1, k0, k1, i, j, k
    cdef double a_ik

    for i0 in range(0, arow, block_size):
        i1 = min(i0 + block_size, arow)
        for j0 in range(0, bcol, block_size):
            j1 = min(j0 + block_size, bcol)
            for k0 in range(0, acol, block_size):
                k1 = min(k0 + block_size, acol)
                for i in range(i0, i1):
                    for k in range(k0, k1):
                        a_ik = a[i,k]
                        for j in range(j0, j1):
                            c[i,j] += a_ik * b[k,j]
    return _c

def matmul_nn_tiling_unrolled(double[:,::1] a, double[:,::1] b):
    cdef int arow = a.shape[0]
    cdef int acol = a.shape[1]
    cdef int brow = b.shape[0]
    cdef int bcol = b.shape[1]
    assert acol == brow
    _c = np.zeros((arow, bcol))
    cdef double[:, ::1] c = _c
    cdef int block_size = 200
    cdef int i0, i1, j0, j1, k0, k1, i, j, k
    cdef double a_ik, b_kj
    cdef double a_ik0, a_ik1, a_ik2, a_ik3

    for i0 in range(0, arow, block_size):
        i1 = min(i0 + block_size, arow)
        for j0 in range(0, bcol, block_size):
            j1 = min(j0 + block_size, bcol)
            for k0 in range(0, acol, block_size):
                k1 = min(k0 + block_size, acol)
                for i in range(i0, i1-3, 4):
                    for k in range(k0, k1):
                        a_ik0 = a[i+0,k]
                        a_ik1 = a[i+1,k]
                        a_ik2 = a[i+2,k]
                        a_ik3 = a[i+3,k]
                        for j in range(j0, j1):
                            b_kj = b[k,j]
                            c[i+0,j] += a_ik0 * b_kj
                            c[i+1,j] += a_ik1 * b_kj
                            c[i+2,j] += a_ik2 * b_kj
                            c[i+3,j] += a_ik3 * b_kj
                for i in range(i+4, i1):
                    for k in range(k0, k1):
                        a_ik = a[i,k]
                        for j in range(j0, j1):
                            c[i,j] += a_ik * b[k,j]
    return c
