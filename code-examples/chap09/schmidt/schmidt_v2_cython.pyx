#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
#cython: cdivision=True
#cython: language_level=3
#cython: emit_linenums=True
#distutils: extra_compile_args=-fopenmp
#distutils: extra_link_args=-fopenmp

import numpy as np
from cython.parallel import prange

def schmidt_orth(double[:, ::1] s):
    cdef int n = s.shape[0]
    _cs = np.zeros((n, n))
    cdef double[:, ::1] cs = _cs
    cdef int i, j, k
    cdef double dot_kj, fac
    cdef double[::1] dot_kj_buf = np.empty(n)

    for j in range(n):
        fac = s[j,j]
        for k in prange(j, schedule='static', nogil=True):
            dot_kj = 0.
            for i in range(n):
                dot_kj = dot_kj + cs[k,i] * s[j,i]
            fac -= dot_kj * dot_kj
            dot_kj_buf[k] = dot_kj

        for i in prange(n, schedule='static', nogil=True):
            for k in range(j):
                cs[j,i] -= dot_kj_buf[k] * cs[k,i]

        if fac <= 0:
            raise RuntimeError(f'schmidt_orth fail. j={j} fac={fac}')
        fac = fac**-.5
        cs[j,j] = fac
        for i in range(j):
            cs[j,i] *= fac
    return _cs.T
