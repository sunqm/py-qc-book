from transpose import transpose
import numpy as np
import numba

@numba.njit(cache=True)
def matmul_nn_v1(a, b):
    '''Naive implmentation of the matrix multiplication a * b'''
    arow, acol = a.shape
    brow, bcol = b.shape
    assert acol == brow
    c = np.zeros((arow, bcol))
    for i in range(arow):
        for j in range(bcol):
            c_ij = 0
            for k in range(acol):
                c_ij += a[i,k] * b[k,j]
            c[i,j] = c_ij
    return c

def matmul_nn_v2(a, b):
    '''Matrix multiplication a * b with a transposed matrix intermediate'''
    b_t = transpose(b)
    return matmul_nt(a, b_t)

@numba.njit(cache=True)
def matmul_nn_v3(a, b):
    '''Matrix multiplication a * b with improved memory efficiency'''
    arow, acol = a.shape
    brow, bcol = b.shape
    assert acol == brow
    c = np.zeros((arow, bcol))
    for i in range(arow):
        for k in range(acol):
            a_ik = a[i,k]
            for j in range(bcol):
                c[i,j] += a_ik * b[k,j]
    return c

@numba.njit(cache=True)
def matmul_nt(a, b):
    arow, acol = a.shape
    brow, bcol = b.shape
    assert acol == bcol
    c = np.zeros((arow, brow))
    for i in range(arow):
        for j in range(brow):
            c_ij = 0
            for k in range(acol):
                c_ij += a[i,k] * b[j,k]
            c[i,j] = c_ij
    return c

@numba.njit(cache=True)
def matmul_tn(a, b):
    arow, acol = a.shape
    brow, bcol = b.shape
    assert arow == brow
    c = np.zeros((acol, bcol))
    for i in range(acol):
        for k in range(arow):
            a_ki = a[k,i]
            for j in range(bcol):
                c[i,j] += a_ki * b[k,j]
    return c

@numba.njit(cache=True)
def matmul_nn_tiling(a, b):
    '''Matrix multiplication a * b with loop tiling'''
    arow, acol = a.shape
    brow, bcol = b.shape
    assert acol == brow
    c = np.zeros((arow, bcol))
    block_size = 200

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
    return c

@numba.njit(cache=True)
def matmul_nn_tiling_unrolled(a, b):
    '''Matrix multiplication a * b with loop tiling and unrolling'''
    arow, acol = a.shape
    brow, bcol = b.shape
    assert acol == brow
    c = np.zeros((arow, bcol))
    block_size = 200

    for i0 in range(0, arow, block_size):
        i1 = min(i0 + block_size, arow)
        for j0 in range(0, bcol, block_size):
            j1 = min(j0 + block_size, bcol)
            for k0 in range(0, acol, block_size):
                k1 = min(k0 + block_size, acol)
                for i in range(i0, i1-7, 8):
                    for k in range(k0, k1):
                        a_ik0 = a[i+0,k]
                        a_ik1 = a[i+1,k]
                        a_ik2 = a[i+2,k]
                        a_ik3 = a[i+3,k]
                        a_ik4 = a[i+4,k]
                        a_ik5 = a[i+5,k]
                        a_ik6 = a[i+6,k]
                        a_ik7 = a[i+7,k]
                        for j in range(j0, j1):
                            b_kj = b[k,j]
                            c[i+0,j] += a_ik0 * b_kj
                            c[i+1,j] += a_ik1 * b_kj
                            c[i+2,j] += a_ik2 * b_kj
                            c[i+3,j] += a_ik3 * b_kj
                            c[i+4,j] += a_ik4 * b_kj
                            c[i+5,j] += a_ik5 * b_kj
                            c[i+6,j] += a_ik6 * b_kj
                            c[i+7,j] += a_ik7 * b_kj
                for i in range(i+8, i1):
                    for k in range(k0, k1):
                        a_ik = a[i,k]
                        for j in range(j0, j1):
                            c[i,j] += a_ik * b[k,j]
    return c

@numba.njit(fastmath=True, cache=True)
def matmul_nn_tiling_simd(a, b):
    '''Matrix multiplication a * b with loop tiling'''
    arow, acol = a.shape
    brow, bcol = b.shape
    assert acol == brow
    c = np.zeros((arow, bcol))
    block_size = 200

    for i0 in range(0, arow, block_size):
        i1 = min(i0 + block_size, arow)
        for j0 in range(0, bcol, block_size):
            j1 = min(j0 + block_size, bcol)
            for k0 in range(0, acol, block_size):
                k1 = min(k0 + block_size, acol)
                for i in range(i0, i1):
                    cp = c[i,j0:j1]
                    for k in range(k0, k1):
                        a_ik = a[i,k]
                        bp = b[k,j0:j1]
                        for j in range(j1-j0):
                            cp[j] += a_ik * bp[j]
    return c
