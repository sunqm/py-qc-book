import numpy as np

#pythran export matmul_nn_v3(float64[:,:], float64[:,:])
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

#pythran export matmul_nn_tiling(float64[:,:], float64[:,:])
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

#pythran export matmul_nn_tiling_unrolled(float64[:,:], float64[:,:])
def matmul_nn_tiling_unrolled(a, b):
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
