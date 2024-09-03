import numpy as np

def transpose(a):
    block_size = 200
    arow, acol = a.shape
    out = np.empty((acol,arow), a.dtype)
    for c0 in range(0, acol, block_size):
        c1 = min(acol, c0 + block_size)
        for r0 in range(0, arow, block_size):
            r1 = min(arow, r0 + block_size)
            out[c0:c1,r0:r1] = a[r0:r1,c0:c1].T
    return out
