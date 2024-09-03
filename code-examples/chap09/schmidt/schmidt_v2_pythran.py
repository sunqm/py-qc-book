import numpy as np

#pythran export schmidt_orth(float64[:,:])
def schmidt_orth(s):
    n = s.shape[0]
    cs = np.zeros((n, n))
    for j in range(n):
        fac = s[j,j]
        for k in range(j):
            dot_kj = cs[k].dot(s[j])
            cs[j] -= dot_kj * cs[k]
            fac -= dot_kj * dot_kj

        if fac <= 0:
            raise RuntimeError('schmidt_orth')
        fac = fac**-.5
        cs[j,j] = fac
        for i in range(j):
            cs[j,i] *= fac
    return cs.T
