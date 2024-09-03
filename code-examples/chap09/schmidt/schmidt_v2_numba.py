import numpy as np
import numba

@numba.njit(cache=True)
def schmidt_orth(s):
    '''
    Schmidt orthogonalization for overlap matrix S

    \sum_{mn} c_{mi} s_{mn} c_{nj} = \delta_{ij}
    '''
    n = s.shape[0]
    cs = np.zeros((n, n))
    tmp = np.empty(n)
    for j in range(n):
        fac = s[j,j]
        for k in range(j):
            # The dot method is overloaded by numba, which is compiled to
            # a call to cblas_ddot provided by scipy.linalg.cython_blas
            dot_kj = cs[k].dot(s[j])
            # Expanding the broadcasting may help LLVM detect SIMD vectorization
            for i in range(n):
                cs[j,i] -= dot_kj * cs[k,i]
            fac -= dot_kj * dot_kj

        if fac <= 0:
            raise RuntimeError(f'schmidt_orth fail. {j=} {fac=}')
        fac = fac**-.5
        cs[j,j] = fac
        for i in range(j):
            cs[j,i] *= fac
    return cs.T

if __name__ == '__main__':
    s = np.random.rand(2000, 2000)
    s = s.dot(s.T)
    cs = schmidt_orth(s)
    #ref = np.linalg.inv(np.linalg.cholesky(s))
    #print(abs(cs - ref.T).max())
