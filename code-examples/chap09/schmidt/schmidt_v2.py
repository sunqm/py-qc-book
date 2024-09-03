import numpy as np

def schmidt_orth(s):
    '''
    Schmidt orthogonalization for overlap matrix S

    \sum_{mn} c_{mi} s_{mn} c_{nj} = \delta_{ij}
    '''
    fdot = np.dot
    n = s.shape[0]
    cs = np.zeros((n, n))
    for j in range(n):
        fac = s[j,j]
        for k in range(j):
            dot_kj = fdot(cs[k], s[j])
            cs[j] -= dot_kj * cs[k]
            fac -= dot_kj * dot_kj

        if fac <= 0:
            raise RuntimeError(f'schmidt_orth fail. {j=} {fac=}')
        fac = fac**-.5
        cs[j,j] = fac
        cs[j,:j] *= fac
    return cs.T

if __name__ == '__main__':
    s = np.random.rand(2000, 2000)
    s = s.dot(s.T)
    cs = schmidt_orth(s)
    #ref = np.linalg.inv(np.linalg.cholesky(s))
    #print(abs(cs - ref.T).max())
