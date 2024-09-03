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

def schmidt_orth_recursive(s):
    '''
    Recursively solve Schmidt orthogonalization for overlap matrix S

    \sum_{mn} c_{mi} s_{mn} c_{nj} = \delta_{ij}
    '''
    n = s.shape[0]
    if n < 8:
        return schmidt_orth(s)

    n = s.shape[0]
    m = n // 2
    c_block11 = schmidt_orth_recursive(s[:m,:m])
    s_block12 = np.dot(c_block11.T, s[:m,m:])
    c_block12 = -np.dot(c_block11, s_block12)
    s_block22 = s[m:,m:] - np.dot(s_block12.T, s_block12)
    c_block22 = schmidt_orth_recursive(s_block22)
    cs = np.zeros((n, n))
    cs[:m,:m] = c_block11
    cs[m:,m:] = c_block22
    cs[:m,m:] = np.dot(c_block12, c_block22)
    return cs

if __name__ == '__main__':
    s = np.random.rand(2000, 2000)
    s = s.dot(s.T)
    cs = schmidt_orth_recursive(s)
    #ref = np.linalg.inv(np.linalg.cholesky(s))
    #print(abs(cs - ref.T).max())
