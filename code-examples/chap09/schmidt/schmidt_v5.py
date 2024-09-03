import numpy as np
import schmidt_unrolled

schmidt_orth_small_size = [getattr(schmidt_unrolled, f'schmidt_orth_n{n}') for n in range(8)]

def schmidt_orth_recursive_unrolled(s):
    '''
    Recursively solve Schmidt orthogonalization for overlap matrix S

    \sum_{mn} c_{mi} s_{mn} c_{nj} = \delta_{ij}
    '''
    n = s.shape[0]
    if n < 8:
        return schmidt_orth_small_size[n](s)

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
