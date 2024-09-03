import numpy as np
import scipy.sparse as sp

def harmonic_oscillator(m, k, ngrids, box_size):
    '''H = 1/2m p^2 + 1/2 k x^2'''
    x = np.linspace(-box_size, box_size, ngrids)
    dx = x[1] - x[0]
    def matvec(psi):
        d2 = psi * -2
        d2[:-1] += psi[1:]
        d2[1:] += psi[:-1]
        return -1/(2*m*dx**2) * d2 + k/2 * x**2 * psi

    H = sp.linalg.LinearOperator((ngrids, ngrids), matvec=matvec)
    return H

if __name__ == '__main__':
    m = 0.5
    k = 0.5
    ngrids = 1000
    box_size = 10.

    A = harmonic_oscillator(m, k, ngrids, box_size)
    e, psi = sp.linalg.eigsh(A, which='SM')
    print(e)
