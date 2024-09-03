import numpy as np

def davidson(A, A_diag, x0=None, tol=1e-5, maxiter=50, space=15):
    '''Davidson diagonalization method to solve  Ax = ex for symmetric matrix A

    Parameters:
    - A: callable, a function that computes the product of a square matrix and a vector.
    - A_diag: np.ndarray, Diagonal elements of A

    Returns:
    - e: the lowest eigenvalue
    - x: the corresponding eigenvector
    '''
    def precond(r, e):
        return r / (A_diag - e)

    # The initial guess
    if x0 is None:
        x0 = 1. / (A_diag - A_diag.argmin() + 1e-2)
    x0 = x0 / np.linalg.norm(x0)

    vs = []
    hv = []
    h = np.empty((space, space))

    for cycle in range(maxiter):
        vs.append(x0)
        hv.append(A(x0))
        n = len(vs)
        for i in range(n):
            h[i,n-1] = np.dot(vs[i], hv[n-1])
            h[n-1,i] = h[i,n-1].conj()

        e, c = np.linalg.eigh(h[:n,:n])
        e0 = e[0]
        c0 = c[:,0]

        x = np.zeros_like(x0)
        for i, ci in enumerate(c0):
            x += vs[i] * ci
        residual = -e0 * x
        for i, ci in enumerate(c0):
            residual += hv[i] * ci
        norm = np.linalg.norm(residual)
        print(f'davidson {cycle=} {e0=} residual={norm:.5g}')
        if norm < tol:
            break

        if len(vs) >= space:
            # Restart the calculation, the last x is used as initial guess
            x0, vs, hv = x, [], []
            continue

        x0 = precond(residual, e0)
        # orthogonalize the basis
        for v in vs:
            x0 -= v * np.dot(v, x0)
        norm = np.linalg.norm(x0)
        if norm < 1e-10:
            raise RuntimeError('Linearly dependent basis vectors')
        x0 /= norm
    else:
        raise RuntimeError('davidson not converged')

    return e0, x
