import numpy as np

def solve_krylov(A, b, x0=None, tol=1e-5, maxiter=50):
    '''Krylov subspace method to solve  (1+A) x = b.

    Parameters:
    - A: callable, a function that computes the product of a square matrix and a vector.
    - b: np.ndarray, the right-hand side vector of the linear equation.
    '''
    if x0 is not None:
        x = solve_krylov(A, b-A(x0)-x0, tol=tol, maxiter=maxiter)
        return x + x0

    vs = [b]
    hv = []
    h = np.empty((maxiter, maxiter))
    g = np.empty(maxiter)
    c = []

    for cycle in range(maxiter):
        # Generate new vector in Krylov subspace and orthogonalize it
        v1 = A(vs[-1])
        hv.append(v1 + vs[cycle]) # (I+A)*v
        for v in vs:
            v1 -= v * np.dot(v, v1)
        # Add to subspace if it linearly independent to others
        v_norm = np.linalg.norm(v1)
        if v_norm < 1e-10:
            break
        v1 /= v_norm
        vs.append(v1)

        for i in range(cycle+1):
            h[i,cycle] = np.dot(vs[i], hv[cycle])
            h[cycle,i] = np.dot(vs[cycle], hv[i])
        g[cycle] = np.dot(vs[cycle], b)
        c = np.linalg.solve(h[:cycle+1,:cycle+1], g[:cycle+1])

        residual = -b
        for i, ci in enumerate(c):
            # Reuse hv vectors to reduce the matvec A(x)
            residual += hv[i] * ci
        norm = np.linalg.norm(residual)
        print(f'krylov {cycle=} residual={norm:.5g}')
        if norm < tol:
            break
    else:
        raise RuntimeError('solve_krylov not converged')

    x = np.zeros_like(b)
    for i, ci in enumerate(c):
        x += vs[i] * ci
    return x
