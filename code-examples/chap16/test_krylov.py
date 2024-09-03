import numpy as np
from .krylov import solve_krylov

def test_krylov():
    n = 100
    a = np.random.rand(n,n) * .1
    b = np.random.rand(n)
    matvec = lambda x: a.dot(x.ravel())
    x = solve_krylov(matvec, b, tol=1e-5)
    ref = np.linalg.solve(np.eye(n)+a, b)
    assert abs(ref - x).max() < 1e-5
