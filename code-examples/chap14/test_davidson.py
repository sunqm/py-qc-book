import numpy as np
from .davidson import davidson

def test_eigh_root1():
    np.random.seed(1)
    n = 40
    a = np.random.rand(n, n)
    a = a.T + a
    aop = lambda x: a.dot(x)
    e = davidson(aop, a.diagonal())[0]
    ref, _ = np.linalg.eigh(a)
    assert abs(e - ref[0]).max() < 1e-8
