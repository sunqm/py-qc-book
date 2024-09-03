import time
import numpy as np
import matmul

def test_matmul_nn():
    m, n, k = 10, 30, 20
    a = np.random.rand(m, k)
    b = np.random.rand(k, n)
    c = matmul.matmul_nn_v1(a, b)
    assert abs(a.dot(b) - c).max() < 1e-14
    c = matmul.matmul_nn_v2(a, b)
    assert abs(a.dot(b) - c).max() < 1e-14
    c = matmul.matmul_nn_v3(a, b)
    assert abs(a.dot(b) - c).max() < 1e-14
    c = matmul.matmul_nn_tiling(a, b)
    assert abs(a.dot(b) - c).max() < 1e-14
    c = matmul.matmul_nn_tiling_unrolled(a, b)
    assert abs(a.dot(b) - c).max() < 1e-14

def test_matmul_nt():
    a = np.random.rand(m, k)
    b = np.random.rand(n, k)
    c = matmul.matmul_nt(a, b)
    assert abs(a.dot(b.T) - c).max() < 1e-14

def test_matmul_tn():
    a = np.random.rand(k, m)
    b = np.random.rand(k, n)
    c = matmul.matmul_tn(a, b)
    assert abs(a.T.dot(b) - c).max() < 1e-14
