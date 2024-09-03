import time
import numpy as np
import matmul
import matmul_cython
import matmul_pythran


def timer(f, *args):
    # Warm up
    a = np.eye(2)
    f(a, a)

    t0 = time.perf_counter()
    f(*args)
    t1 = time.perf_counter()
    print(t1 - t0)

m, n, k = 5000, 5000, 5000
np.random.seed(2)
a = np.random.rand(m, k)
b = np.random.rand(k, n)
timer(matmul.matmul_nn_v1, a, b)      # 214.9
timer(matmul.matmul_nn_v2, a, b)      # 141.9
timer(matmul.matmul_nn_v3, a, b)      # 77.4
timer(matmul.matmul_nn_tiling, a, b)  # 107.4
timer(matmul.matmul_nn_tiling_unrolled, a, b)  # 57.8
timer(matmul.matmul_nn_tiling_simd, a, b)      # 21.5

timer(matmul_cython.matmul_nn_tiling, a, b)            # 26.8
timer(matmul_cython.matmul_nn_tiling_unrolled, a, b)   # 18.2

timer(matmul_pythran.matmul_nn_v3, a, b)               # 78.7
timer(matmul_pythran.matmul_nn_tiling, a, b)           # 26.8
timer(matmul_pythran.matmul_nn_tiling_unrolled, a, b)  # 18.5
