import numpy as np
import matmul
import llvmlite.binding as llvm
#llvm.set_option('', '--debug-only=loop-vectorize')

def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print('No instructions found')

matmul.matmul_nn_v3(np.eye(3), np.eye(3))
matmul.matmul_nn_tiling_simd(np.eye(3), np.eye(3))

find_instr(matmul.matmul_nn_v3, 'ymm')
find_instr(matmul.matmul_nn_tiling_simd, 'ymm')
