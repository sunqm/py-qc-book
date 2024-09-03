import time
import numpy as np
from basis import CGTO, Molecule, n_cart, gto_offsets
import eri_MD
import eri_OS
import eri_rys

def get_eri_tensor(f, gtos):
    offsets = gto_offsets(gtos)
    nao = offsets[-1]
    V = np.empty((nao, nao, nao, nao))
    for i, bas_i in enumerate(gtos):
        i0, i1 = offsets[i], offsets[i+1]
        for j, bas_j in enumerate(gtos):
            j0, j1 = offsets[j], offsets[j+1]
            for k, bas_k in enumerate(gtos):
                k0, k1 = offsets[k], offsets[k+1]
                for l, bas_l in enumerate(gtos):
                    l0, l1 = offsets[l], offsets[l+1]
                    V[i0:i1,j0:j1,k0:k1,l0:l1] = f(bas_i, bas_j, bas_k, bas_l)
    return V

def timing(f, gtos):
    get_eri_tensor(f, gtos[:1]) # warm up
    t0 = time.perf_counter()
    v = get_eri_tensor(f, gtos)
    t1 = time.perf_counter()
    return t1 - t0

benzene = '''
C        0.0000     1.3990     0.0000
C       -1.2095     0.7442     0.0000
C       -1.2095    -0.7442     0.0000
C        0.0000    -1.3990     0.0000
C        1.2095    -0.7442     0.0000
C        1.2095     0.7442     0.0000
H        0.0000     2.4939     0.0000
H       -2.1994     1.3083     0.0000
H       -2.1994    -1.3083     0.0000
H        0.0000    -2.4939     0.0000
H        2.1994    -1.3083     0.0000
H        2.1994     1.3083     0.0000
'''
gtos = Molecule.from_xyz(benzene).assign_basis({'C': '6-31G*', 'H': '6-31G'})

print(timing(eri_MD.contracted_ERI, gtos))
print(timing(eri_OS.contracted_ERI, gtos))
print(timing(eri_rys.contracted_ERI, gtos))
