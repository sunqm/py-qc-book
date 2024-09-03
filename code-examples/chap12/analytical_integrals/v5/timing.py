import time
import numpy as np
from numba.typed import List
from basis import CGTO, jitCGTO, Molecule
import eri_OS_rys

def timing(f_eri_tensor, gtos):
    f_eri_tensor(gtos[:1]) # warm up
    t0 = time.perf_counter()
    v = f_eri_tensor(gtos)
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

# warmup
eri_OS_rys.get_eri_tensor(gtos[:1])

print(timing(eri_OS_rys.get_eri_tensor, gtos))

gtos = List(gto.to_jitCGTO() for gto in gtos)
print(timing(eri_OS_rys.get_eri_tensor_jit, gtos))

from pyscf import gto
mol = gto.M(atom=benzene, basis={'C': '6-31g*', 'H': '6-31g'})
print(mol.nbas, mol.nao_cart())
t0 = time.perf_counter()
v = mol.intor('int2e_cart', aosym='s1')
t1 = time.perf_counter()
print(t1 - t0)
