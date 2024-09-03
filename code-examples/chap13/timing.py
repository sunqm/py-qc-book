import time
import numpy as np
from py_qc_book.chap12.analytical_integrals.v5.basis import Molecule, n_cart
from py_qc_book.chap12.analytical_integrals.v5.eri_OS_rys import get_eri_tensor
import j8fold
import k8fold
import j8fold_v2
import k8fold_v2

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
get_eri_tensor(gtos[:1])

t0 = time.perf_counter()
eri_tensor = get_eri_tensor(gtos)
nao = eri_tensor.shape[0]
dm = np.ones((nao, nao))
jref = np.einsum('ijkl,ji->kl', eri_tensor, dm)
kref = np.einsum('ijkl,jk->il', eri_tensor, dm)
t1 = time.perf_counter()
print(t1 - t0)

jmat = j8fold.build_j_8fold(gtos, dm)
t2 = time.perf_counter()
kmat = k8fold.build_k_8fold(gtos, dm)
t3 = time.perf_counter()
print(abs(jref - jmat).max())
print(abs(kref - kmat).max())
print(t2 - t1, t3 - t2)

# Blocking
t1 = time.perf_counter()
jmat = j8fold_v2.build_j_8fold(gtos, dm)
t2 = time.perf_counter()
kmat = k8fold_v2.build_k_8fold(gtos, dm)
t3 = time.perf_counter()
print(t2 - t1, t3 - t2)

import pyscf
from pyscf.scf.hf import get_jk
mol = pyscf.M(atom=benzene, basis={'C': '6-31g*', 'H': '6-31g'}, cart=True)
t0 = time.perf_counter()
v = get_jk(mol, dm)
t1 = time.perf_counter()
print(t1 - t0)
