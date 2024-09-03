import yaml
import numpy as np
from basis import CGTO, n_cart, gto_offsets
import coulomb_1e_MD
import eri_MD
import eri_OS
import eri_rys

prim_gtos = '''
- angular_momentum: 0
  exponents:
  - 0.75
  coefficients:
  - 1.
  coordinates:
  - 0.3
  - 0.0
  - 1.0
- angular_momentum: 1
  exponents:
  - 0.32
  coefficients:
  - 1.
  coordinates:
  - 0.2
  - 1.9
  - 0.0
- angular_momentum: 2
  exponents:
  - 0.51
  coefficients:
  - 1.
  coordinates:
  - 0.5
  - 0.0
  - 0.5
'''

def new_gtos():
    gtos = (CGTO.from_bse('STO-3G', 'C', np.array([0.1, 0.5, 0.8]))
            + CGTO.from_yaml(prim_gtos))
    return gtos

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

def test_coulomb_1e_MD():
    gtos = new_gtos()
    Rc = np.ones(3)
    assert abs(abs(coulomb_1e_MD.get_matrix(gtos, Rc)).sum() - 35.63710447871082) < 1e-9

def test_eri_MD():
    gtos = new_gtos()
    V = get_eri_tensor(eri_MD.contracted_ERI, gtos)
    assert abs(abs(V).sum() - 1277.9590013752425) < 1e-8

def test_eri_OS():
    gtos = new_gtos()
    V = get_eri_tensor(eri_OS.contracted_ERI, gtos)
    assert abs(abs(V).sum() - 1277.9590013752425) < 1e-8

def test_eri_rys():
    gtos = new_gtos()
    V = get_eri_tensor(eri_rys.contracted_ERI, gtos)
    assert abs(abs(V).sum() - 1277.9590013752425) < 1e-8

def test_R_tensor():
    gtos = new_gtos()[3:]
    R = np.array([0.2, 0.5, 0.1])
    assert abs(abs(coulomb_1e_MD.get_R_tensor(0, 0.35, R)).sum() - 0.966075490706908) < 1e-9
    assert abs(abs(coulomb_1e_MD.get_R_tensor(2, 0.35, R)).sum() - 2.253353577364496) < 1e-9
    assert abs(abs(coulomb_1e_MD.get_R_tensor(4, 0.35, R)).sum() - 6.495286612282999) < 1e-9
