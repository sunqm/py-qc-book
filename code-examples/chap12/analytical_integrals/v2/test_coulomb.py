import yaml
import numpy as np
from basis import CGTO, n_cart, gto_offsets
import coulomb_1e_MD
import eri_MD
import eri_OS
import eri_rys

C_sto3g = '''
- angular_momentum: 0
  exponents:
  - 0.7161683735E+02
  - 0.1304509632E+02
  - 0.3530512160E+01
  coefficients:
  - 0.1543289673E+00
  - 0.5353281423E+00
  - 0.4446345422E+00
- angular_momentum: 0
  exponents:
  - 0.2941249355E+01
  - 0.6834830964E+00
  - 0.2222899159E+00
  coefficients:
  - -0.9996722919E-01
  - 0.3995128261E+00
  - 0.7001154689E+00
- angular_momentum: 1
  exponents:
  - 0.2941249355E+01
  - 0.6834830964E+00
  - 0.2222899159E+00
  coefficients:
  - 0.1559162750E+00
  - 0.6076837186E+00
  - 0.3919573931E+00
'''

prim_basis = '''
- angular_momentum: 0
  exponents:
  - 0.75
  coefficients:
  - 1.
- angular_momentum: 1
  exponents:
  - 0.32
  coefficients:
  - 1.
- angular_momentum: 2
  exponents:
  - 0.51
  coefficients:
  - 1.
'''

def new_basis():
    r = np.array([[0.1, 0.5, 0.8],
                  [0.3, 0.0, 1.0],
                  [0.2, 1.9, 0.0],
                  [0.5, 0.0, 0.5]])
    basis = []
    for item in yaml.load(C_sto3g, Loader=yaml.CSafeLoader):
        item['coordinates'] = r[0]
        basis.append(CGTO.from_dict(item))

    for i, item in enumerate(yaml.load(prim_basis, Loader=yaml.CSafeLoader)):
        item['coordinates'] = r[i+1]
        basis.append(CGTO.from_dict(item))
    return basis

def get_tensor(f, basis):
    offsets = gto_offsets(basis)
    nao = offsets[-1]
    V = np.empty((nao, nao, nao, nao))
    for i, bas_i in enumerate(basis):
        i0, i1 = offsets[i], offsets[i+1]
        for j, bas_j in enumerate(basis):
            j0, j1 = offsets[j], offsets[j+1]
            for k, bas_k in enumerate(basis):
                k0, k1 = offsets[k], offsets[k+1]
                for l, bas_l in enumerate(basis):
                    l0, l1 = offsets[l], offsets[l+1]
                    V[i0:i1,j0:j1,k0:k1,l0:l1] = f(bas_i, bas_j, bas_k, bas_l)
    return V

def test_coulomb_1e_MD():
    basis = new_basis()
    Rc = np.ones(3)
    assert abs(coulomb_1e_MD.get_matrix(basis, Rc).sum() - 25.447703586595313) < 1e-9

def test_eri_MD():
    basis = new_basis()
    V = get_tensor(eri_MD.contracted_ERI, basis)
    assert abs(V.sum() - 480.579854249) < 1e-8

def test_eri_OS():
    basis = new_basis()
    V = get_tensor(eri_OS.contracted_ERI, basis)
    assert abs(V.sum() - 480.579854249) < 1e-8

def test_eri_rys():
    basis = new_basis()
    V = get_tensor(eri_rys.contracted_ERI, basis)
    assert abs(V.sum() - 480.579854249) < 1e-8

def test_R_tensor():
    basis = new_basis()[3:]
    R = np.array([0.2, 0.5, 0.1])
    assert abs(abs(coulomb_1e_MD.get_R_tensor(0, 0.35, R)).sum() - 0.966075490706908) < 1e-9
    assert abs(abs(coulomb_1e_MD.get_R_tensor(2, 0.35, R)).sum() - 2.253353577364496) < 1e-9
    assert abs(abs(coulomb_1e_MD.get_R_tensor(4, 0.35, R)).sum() - 6.495286612282999) < 1e-9
