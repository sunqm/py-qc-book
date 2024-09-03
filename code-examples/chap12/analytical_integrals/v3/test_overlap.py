import yaml
import numpy as np
from basis import CGTO
import overlap_MD
import overlap_OS

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
  - 0.9996722919E-01
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

def test_overlap_MD():
    basis = new_basis()
    assert abs(overlap_MD.overlap_matrix(basis).sum() - 27.282308953984028) < 1e-9

def test_overlap_OS():
    basis = new_basis()
    assert abs(overlap_OS.overlap_matrix(basis).sum() - 27.282308953984028) < 1e-9

def test_E_tensor():
    basis = new_basis()[3:]
    s = 0
    for bas_i in basis:
        for bas_j in basis:
            Et = overlap_MD.get_E_tensor(bas_i.angular_momentum, bas_j.angular_momentum,
                                         bas_i.exponents[0], bas_j.exponents[0],
                                         bas_i.coordinates, bas_j.coordinates)
            s += Et.sum()
    assert abs(s - 40.8402112684493) < 1e-9
