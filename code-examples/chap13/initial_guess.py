import numpy as np
import scipy.linalg
import basis_set_exchange as bse
from py_qc_book.chap12.analytical_integrals.v5.basis import (
    Molecule, CGTO, num_functions)
from py_qc_book.chap12.analytical_integrals.v5 import overlap_MD
from .simple_scf import scf_iter, SCFWavefunction
from .uhf import UHF

class InitialGuess(SCFWavefunction):
    def __init__(self, dm):
        self.density_matrices = dm

    @property
    def orbitals(self):
        raise RuntimeError('Orbitals not available in initial guess')

def ano_Fe(r, with_3d=True):
    gtos = []
    data = bse.get_basis('ANO-RCC', elements=['Fe'])
    for elem_basis in data['elements'].values():
        for raw_basis in elem_basis['electron_shells']:
            l = int(raw_basis['angular_momentum'][0])
            match l:
                case 0: # 1s - 4s
                    coefs = raw_basis['coefficients'][:4]
                case 1: # 2p, 3p
                    coefs = raw_basis['coefficients'][:2]
                case 2: # 3d
                    if not with_3d:
                        continue
                    coefs = raw_basis['coefficients'][:1]
                case _:
                    continue

            for c in coefs:
                gtos.append(CGTO.from_dict({
                    'angular_momentum': l,
                    'exponents': raw_basis['exponents'],
                    'coefficients': c,
                    'coordinates': r,
                }))
    return gtos

# Assign all 3d orbitals for one iron atom with spin-up electrons, and the other
# iron atom with spin-down electrons.
xyz = '''
Fe 0.  0 0
Fe 2.5 0 0'''
mol = Molecule.from_xyz(xyz)

gtos = mol.assign_basis({'Fe': '6-31G'})
nao = num_functions(gtos)
rs = mol.coordinates
with_3d = True

# spin-up density matrix
s = overlap_MD.get_matrix(
    gtos + ano_Fe(rs[0], with_3d) + ano_Fe(rs[1], not with_3d),
    overlap_MD.primitive_overlap)
s_t = s[:nao,:nao]
s_ta = s[:nao,nao:]
c = scipy.linalg.solve(s_t, s_ta)
dm_up = np.einsum('pi,qi->pq', c, c)

# spin-down density matrix
s = overlap_MD.get_matrix(
    gtos + ano_Fe(rs[0], not with_3d) + ano_Fe(rs[1], with_3d),
    overlap_MD.primitive_overlap)
s_t = s[:nao,:nao]
s_ta = s[:nao,nao:]
c = scipy.linalg.solve(s_t, s_ta)
dm_down = np.einsum('pi,qi->pq', c, c)

dm = (dm_up, dm_down)
wfn = scf_iter(UHF, InitialGuess(dm))
