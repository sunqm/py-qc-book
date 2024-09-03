from typing import List
import numpy as np
from py_qc_book.chap12.analytical_integrals.v5.basis import Molecule, CGTO
from py_qc_book.chap12.analytical_integrals.v5 import coulomb_1e_MD
from py_qc_book.chap13.simple_scf import SCFWavefunction, RHF, scf_iter
from py_qc_book.chap13.eval_gto import eval_gtos
from py_qc_book.chap03.render_cube import render_cube

def eval_density(mol: Molecule, gtos: List[CGTO], wfn: SCFWavefunction, grids):
    dm = wfn.density_matrices
    ao = eval_gtos(gtos, grids)[0]
    rho = np.einsum('pg,pq,qg->g', ao, dm, ao)
    return rho

def eval_mep(mol: Molecule, gtos: List[CGTO], wfn: SCFWavefunction, grids):
    dist = np.linalg.norm(grids[:,None,:] - mol.coordinates, axis=2)
    mep = -np.einsum('z,gz->g', mol.nuclear_charges, 1./dist)

    dm = wfn.density_matrices
    for i, r in enumerate(grids):
        v = coulomb_1e_MD.get_matrix(gtos, r)
        mep[i] += np.einsum('ij,ji->', v, dm)
    return mep

if __name__ == '__main__':
    xyz = '''
    N -.75 0 0
    N 0.75 0 0'''
    mol = Molecule.from_xyz(xyz)
    gtos = mol.assign_basis({'N': '6-31g'})
    model = RHF(mol, gtos)
    wfn = scf_iter(model)

    boundary = [[-5., 5.],
                [-5., 5.],
                [-5., 5.]]
    mesh = [20, 20, 20]
    mgrids = np.mgrid[[slice(r[0], r[1], m*1j) for r, m in zip(boundary, mesh)]]
    grids = mgrids.reshape(3, -1).T
    rho = eval_density(mol, gtos, wfn, grids).reshape(mesh)
    mep = eval_mep(mol, gtos, wfn, grids).reshape(mesh)

    origin = mgrids[:,0,0,0]
    voxel = np.array([mgrids[:,1,0,0] - origin,
                      mgrids[:,0,1,0] - origin,
                      mgrids[:,0,0,1] - origin])

    with open('density.cub', 'w') as f:
        render_cube(mol.elements, mol.coordinates, voxel, origin, rho)
    with open('mep.cub', 'w') as f:
        render_cube(mol.elements, mol.coordinates, voxel, origin, mep)
