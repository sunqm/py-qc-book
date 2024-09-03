from py_qc_book.chap12.analytical_integrals.v5.basis import Molecule
from simple_scf import RHF, scf_iter

def test_scf():
    xyz = '''
    N 0.  0 0
    N 1.5 0 0'''
    mol = Molecule.from_xyz(xyz)
    gtos = mol.assign_basis({'N': '6-31g'})
    model = RHF(mol, gtos)
    wfn = scf_iter(model)
    print('RHF energy', model.total_energy(wfn))

    model = RHF.restore(model.chkfile, model.diis.filename)
    scf_iter(model, model.wfn)
    print('RHF energy', model.total_energy(wfn))

    import pyscf
    mol = pyscf.M(atom=list(zip(mol.elements, mol.coordinates)),
                       basis='631g', cart=True, unit='B')
    mol.RHF().set(init_guess='hcore').run()
