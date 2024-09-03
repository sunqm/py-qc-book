import tempfile
import numpy as np
import pyscf
from py_qc_book.chap13.diis import DIIS

def update_CCD_amplitudes(H, t2, level_shift=0):
    nvir, nocc = t2.shape[1:3]
    fock = H['fock']
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]
    e_o = foo.diagonal()
    e_v = fvv.diagonal() + level_shift

    Fvv = fvv - .5 * einsum('klcd,bdkl->bc', H['oovv'], t2)
    Foo = foo + .5 * einsum('klcd,cdjl->kj', H['oovv'], t2)
    Fvv[np.diag_indices(nvir)] -= e_v
    Foo[np.diag_indices(nocc)] -= e_o

    t2out = .25 * H['vvoo']
    t2out -= einsum('bkcj,acik->abij', H['vovo'], t2)
    t2out += .5 * einsum('bc,acij->abij', Fvv, t2)
    t2out -= .5 * einsum('kj,abik->abij', Foo, t2)
    t2out += .5 * einsum('klcd,acik,bdjl->abij', H['oovv'], t2, t2)
    t2out = t2out - t2out.transpose(0,1,3,2)
    t2out = t2out - t2out.transpose(1,0,2,3)
    oooo = .5 * einsum('klcd,cdij->ijkl', H['oovv'], t2) + np.asarray(H['oooo'])
    t2out += .5 * einsum('ijkl,abkl->abij', oooo, t2)
    t2out += .5 * einsum('abcd,cdij->abij', H['vvvv'], t2)

    t2out /= e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None]
    return t2out

def get_CCD_corr_energy(H, t2):
    return .25 * einsum('ijab,abij->', H['oovv'], t2)

def mo_integrals(mol: pyscf.gto.Mole, orbitals):
    '''MO integrals in physists notation <pq||rs>'''
    nmo = orbitals.shape[1]
    eri = np.zeros([nmo*2]*4)
    eri[ ::2, ::2, ::2, ::2] = eri[ ::2, ::2,1::2,1::2] = \
    eri[1::2,1::2, ::2, ::2] = eri[1::2,1::2,1::2,1::2] = \
        pyscf.ao2mo.kernel(mol, orbitals, compact=False).reshape([nmo]*4)
    eri = eri.transpose(0,2,1,3) - eri.transpose(2,0,1,3)

    no = mol.nelectron
    H = {}
    H['vvoo'] = vvoo = eri[no:,no:,:no,:no]
    H['oovv'] = vvoo.conj().transpose(2,3,0,1)
    H['vovo'] = eri[no:,:no,no:,:no]
    H['oooo'] = eri[:no,:no,:no,:no]
    H['vvvv'] = eri[no:,no:,no:,no:]

    hcore = pyscf.scf.hf.get_hcore(mol)
    hcore = einsum('pq,pi,qj->ij', hcore, orbitals, orbitals)
    hcore_mo = np.zeros([nmo*2]*2)
    hcore_mo[::2,::2] = hcore_mo[1::2,1::2] = hcore
    H['fock'] = hcore_mo + einsum('ipiq->pq', eri[:no,:,:no,:])
    return H

def einsum(*args):
    return np.einsum(*args, optimize=True)

def mp2(H):
    nocc = H['oooo'].shape[0]
    fock = H['fock']
    e_o = fock.diagonal()[:nocc]
    e_v = fock.diagonal()[nocc:]
    eijab = e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None]
    t2 = np.asarray(H['vvoo']) / eijab
    e = get_CCD_corr_energy(H, t2)
    return e, t2

def CCD_solve(mf: pyscf.scf.hf.RHF, conv_tol=1e-5, max_cycle=100):
    '''A fixed-point iteration solver for spin-orbital CCD'''
    mol = mf.mol
    orbitals = mf.mo_coeff
    e_hf = mf.e_tot

    H = mo_integrals(mol, orbitals)
    e_corr, t2 = mp2(H) # initial guess
    e_ccd = e_hf + e_corr
    print(f'E(MP2)={e_ccd}')

    with tempfile.TemporaryDirectory() as tmpdir:
        diis = DIIS(f'{tmpdir}/diis')
        for cycle in range(max_cycle):
            t2, t2_prev = update_CCD_amplitudes(H, t2), t2
            e_ccd, e_prev = get_CCD_corr_energy(H, t2) + e_hf, e_ccd
            print(f'{cycle=}, E(CCD)={e_ccd}, dE={e_ccd-e_prev}')
            if abs(t2 - t2_prev).max() < conv_tol:
                break
            t2 = diis.update(t2 - t2_prev, t2)
    return e_ccd

if __name__ == '__main__':
    mol = pyscf.M(atom='N 0. 0 0; N 1.5 0 0', basis='cc-pvdz')
    mf = mol.RHF().run()
    e_ccd = CCD_solve(mf)
    assert abs(e_ccd - -109.0822455) < 1e-6
