import tempfile
import threading
from typing import Dict, Union
from contextlib import contextmanager
import numpy as np
import h5py
import pyscf
from py_qc_book.chap13.diis import DIIS

# Using the producer-consumer model, we can implement the readahead function
@contextmanager
def readahead(h5obj, tasks, maxsize=3):
    assert maxsize > 0
    cache = {}
    # This mutex lock is shared by the prefetch function and loader.
    # It must be held whenever mutating the cache.
    mutex = threading.RLock()
    not_full = threading.Condition(mutex)
    not_empty = threading.Condition(mutex)
    terminate = False

    def prefetch(tasks):
        for task in tasks:
            if terminate:
                break
            # Use a condition variable to block the producer.
            # To prevent the cache storing too many items.
            data = np.asarray(h5obj[task])
            with not_full:
                cache[task] = data
                not_empty.notify()
                while len(cache) >= maxsize:
                    not_full.wait()
    daemon = threading.Thread(target=prefetch, args=(tasks,))
    daemon.start()

    def loader(key):
        with not_empty:
            # wait until the required task has been prepared
            if not cache:
                not_empty.wait()
            data = cache.pop(key)
            not_full.notify()
        return data

    yield loader

    terminate = True
    with not_full:
        # release any locks in prefetch, then the terminate condition in
        # prefetch function will be triggered
        not_full.notify()
    daemon.join()

def update_CCD_amplitudes(H: Union[Dict, h5py.Group], t2: np.ndarray):
    nvir, nocc = t2.shape[1:3]
    fock = H['fock']
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]
    e_o = foo.diagonal()
    e_v = fvv.diagonal()

    with readahead(H, [
        'oovv', 'vvoo', 'vovo', 'oooo', 'vvvv',
    ]) as load:
        oovv = load('oovv')
        Fvv = fvv - .5 * einsum('klcd,bdkl->bc', oovv, t2)
        Foo = foo + .5 * einsum('klcd,cdjl->kj', oovv, t2)
        Fvv[np.diag_indices(nvir)] -= e_v
        Foo[np.diag_indices(nocc)] -= e_o

        t2out = .25 * load('vvoo')
        t2out -= einsum('bkcj,acik->abij', load('vovo'), t2)
        t2out += .5 * einsum('bc,acij->abij', Fvv, t2)
        t2out -= .5 * einsum('kj,abik->abij', Foo, t2)
        t2out += .5 * einsum('klcd,acik,bdjl->abij', oovv, t2, t2)
        t2out = t2out - t2out.transpose(0,1,3,2)
        t2out = t2out - t2out.transpose(1,0,2,3)
        oooo = .5 * einsum('klcd,cdij->ijkl', oovv, t2) + load('oooo')
        t2out += .5 * einsum('ijkl,abkl->abij', oooo, t2)
        t2out += .5 * einsum('abcd,cdij->abij', load('vvvv'), t2)

    t2out /= e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None]
    return t2out

def get_CCD_corr_energy(H, t2):
    return .25 * einsum('ijab,abij->', H['oovv'], t2)

def mo_integrals(mol: pyscf.gto.Mole, orbitals, Hfile=None):
    '''MO integrals in physists notation <pq||rs>'''
    no = mol.nelectron
    nmo = orbitals.shape[1]
    eri = np.zeros([nmo*2]*4)
    eri[ ::2, ::2, ::2, ::2] = eri[ ::2, ::2,1::2,1::2] = \
    eri[1::2,1::2, ::2, ::2] = eri[1::2,1::2,1::2,1::2] = \
        pyscf.ao2mo.kernel(mol, orbitals, compact=False).reshape([nmo]*4)
    eri = eri.transpose(0,2,1,3) - eri.transpose(2,0,1,3)

    if Hfile is None:
        Hfile = tempfile.mktemp()
    with h5py.File(Hfile, 'w') as H:
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
    return Hfile

def einsum(*args):
    return np.einsum(*args, optimize=True)

def mp2(H):
    nocc = H['oooo'].shape[0]
    fock = np.asarray(H['fock'])
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

    with tempfile.TemporaryDirectory() as tmpdir:
        Hfile = mo_integrals(mol, orbitals, f'{tmpdir}/H')
        diis = DIIS(f'{tmpdir}/diis')
        e_ccd = e_hf
        with h5py.File(Hfile, 'r') as H:
            e_corr, t2 = mp2(H) # initial guess
            e_ccd = e_hf + e_corr
            print(f'E(MP2)={e_ccd}')

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
