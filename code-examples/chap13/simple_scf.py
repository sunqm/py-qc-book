from collections import deque
from functools import lru_cache
import tempfile
import pickle
import numpy as np
import scipy.linalg
import json
import h5py
from py_qc_book.chap12.analytical_integrals.v5.eri_OS_rys import get_eri_tensor
from py_qc_book.chap12.analytical_integrals.v5 import overlap_MD
from py_qc_book.chap12.analytical_integrals.v5 import coulomb_1e_MD
from py_qc_book.chap13.diis import DIIS, _HashableVector

einsum = np.einsum

class SCFWavefunction:
    gtos = None
    orbitals = None
    energies = None
    occupancies = None
    density_matrices = None

def scf_iter(model, wfn: SCFWavefunction = None) -> SCFWavefunction:
    hcore, s = model.get_hcore_s()
    if wfn is None:
        wfn = model.get_initial_guess()
    converged = False
    while not converged:
        f = model.get_fock(wfn, hcore, s)
        mo_orbitals, mo_energies = model.eigen(f, s)
        wfn, wfn_old = model.make_wfn(mo_orbitals, mo_energies), wfn
        if model.check_convergence(wfn, wfn_old):
            converged = True
    return wfn

class RHF:
    def __init__(self, mol, gtos):
        self.mol = mol
        self.gtos = gtos
        self.threshold = 1e-7
        self.chkfile = tempfile.mktemp()
        diisfile = tempfile.mktemp()
        print(f'Checkpoint file is {self.chkfile}. DIIS is saved in {diisfile}')
        self.diis = CDIIS(diisfile)

    @lru_cache
    def get_hcore_s(self):
        s = overlap_MD.get_matrix(self.gtos, overlap_MD.primitive_overlap)
        t = overlap_MD.get_matrix(self.gtos, overlap_MD.primitive_kinetic)
        v = sum(-z * coulomb_1e_MD.get_matrix(self.gtos, Rc)
                for z, Rc in zip(self.mol.nuclear_charges, self.mol.coordinates))
        return t+v, s

    def get_initial_guess(self):
        h, s = self.get_hcore_s()
        c, e = self.eigen(h, s)
        return self.make_wfn(c, e)

    @property
    @lru_cache
    def eri_tensor(self):
        return get_eri_tensor(self.gtos)

    @lru_cache(2)
    def get_jk(self, wfn):
        dm = wfn.density_matrices
        j = einsum('ijkl,lk->ij', self.eri_tensor, dm)
        k = einsum('ijkl,jk->il', self.eri_tensor, dm)
        return j, k

    def get_veff(self, wfn):
        j, k = self.get_jk(wfn)
        return j - k * .5

    def get_fock(self, wfn, hcore, s):
        veff = self.get_veff(wfn)
        f = hcore + veff
        self.diis.update(f, wfn.density_matrices, s)
        return f

    def eigen(self, fock, overlap):
        e, c = scipy.linalg.eigh(fock, overlap)
        return c, e

    def make_wfn(self, orbitals, energies):
        wfn = RestrictedCloseShell(self, orbitals, energies)
        with open(self.chkfile, 'wb') as f:
            pickle.dump({'wfn': wfn, 'mol': self.mol, 'gtos': self.gtos}, f)
        return wfn

    def check_convergence(self, wfn, wfn_old):
        t1 = t2 = self.threshold * 1e3
        t3 = self.threshold
        return (np.linalg.norm(wfn.density_matrices - wfn_old.density_matrices) < t1
                and np.linalg.norm(self.orbital_gradients(wfn)) < t2
                and abs(self.total_energy(wfn) - self.total_energy(wfn_old)) < t3)

    def total_energy(self, wfn):
        hcore, s = self.get_hcore_s()
        j, k = self.get_jk(wfn)
        dm = wfn.density_matrices
        e = einsum('ij,ji', hcore, dm)
        e += einsum('ij,ji', j, dm) * .5
        e -= einsum('ij,ji', k, dm) * .25
        e += self.mol.nuclear_repulsion_energy()
        return e

    def orbital_gradients(self, wfn):
        hcore, s = self.get_hcore_s()
        fock = self.get_fock(wfn, hcore, s)
        fock_mo = wfn.orbitals.T.dot(fock).dot(wfn.orbitals) # Fock in MO basis
        return fock_mo[(wfn.occupancies!=0) & (wfn.occupancies[:,None]==0)] * 2.

    @classmethod
    def restore(cls, chkfile, diisfile=None):
        with open(chkfile, 'rb') as f:
            attrs = pickle.load(f)
        obj = cls(attrs['mol'], attrs['gtos'])
        obj.wfn = attrs['wfn']
        if diisfile is not None:
            obj.diis = CDIIS.restore(diisfile)
        return obj

class RestrictedCloseShell(SCFWavefunction):
    def __init__(self, mf, orbitals, energies):
        self.gtos = mf.gtos
        self.orbitals = orbitals
        self.energies = energies
        nocc = sum(mf.mol.nuclear_charges) // 2
        self.occupancies = np.zeros_like(energies)
        self.occupancies[:nocc] = 2.

    @property
    @lru_cache
    def density_matrices(self):
        c = self.orbitals
        return (c * self.occupancies).dot(c.T)

class CDIIS(DIIS):
    def __init__(self, filename, max_space=8):
        super().__init__(filename, max_space)
        self.c = None # The orthonormal basis for error vectors

    def update(self, f, d, s):
        with h5py.File(self.filename, mode='a') as h5f:
            if self.c is None:
                _, self.c = scipy.linalg.eigh(s)
                h5f['c'] = self.c
            errvec = f.dot(d).dot(s)
            errvec = errvec - errvec.T # FDS - SDF
            # Transforms error vector to orthonormal basis
            errvec = self.c.T.dot(errvec).dot(self.c).ravel()

            head, self.head = self.head, (self.head + 1) % self.max_space
            self.keys.append(head)
            if f'e{head}' in h5f:
                # Reuse existing datasets
                h5f[f'e{head}'][:] = errvec
                h5f[f't{head}'][:] = f
            else:
                h5f[f'e{head}'] = errvec
                h5f[f't{head}'] = f
            if 'metadata' in h5f:
                del h5f['metadata']
            h5f['metadata'] = self.dumps()
            h5f.flush()

            errvecs = [_HashableVector(h5f[f'e{key}']) for key in self.keys]
            space = len(self.keys)
            B = np.zeros((space+1, space+1))
            B[-1,:-1] = B[:-1,-1] = 1.
            for i, e1 in enumerate(errvecs):
                for j, e2 in enumerate(errvecs):
                    if j < i:
                        continue
                    B[i,j] = B[j,i] = e1.dot(e2)

            while np.linalg.cond(B) > 1e12:
                B = B[1:,1:]
                self.keys.popleft()

            g = np.zeros(len(self.keys)+1)
            g[-1] = 1
            c = scipy.linalg.solve(B, g, assume_a='sym')[:-1]

            sol = np.zeros_like(f)
            for key, x in zip(self.keys, c):
                sol += h5f[f't{key}'][:] * x
            return sol

    @classmethod
    def restore(cls, filename):
        with h5py.File(filename, mode='r') as f:
            attrs = json.loads(f['metadata'][()])
            obj = cls(filename)
            obj.keys = deque(attrs['keys'], maxlen=attrs['max_space'])
            obj.head = attrs['head']
            obj.c = f['c'][()]
        return obj
