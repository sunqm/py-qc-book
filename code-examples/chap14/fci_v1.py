from functools import lru_cache
from typing import List
import numpy as np
from py_qc_book.chap13.davidson import davidson

class String:
    def __init__(self, occupied_orbitals: List):
        self.occupied_orbitals = set(occupied_orbitals)

    def __repr__(self):
        return f'Det{self.occupied_orbitals}'

    @classmethod
    def vacuum(cls):
        return cls(set())

    @classmethod
    def fully_occupied(cls, n):
        return cls(set(range(n)))

    def add_occupancy(self, orbital_id):
        assert orbital_id not in self.occupied_orbitals
        return String(self.occupied_orbitals.union([orbital_id]))

    def annihilate(self, orbital_id):
        '''Apply an annihilation operator. Returns the sign and a new determinant.'''
        if orbital_id not in self.occupied_orbitals:
            return 0, String.vacuum()
        sign = (-1) ** sum(i > orbital_id for i in self.occupied_orbitals)
        return sign, String(self.occupied_orbitals.difference([orbital_id]))

    def create(self, orbital_id):
        '''Apply a creation operator. Returns the sign and a new determinant.'''
        if orbital_id in self.occupied_orbitals:
            return 0., String.vacuum()
        sign = (-1) ** sum(i > orbital_id for i in self.occupied_orbitals)
        return sign, String(self.occupied_orbitals.union([orbital_id]))

    def __hash__(self):
        return hash(tuple(self.occupied_orbitals))

    def __eq__(self, other):
        return self.occupied_orbitals == other.occupied_orbitals

@lru_cache(200)
def make_strings(norb: int, noccupied: int):
    assert norb >= noccupied
    if norb == 0:
        return [String.vacuum()]
    elif noccupied == 0:
        return [String.vacuum()]
    elif norb == noccupied:
        return [String.fully_occupied(norb)]
    return (make_strings(norb-1, noccupied) +
            [s.add_occupancy(norb-1) for s in make_strings(norb-1, noccupied-1)])

def Etensor_value(create_p: int, annihilate_q: int, strI: String, strJ: String):
    r'''The value of <stringI|p^\dagger q|stringJ>'''
    sign1, strK1 = strJ.annihilate(annihilate_q)
    sign2, strK2 = strK1.create(create_p)
    if strK2 == strI:
        return sign1 * sign2
    else:
        return 0

def make_Etensor(norb, nelec):
    strings = make_strings(norb, nelec)
    na = len(strings)
    Et = np.zeros((norb,norb,na,na))
    for p in range(norb):
        for q in range(norb):
            for i, strI in enumerate(strings):
                for j, strJ in enumerate(strings):
                    Et[p,q,i,j] = Etensor_value(p, q, strI, strJ)
    return Et

def merge_h1_eri(h, eri, nelec):
    v = eri * .5
    if nelec > 0:
        f = (h - einsum('prrq->pq', eri) * .5) / (2 * nelec)
        for k in range(eri.shape[0]):
            v[k,k,:,:] += f
            v[:,:,k,k] += f
    return v

def compute_hc(h1, eri, fciwfn, norb, nelec_a, nelec_b):
    Etensor_a = make_Etensor(norb, nelec_a)
    Etensor_b = make_Etensor(norb, nelec_b)
    na = Etensor_a.shape[-1]
    nb = Etensor_b.shape[-1]
    fciwfn = fciwfn.reshape(na, nb)

    d = einsum('pqKI,IJ->pqKJ', Etensor_a, fciwfn)
    d += einsum('pqKJ,IJ->pqIK', Etensor_b, fciwfn)

    v = merge_h1_eri(h1, eri, nelec_a + nelec_b)
    g = einsum('pqrs,rsIJ->pqIJ', v, d)

    sigma = einsum('pqKI,pqIJ->KJ', Etensor_a, g)
    sigma += einsum('pqKJ,pqIJ->IK', Etensor_b, g)
    return sigma

def make_hdiag(h1, eri, norb, nelec_a, nelec_b):
    strs_a = make_strings(norb, nelec_a)
    strs_b = make_strings(norb, nelec_b)
    occs_a = [np.array(list(str_a.occupied_orbitals)) for str_a in strs_a]
    occs_b = [np.array(list(str_b.occupied_orbitals)) for str_b in strs_b]
    diagj = einsum('iijj->ij', eri)
    diagk = einsum('ijji->ij', eri)
    hdiag = []
    for aocc in occs_a:
        for bocc in occs_b:
            e1 = h1[aocc,aocc].sum() + h1[bocc,bocc].sum()
            e2 =(diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() +
                 diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() -
                 diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum())
            hdiag.append(e1 + e2*.5)
    return np.array(hdiag)

def FCI_solve(h1, eri, norb, nelec_a, nelec_b):
    # matrix-vector product
    def matvec(x):
        hc = compute_hc(h1, eri, x, norb, nelec_a, nelec_b)
        return hc.ravel()

    # Diagonal elements
    h_diag = make_hdiag(h1, eri, norb, nelec_a, nelec_b)

    e, wfn = davidson(matvec, h_diag)
    return e, wfn

def einsum(*args):
    return np.einsum(*args, optimize=True)
