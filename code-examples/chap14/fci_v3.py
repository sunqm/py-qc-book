import math
from itertools import product
from functools import lru_cache
from typing import List
import numpy as np
import numba

einsum = np.einsum

class String:
    def __init__(self, occupied_orbitals: List):
        self.occupied_orbitals = set(occupied_orbitals)

    def __repr__(self):
        return f'String{self.occupied_orbitals}'

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

    def as_bin(self):
        binstr = 0
        for i in self.occupied_orbitals:
            binstr |= (1 << i)
        return binstr

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

@lru_cache(200)
def make_Elt(norb, nelec):
    '''The lookup table for non-zero elements of E tensor'''
    strings = make_strings(norb, nelec)
    strings_address = {s.as_bin(): i for i, s in enumerate(strings)}

    Elt = []
    for k, binI in enumerate(strings_address):
        table_k = []
        occs = []
        uoccs = []
        sign_cache = []
        sign = 1
        for i in reversed(range(norb)):
            sign_cache.append(sign)
            if (1 << i) & binI:
                occs.append(i)
                sign = -sign
            else:
                uoccs.append(i)
        sign_cache = sign_cache[::-1]
        occs = occs[::-1]
        uoccs = uoccs[::-1]

        for p in occs:
            table_k.append([p, p, k, 1])
        for q, p in product(occs, uoccs):
            binJ = binI ^ (1 << q) ^ (1 << p)
            if p > q:
                sign = sign_cache[p] * sign_cache[q]
            else:
                sign = -sign_cache[p] * sign_cache[q]
            table_k.append([p, q, strings_address[binJ], sign])
        Elt.append(table_k)
    return np.array(Elt, dtype=int)

def merge_h1_eri(h, eri, nelec):
    v = eri * .5
    if nelec > 0:
        f = (h - einsum('prrq->pq', eri) * .5) / (2 * nelec)
        for k in range(eri.shape[0]):
            v[k,k,:,:] += f
            v[:,:,k,k] += f
    return v

@numba.njit
def _compute_hc_block1(v, fciwfn, norb, Elt, blocksize=40):
    Elt_a, Elt_b = Elt
    na, nb = fciwfn.shape
    sigma = np.zeros_like(fciwfn)

    for Ka0 in range(0, na, blocksize):
        ma = min(na - Ka0, blocksize)
        Ka1 = Ka0 + ma
        for Kb0 in range(0, nb, blocksize):
            mb = min(nb - Kb0, blocksize)
            Kb1 = Kb0 + mb
            d = np.zeros((norb,norb,ma,mb))

            for I, tab in enumerate(Elt_a[Ka0:Ka1]):
                for a, i, J, sign in tab:
                    for K in range(mb):
                        d[i,a,I,K] += sign * fciwfn[J,Kb0+K]
            for I, tab in enumerate(Elt_b[Kb0:Kb1]):
                for a, i, J, sign in tab:
                    for K in range(ma):
                        d[i,a,K,I] += sign * fciwfn[Ka0+K,J]

            g = v.dot(d.reshape(norb**2,-1)).reshape(norb,norb,ma,mb)

            for I, tab in enumerate(Elt_a[Ka0:Ka1]):
                for a, i, J, sign in tab:
                    for K in range(mb):
                        sigma[J,Kb0+K] += sign * g[a,i,I,K]
            for I, tab in enumerate(Elt_b[Kb0:Kb1]):
                for a, i, J, sign in tab:
                    for K in range(ma):
                        sigma[Ka0+K,J] += sign * g[a,i,K,I]
    return sigma

@numba.njit
def _compute_hc_block2(v, fciwfn, norb, Elt, blocksize=40):
    Elt_a, Elt_b = Elt
    na, nb = fciwfn.shape
    sigma = np.zeros_like(fciwfn)
    d_buf = np.empty(norb**2*blocksize**2)
    g_buf = np.empty(norb**2*blocksize**2)

    for Ka0 in range(0, na, blocksize):
        ma = min(na - Ka0, blocksize)
        Ka1 = Ka0 + ma
        for Kb0 in range(0, nb, blocksize):
            mb = min(nb - Kb0, blocksize)
            Kb1 = Kb0 + mb
            d = d_buf[:norb*norb*ma*mb].reshape(norb,norb,ma,mb)
            d[:] = 0.

            for I, tab in enumerate(Elt_a[Ka0:Ka1]):
                for a, i, J, sign in tab:
                    for K in range(mb):
                        d[i,a,I,K] += sign * fciwfn[J,Kb0+K]
            for I, tab in enumerate(Elt_b[Kb0:Kb1]):
                for a, i, J, sign in tab:
                    for K in range(ma):
                        d[i,a,K,I] += sign * fciwfn[Ka0+K,J]

            g = g_buf[:norb*norb*ma*mb].reshape(norb*norb,ma*mb)
            g = np.dot(v, d.reshape(norb**2,-1), out=g).reshape(norb,norb,ma,mb)
            d_buf, g_buf = g_buf, d_buf

            for I, tab in enumerate(Elt_a[Ka0:Ka1]):
                for a, i, J, sign in tab:
                    for K in range(mb):
                        sigma[J,Kb0+K] += sign * g[a,i,I,K]
            for I, tab in enumerate(Elt_b[Kb0:Kb1]):
                for a, i, J, sign in tab:
                    for K in range(ma):
                        sigma[Ka0+K,J] += sign * g[a,i,K,I]
    return sigma

def compute_hc(h1, eri, fciwfn, norb, nelec_a, nelec_b, blocksize=40):
    Elt_a = make_Elt(norb, nelec_a)
    Elt_b = make_Elt(norb, nelec_b)
    v = merge_h1_eri(h1, eri, nelec_a + nelec_b).reshape(norb**2,-1)
    return _compute_hc_block2(v, fciwfn, norb, (Elt_a, Elt_b), blocksize)
