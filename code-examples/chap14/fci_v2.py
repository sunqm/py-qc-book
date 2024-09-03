from functools import lru_cache
from typing import List
import numpy as np

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

@lru_cache(10)
def make_Elt(norb, nelec):
    '''The lookup table for non-zero elements of E tensor'''
    strings = make_strings(norb, nelec)
    # To setup the map from string to address,
    # __hash__ and __eq__ methods must be created for String object.
    strings_address = {s: i for i, s in enumerate(strings)}
    Elt = []
    for k, strI in enumerate(strings):
        table_k = []
        occs = strI.occupied_orbitals
        uoccs = [p for p in range(norb) if p not in occs]
        for p in occs:
            table_k.append([p, p, k, 1])
        for q in occs:
            sign1, strK1 = strI.annihilate(q)
            for p in uoccs:
                sign2, strJ = strK1.create(p)
                # Applying a^\dagger_p a_q on address of strI leads to
                # to address of the output string (strJ)
                table_k.append([p, q, strings_address[strJ], sign1*sign2])
        Elt.append(table_k)
    return Elt

def merge_h1_eri(h, eri, nelec):
    v = eri * .5
    if nelec > 0:
        f = (h - einsum('prrq->pq', eri) * .5) / (2 * nelec)
        for k in range(eri.shape[0]):
            v[k,k,:,:] += f
            v[:,:,k,k] += f
    return v

def compute_hc(h1, eri, fciwfn, norb, nelec_a, nelec_b):
    Elt_a = make_Elt(norb, nelec_a)
    Elt_b = make_Elt(norb, nelec_b)
    na = len(Elt_a)
    nb = len(Elt_b)

    d = np.zeros((norb,norb,na,nb))
    for I, tab in enumerate(Elt_a):
        for a, i, J, sign in tab:
            d[a,i,J] += sign * fciwfn[I]
    for I, tab in enumerate(Elt_b):
        for a, i, J, sign in tab:
            d[a,i,:,J] += sign * fciwfn[:,I]

    v = merge_h1_eri(h1, eri, nelec_a + nelec_b)
    g = einsum('pqrs,rsIJ->pqIJ', v, d)

    sigma = np.zeros_like(fciwfn)
    for I, tab in enumerate(Elt_a):
        for a, i, J, sign in tab:
            sigma[J] += sign * g[a,i,I]
    for I, tab in enumerate(Elt_b):
        for a, i, J, sign in tab:
            sigma[:,J] += sign * g[a,i,:,I]
    return sigma

def einsum(*args):
    return np.einsum(*args, optimize=True)
