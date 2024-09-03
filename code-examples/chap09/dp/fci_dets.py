from functools import lru_cache
from typing import List

class Determinant:
    def __init__(self, occupied_orbitals: List):
        self.occupied_orbitals = occupied_orbitals

    def __repr__(self):
        return f'Det{self.occupied_orbitals}'

    @classmethod
    def vacuum(cls):
        return cls([])

    @classmethod
    def fully_occupied(cls, n):
        return cls(list(range(n)))

    def add_occupancy(self, orbital_id):
        assert orbital_id not in self.occupied_orbitals
        return Determinant(sorted(self.occupied_orbitals + [orbital_id]))

@lru_cache
def dets(n, m) -> List[Determinant]:
    assert n >= m
    if n == 0:
        return [Determinant.vacuum()]
    elif m == 0:
        return [Determinant.vacuum()]
    elif n == m:
        return [Determinant.fully_occupied(n)]
    return (dets(n-1, m) +
            [det.add_occupancy(n-1) for det in dets(n-1, m-1)])
