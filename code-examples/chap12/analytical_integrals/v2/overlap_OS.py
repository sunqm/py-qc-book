from typing import List
import numpy as np
from basis import CGTO, n_cart, iter_cart_xyz


def overlap_matrix(gtos: List[CGTO]) -> np.ndarray:
    S = []
    for bas_i in gtos:
        S.append([contracted_overlap(bas_i, bas_j) for bas_j in gtos])
    return np.block(S)

def contracted_overlap(bas_i: CGTO, bas_j: CGTO) -> np.ndarray:
    li, lj = bas_i.angular_momentum, bas_j.angular_momentum
    norm_ci = bas_i.norm_coefficients
    norm_cj = bas_j.norm_coefficients
    Ra, Rb = bas_i.coordinates, bas_j.coordinates
    nfi = n_cart(li)
    nfj = n_cart(lj)
    S = np.zeros((nfi, nfj))

    for ai, ci in zip(bas_i.exponents, norm_ci):
        for aj, cj in zip(bas_j.exponents, norm_cj):
            S += ci * cj * primitive_overlap(li, lj, ai, aj, Ra, Rb)
    return S

def primitive_overlap(li: int, lj: int, ai: float, aj: float, Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab**2)

    lij = li + lj
    S = np.zeros((lij+1, lj+1, 3))
    for j in range(lj+1):
        for i in range(lij+1-j):
            if j > 0:
                S[i,j] = S[i+1,j-1] + Rab * S[i,j-1]
            elif i > 1:
                S[i,j] = Rpa * S[i-1,j] + (i-1)/(2*aij) * S[i-2,j]
            elif i == 1:
                S[1,0] = Rpa * S[0,0]
            else:
                S[0,0] = (np.pi/aij)**.5 * Kab

    nfi = n_cart(li)
    nfj = n_cart(lj)
    ovlp = np.zeros((nfi, nfj))
    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            Ix = S[ix,jx,0]
            Iy = S[iy,jy,1]
            Iz = S[iz,jz,2]
            ovlp[i,j] = Ix * Iy * Iz
    return ovlp

def lru_cache(f):
    cache = {}
    def f_cached(*args):
        if args in cache:
            return cache[args]
        result = f(*args)
        cache[args] = result
        return result
    f_cached._cache = cache
    return f_cached

def lru_cache_keys(li: int, lj: int, ai: float, aj: float, Ra: np.ndarray, Rb: np.ndarray):
    '''
    Using the custom lru_cache function to inspect the recursive DP function
    '''
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab**2)

    @lru_cache
    def get_S(i, j):
        if i < 0 or j < 0:
            return 0
        if i == 0 and j == 0:
            return (np.pi/aij)**.5 * Kab
        if j == 0:
            if i == 1:
                return Rpa * get_S(i-1, j)
            return (i-1)/(2*aij) * get_S(i-2, j) + Rpa * get_S(i-1, j)
        return Rab * get_S(i, j-1) + get_S(i+1, j-1)

    get_S(li, lj)
    return get_S._cache.keys()

if __name__ == "__main__":
    print(lru_cache_keys(2, 2, 1., 1., np.zeros(3), np.zeros(3)))
