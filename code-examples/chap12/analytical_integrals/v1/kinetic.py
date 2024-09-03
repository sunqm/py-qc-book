from functools import lru_cache
from typing import List
import numpy as np
from basis import CGTO, n_cart, iter_cart_xyz

def get_T_matrix(gtos: List[CGTO]) -> np.ndarray:
    S = []
    for bas_i in gtos:
        S.append([contracted_kinetic(bas_i, bas_j) for bas_j in gtos])
    return np.block(S)

def contracted_kinetic(bas_i: CGTO, bas_j: CGTO) -> np.ndarray:
    li, lj = bas_i.angular_momentum, bas_j.angular_momentum
    norm_ci = bas_i.norm_coefficients
    norm_cj = bas_j.norm_coefficients
    Ra, Rb = bas_i.coordinates, bas_j.coordinates
    nfi = n_cart(li)
    nfj = n_cart(lj)
    S = np.zeros((nfi, nfj))

    for ai, ci in zip(bas_i.exponents, norm_ci):
        for aj, cj in zip(bas_j.exponents, norm_cj):
            S += ci * cj * primitive_kinetic(li, lj, ai, aj, Ra, Rb)
    return S

def primitive_kinetic(li: int, lj: int, ai: float, aj: float, Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    Rpb = Rp - Rb
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab**2)

    nfi = n_cart(li)
    nfj = n_cart(lj)
    T = np.zeros((nfi, nfj))

    @lru_cache(1000)
    def get_S(i, j):
        if i < 0 or j < 0:
            return 0
        if j > 0:
            return get_S(i+1, j-1) + Rab * get_S(i, j-1)
        if i > 1:
            return Rpa * get_S(i-1, j) + (i-1)/(2*aij) * get_S(i-2, j)
        if i == 1:
            return Rpa * get_S(i-1, j)
        return (np.pi/aij)**.5 * Kab

    @lru_cache(1000)
    def get_S_deriv2(i, j):
        I = 4*ai*aj*get_S(i+1, j+1)
        if j > 0:
            I -= 2*ai*j*get_S(i+1,j-1)
        if i > 0:
            I -= 2*aj*i*get_S(i-1,j+1)
        if i > 0 and j > 0:
            I += i*j*get_S(i-1,j-1)
        return I

    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            Ix2 = get_S_deriv2(ix, jx)[0]
            Iy2 = get_S_deriv2(iy, jy)[1]
            Iz2 = get_S_deriv2(iz, jz)[2]
            Ix = get_S(ix, jx)[0]
            Iy = get_S(iy, jy)[1]
            Iz = get_S(iz, jz)[2]
            T[i,j] = Ix2 * Iy * Iz + Ix * Iy2 * Iz + Ix * Iy * Iz2
    T *= .5
    return T
