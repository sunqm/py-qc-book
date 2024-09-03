from typing import List
import numpy as np
import numba
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
            val = primitive_overlap(li, lj, ai, aj, Ra, Rb)
            S += ci * cj * val
    return S

def primitive_overlap(li: int, lj: int, ai: float, aj: float, Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    Rpb = Rp - Rb
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab**2)

    lij = li + lj
    S = np.zeros((lij+1, lj+1, 3))
    S[0,0] = (np.pi/aij)**.5 * Kab

    if lij > 0:
        S[1,0] = Rpa * S[0,0]
        for i in range(1, lij):
            S[i+1,0] = Rpa * S[i,0] + i/(2*aij) * S[i-1,0]
    for j in range(1, lj+1):
        for i in range(lij+1-j):
            S[i,j] = Rab * S[i,j-1] + S[i+1,j-1]

    nfi = n_cart(li)
    nfj = n_cart(lj)
    ovlp = np.zeros((nfi, nfj))
    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            ovlp[i,j] = S[ix,jx,0] * S[iy,jy,1] * S[iz,jz,2]
    return ovlp
