from typing import List
import numpy as np
from scipy.special import roots_hermite
from basis import CGTO, n_cart, iter_cart_xyz

def overlap_matrix(gtos: List[CGTO]) -> np.ndarray:
    S = []
    for bas_i in gtos:
        S.append([contracted_overlap(bas_i, bas_j) for bas_j in gtos])
    return np.block(S)

def contracted_overlap(bas_i, bas_j) -> np.ndarray:
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

def primitive_overlap(li, lj, ai, aj, Ra, Rb) -> np.ndarray:
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    Rpb = Rp - Rb
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab**2)

    nfi = n_cart(li)
    nfj = n_cart(lj)
    S = np.zeros((nfi, nfj))

    r, w = roots_hermite((li+lj+2)//2)
    rt = r / aij**.5
    wt = w / aij**.5

    poly_i = [[(x+rt)**n for n in range(li+1)] for x in Rpa]
    poly_j = [[(x+rt)**n for n in range(lj+1)] for x in Rpb]
    I2d = np.einsum('x,xin,xjn->xijn', Kab, poly_i, poly_j)
    Ix, Iy, Iz = np.einsum('n,xijn->xij', wt, I2d)

    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            S[i,j] = Ix[ix,jx] * Iy[iy,jy] * Iz[iz,jz]
    return S
