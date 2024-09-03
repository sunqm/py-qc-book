from typing import List
import numpy as np
import numba
from basis import CGTO, n_cart


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

@numba.njit(cache=True)
def primitive_overlap(li: int, lj: int, ai: float, aj: float, Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab**2)
    Xpa, Ypa, Zpa = Rpa
    Xab, Yab, Zab = Rab

    lij = li + lj
    I2d = np.zeros((3, lij+1, lj+1))
    I2d[:,0,0] = (np.pi/aij)**.5 * Kab
    I2dx, I2dy, I2dz = I2d

    if lij > 0:
        I2dx[1,0] = Xpa * I2dx[0,0]
        I2dy[1,0] = Ypa * I2dy[0,0]
        I2dz[1,0] = Zpa * I2dz[0,0]

    j = 0
    for i in range(1, lij):
        I2dx[i+1,j] = Xpa * I2dx[i,j] + i/(2*aij) * I2dx[i-1,j]
        I2dy[i+1,j] = Ypa * I2dy[i,j] + i/(2*aij) * I2dy[i-1,j]
        I2dz[i+1,j] = Zpa * I2dz[i,j] + i/(2*aij) * I2dz[i-1,j]

    for j in range(1, lj+1):
        for i in range(lij+1-j):
            I2dx[i,j] = Xab * I2dx[i,j-1] + I2dx[i+1,j-1]
            I2dy[i,j] = Yab * I2dy[i,j-1] + I2dy[i+1,j-1]
            I2dz[i,j] = Zab * I2dz[i,j-1] + I2dz[i+1,j-1]

    nfi = n_cart(li)
    nfj = n_cart(lj)
    ovlp = np.zeros((nfi, nfj))
    i = 0
    for ix in range(li, -1, -1):
        for iy in range(li-ix, -1, -1):
            iz = li - ix - iy
            j = 0
            for jx in range(lj, -1, -1):
                for jy in range(lj-jx, -1, -1):
                    jz = lj - jx - jy
                    ovlp[i,j] = I2dx[ix,jx] * I2dy[iy,jy] * I2dz[iz,jz]
                    j += 1
            i += 1
    return ovlp
