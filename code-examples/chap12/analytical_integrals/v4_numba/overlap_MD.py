from typing import List
import numpy as np
import numba
from basis import CGTO, n_cart


@numba.njit('double[:,:,:,::1](int64, int64, double, double, double[::1], double[::1])', cache=True)
def get_E_cart_components(li, lj, ai, aj, Ra, Rb):
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    Rpb = Rp - Rb
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab**2)

    lij = li + lj
    Et = np.empty((3, li+1, lj+1, lij+1))
    Ex, Ey, Ez = Et
    Xpa, Ypa, Zpa = Rpa
    Xpb, Ypb, Zpb = Rpb

    Ex[0,0,0] = Kab[0]
    Ey[0,0,0] = Kab[1]
    Ez[0,0,0] = Kab[2]
    for t in range(1, lij+1):
        Ex[0,0,t] = 0.
        Ey[0,0,t] = 0.
        Ez[0,0,t] = 0.

    for i in range(1, li+1):
        Ex[i,0,0] = Xpa * Ex[i-1,0,0] + Ex[i-1,0,1]
        Ey[i,0,0] = Ypa * Ey[i-1,0,0] + Ey[i-1,0,1]
        Ez[i,0,0] = Zpa * Ez[i-1,0,0] + Ez[i-1,0,1]
        for t in range(1, lij+1):
            Ex[i,0,t] = i*Ex[i-1,0,t-1] / (2*aij*t)
            Ey[i,0,t] = i*Ey[i-1,0,t-1] / (2*aij*t)
            Ez[i,0,t] = i*Ez[i-1,0,t-1] / (2*aij*t)

    for j in range(1, lj+1):
        Ex[0,j,0] = Xpb * Ex[0,j-1,0] + Ex[0,j-1,1]
        Ey[0,j,0] = Ypb * Ey[0,j-1,0] + Ey[0,j-1,1]
        Ez[0,j,0] = Zpb * Ez[0,j-1,0] + Ez[0,j-1,1]
        for t in range(1, lij+1):
            Ex[0,j,t] = j*Ex[0,j-1,t-1] / (2*aij*t)
            Ey[0,j,t] = j*Ey[0,j-1,t-1] / (2*aij*t)
            Ez[0,j,t] = j*Ez[0,j-1,t-1] / (2*aij*t)
        for i in range(1, li+1):
            Ex[i,j,0] = Xpb * Ex[i,j-1,0] + Ex[i,j-1,1]
            Ey[i,j,0] = Ypb * Ey[i,j-1,0] + Ey[i,j-1,1]
            Ez[i,j,0] = Zpb * Ez[i,j-1,0] + Ez[i,j-1,1]
            for t in range(1, lij+1):
                Ex[i,j,t] = (i*Ex[i-1,j,t-1] + j*Ex[i,j-1,t-1]) / (2*aij*t)
                Ey[i,j,t] = (i*Ey[i-1,j,t-1] + j*Ey[i,j-1,t-1]) / (2*aij*t)
                Ez[i,j,t] = (i*Ez[i-1,j,t-1] + j*Ez[i,j-1,t-1]) / (2*aij*t)
    return Et

@numba.njit('double[:,:,::1](int64, int64, double, double, double[::1], double[::1])', cache=True)
def get_E_tensor(li, lj, ai, aj, Ra, Rb):
    Ex, Ey, Ez = get_E_cart_components(li, lj, ai, aj, Ra, Rb)

    lij = li + lj
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    Et = np.empty((nfi, nfj, nf_ij))

    i = 0
    for ix in range(li, -1, -1):
        for iy in range(li-ix, -1, -1):
            iz = li - ix - iy
            j = 0
            for jx in range(lj, -1, -1):
                for jy in range(lj-jx, -1, -1):
                    jz = lj - jx - jy
                    # products subject to t+u+v <= li+lj
                    n = 0
                    for t in range(lij+1):
                        for u in range(lij+1-t):
                            for v in range(lij+1-t-u):
                                Et[i,j,n] = Ex[ix,jx,t] * Ey[iy,jy,u] * Ez[iz,jz,v]
                                n += 1
                    j += 1
            i += 1
    return Et

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
    Et = get_E_tensor(li, lj, ai, aj, Ra, Rb)
    aij = ai + aj
    return Et[:,:,0] * (np.pi/aij)**1.5
