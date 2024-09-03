from typing import List
import numpy as np
from basis import CGTO, n_cart, iter_cart_xyz

def get_E_cart_components(li, lj, ai, aj, Ra, Rb):
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    Rpb = Rp - Rb
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab**2)

    lij = li + lj
    Et = np.empty((li+1, lj+1, lij+1, 3))
    Et[0,0,0] = Kab
    Et[0,0,1:] = 0.

    for i in range(1, li+1):
        Et[i,0,0] = Rpa * Et[i-1,0,0] + Et[i-1,0,1]
        for t in range(1, lij+1):
            Et[i,0,t] = i*Et[i-1,0,t-1] / (2*aij*t)
    for j in range(1, lj+1):
        Et[0,j,0] = Rpb * Et[0,j-1,0] + Et[0,j-1,1]
        for t in range(1, lij+1):
            Et[0,j,t] = j*Et[0,j-1,t-1] / (2*aij*t)
        for i in range(1, li+1):
            Et[i,j,0] = Rpb * Et[i,j-1,0] + Et[i,j-1,1]
            for t in range(1, lij+1):
                Et[i,j,t] = (i*Et[i-1,j,t-1] + j*Et[i,j-1,t-1]) / (2*aij*t)
    return Et.transpose(3,0,1,2)

def get_E_tensor(li, lj, ai, aj, Ra, Rb):
    Ex, Ey, Ez = get_E_cart_components(li, lj, ai, aj, Ra, Rb)

    lij = li + lj
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    Et = np.empty((nfi, nfj, nf_ij))
    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            # products subject to t+u+v <= li+lj
            for n, (t, u, v) in enumerate(reduced_cart_iter(lij)):
                Et[i,j,n] = Ex[ix,jx,t] * Ey[iy,jy,u] * Ez[iz,jz,v]
    return Et

def reduced_cart_iter(n):
    '''Nested loops for Cartesian components, subject to x+y+z <= n'''
    for x in range(n+1):
        for y in range(n+1-x):
            for z in range(n+1-x-y):
                yield x, y, z

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
