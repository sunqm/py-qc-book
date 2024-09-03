from typing import List
import numpy as np
from scipy.special import roots_hermite
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
    E_cart = np.zeros((li+1, lj+1, lij+1, 3))

    for j in range(lj+1):
        for i in range(li+1):
            for t in range(lij+1):
                if t > 0:
                    if i > 0 and j > 0:
                        E_cart[i,j,t] = (i*E_cart[i-1,j,t-1] + j*E_cart[i,j-1,t-1]) / (2*aij*t)
                    elif j > 0:
                        E_cart[i,j,t] = j*E_cart[i,j-1,t-1] / (2*aij*t)
                    elif i > 0:
                        E_cart[i,j,t] = i*E_cart[i-1,j,t-1] / (2*aij*t)
                    else:
                        E_cart[i,j,t] = 0.
                elif j > 0:
                    E_cart[i,j,0] = Rpb * E_cart[i,j-1,0] + E_cart[i,j-1,1]
                elif i > 0:
                    E_cart[i,j,0] = Rpa * E_cart[i-1,j,0] + E_cart[i-1,j,1]
                else:
                    E_cart[0,0,0] = Kab
    return E_cart.transpose(3,0,1,2)

def get_E_tensor(li, lj, ai, aj, Ra, Rb):
    Ex, Ey, Ez = get_E_cart_components(li, lj, ai, aj, Ra, Rb)
    # products subject to ix + iy + iz = li
    ix, iy, iz = np.array(iter_cart_xyz(li)).T
    jx, jy, jz = np.array(iter_cart_xyz(lj)).T
    Et = np.einsum('ijx,ijy,ijz->ijxyz', Ex[ix[:,None],jx], Ey[iy[:,None],jy], Ez[iz[:,None],jz])
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
    return Et[:,:,0,0,0] * (np.pi/aij)**1.5
