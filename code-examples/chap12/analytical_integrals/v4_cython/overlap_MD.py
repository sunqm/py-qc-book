from typing import List
import numpy as np
from scipy.special import roots_hermite
from basis import CGTO, n_cart
from _tensors import get_E_tensor

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
