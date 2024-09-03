from typing import List
import numpy as np
import numba
from basis import CGTO, n_cart
from _tensors import get_E_tensor, get_R_tensor


def get_matrix(gtos: List[CGTO], Rc) -> np.ndarray:
    V = []
    for bas_i in gtos:
        V.append([contracted_coulomb_1e(bas_i, bas_j, Rc) for bas_j in gtos])
    return np.block(V)

def contracted_coulomb_1e(bas_i, bas_j, Rc) -> np.ndarray:
    li, lj = bas_i.angular_momentum, bas_j.angular_momentum
    norm_ci = bas_i.norm_coefficients
    norm_cj = bas_j.norm_coefficients
    Ra, Rb = bas_i.coordinates, bas_j.coordinates
    nfi = n_cart(li)
    nfj = n_cart(lj)
    V = np.zeros((nfi, nfj))

    for ai, ci in zip(bas_i.exponents, norm_ci):
        for aj, cj in zip(bas_j.exponents, norm_cj):
            V += ci * cj * primitive_coulomb_1e(li, lj, ai, aj, Ra, Rb, Rc)
    return V

def primitive_coulomb_1e(li, lj, ai, aj, Ra, Rb, Rc) -> np.ndarray:
    aij = ai + aj
    Rp = (ai * Ra + aj * Rb) / aij

    Rt = get_R_tensor(li+lj, aij, Rp-Rc)
    Et = get_E_tensor(li, lj, ai, aj, Ra, Rb)

    nfi, nfj = Et.shape[:2]

    fac = 2*np.pi/aij
    #: v = np.einsum('abtuv,tuv->ab', Et, Rt)
    v = np.dot(Et.reshape(nfi*nfj,-1), Rt.ravel())
    v *= fac
    return v.reshape(nfi, nfj)
