from functools import lru_cache
from typing import List
import numpy as np
from scipy.special import roots_hermite
from basis import CGTO, n_cart
from rys_roots import gamma_inc as _gamma_inc
from overlap_MD import get_E_tensor


@lru_cache(100)
def gamma_inc(n, x):
    return _gamma_inc(n, x)[n]

def get_R_tensor(l, a, rpq):
    @lru_cache(4000)
    def get_R(n, t, u, v):
        if t == u == v == 0:
            return (-2*a)**n * gamma_inc(n, a*np.array(rpq).dot(rpq))
        elif t < 0 or u < 0 or v < 0:
            return 0.
        elif t > 0:
            return (t-1) * get_R(n+1, t-2, u, v) + rpq[0] * get_R(n+1, t-1, u, v)
        elif u > 0:
            return (u-1) * get_R(n+1, t, u-2, v) + rpq[1] * get_R(n+1, t, u-1, v)
        elif v > 0:
            return (v-1) * get_R(n+1, t, u, v-2) + rpq[2] * get_R(n+1, t, u, v-1)
        else:
            return 0.

    Rt = np.zeros((l+1, l+1, l+1))
    for t in range(l+1):
        for u in range(l+1):
            for v in range(l+1):
                Rt[t,u,v] = get_R(0, t, u, v)
    return Rt

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

    fac = 2*np.pi/aij
    return fac * np.einsum('abtuv,tuv->ab', Et, Rt)
