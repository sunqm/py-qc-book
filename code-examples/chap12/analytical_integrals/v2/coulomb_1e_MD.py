from typing import List
import numpy as np
from basis import CGTO, n_cart
from rys_roots import gamma_inc
from overlap_MD import get_E_tensor


def get_R_tensor(l, a, rpq):
    Rt = np.zeros((l*3+1, l+1, l+1, l+1))
    for t in range(l+1):
        for u in range(l+1):
            for v in range(l+1):
                if t > 1:
                    Rt[:-1,t,u,v] = (t-1) * Rt[1:,t-2,u,v] + rpq[0] * Rt[1:,t-1,u,v]
                elif t == 1:
                    Rt[:-1,1,u,v] = rpq[0] * Rt[1:,0,u,v]
                elif u > 1:
                    Rt[:-1,t,u,v] = (u-1) * Rt[1:,t,u-2,v] + rpq[1] * Rt[1:,t,u-1,v]
                elif u == 1:
                    Rt[:-1,t,1,v] = rpq[1] * Rt[1:,t,0,v]
                elif v > 1:
                    Rt[:-1,t,u,v] = (v-1) * Rt[1:,t,u,v-2] + rpq[2] * Rt[1:,t,u,v-1]
                elif v == 1:
                    Rt[:-1,t,u,1] = rpq[2] * Rt[1:,t,u,0]
                else: # t == u == v == 0
                    Rt[:,0,0,0] = (-2*a)**np.arange(l*3+1) * gamma_inc(l*3, a*np.array(rpq).dot(rpq))
    return Rt[0]

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
