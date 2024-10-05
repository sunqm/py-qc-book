from typing import List
import numpy as np
from basis import CGTO, n_cart
from rys_roots import gamma_inc
from overlap_MD import get_E_tensor, reduced_cart_iter


def get_R_tensor(l, a, rpq):
    rx, ry, rz = rpq
    Rt = np.zeros((l+1, l+1, l+1, l+1))
    r2 = rx*rx + ry*ry + rz*rz

    Rt[:,0,0,0] = (-2*a)**np.arange(l+1) * gamma_inc(l, a*r2)
    if l == 0:
        return Rt[0]

    # t = u = 0
    for n in range(l):
        Rt[n,0,0,1] = rz * Rt[n+1,0,0,0]
    for v in range(1, l):
        for n in range(l-v):
            Rt[n,0,0,v+1] = v * Rt[n+1,0,0,v-1] + rz * Rt[n+1,0,0,v]

    # t = 0, u = 1
    for v in range(l+1):
        for n in range(l-v):
            Rt[n,0,1,v] = ry * Rt[n+1,0,0,v]
    # u > 1
    for u in range(1, l):
        for v in range(l+1-u):
            for n in range(l-u-v):
                Rt[n,0,u+1,v] = u * Rt[n+1,0,u-1,v] + ry * Rt[n+1,0,u,v]

    # t = 1
    for u in range(l+1):
        for v in range(l+1-u):
            for n in range(l-u-v):
                Rt[n,1,u,v] = rx * Rt[n+1,0,u,v]
    # t > 1
    for t in range(1, l):
        for u in range(l+1-t):
            for v in range(l+1-t-u):
                for n in range(l-t-u-v):
                    Rt[n,t+1,u,v] = t * Rt[n+1,t-1,u,v] + rx * Rt[n+1,t,u,v]
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

    lij = li + lj
    nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    Rt2 = np.zeros(nf_ij)
    for ij, (t, u, v) in enumerate(reduced_cart_iter(lij)):
        Rt2[ij] = Rt[t,u,v]

    fac = 2*np.pi/aij
    return fac * np.einsum('abt,t->ab', Et, Rt2)
