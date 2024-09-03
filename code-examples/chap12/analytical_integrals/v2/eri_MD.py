import itertools
from typing import List
import numpy as np
from basis import CGTO, n_cart
from coulomb_1e_MD import get_R_tensor, get_E_tensor


def contracted_ERI(bas_i, bas_j, bas_k, bas_l) -> np.ndarray:
    li, lj = bas_i.angular_momentum, bas_j.angular_momentum
    lk, ll = bas_k.angular_momentum, bas_l.angular_momentum
    Ra, Rb = bas_i.coordinates, bas_j.coordinates
    Rc, Rd = bas_k.coordinates, bas_l.coordinates
    norm_ci = bas_i.norm_coefficients
    norm_cj = bas_j.norm_coefficients
    norm_ck = bas_k.norm_coefficients
    norm_cl = bas_l.norm_coefficients
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)
    V = np.zeros((nfi, nfj, nfk, nfl))

    for ai, ci in zip(bas_i.exponents, norm_ci):
        for aj, cj in zip(bas_j.exponents, norm_cj):
            for ak, ck in zip(bas_k.exponents, norm_ck):
                for al, cl in zip(bas_l.exponents, norm_cl):
                    V += ci*cj*ck*cl * primitive_ERI(
                        li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
    return V

def primitive_ERI(li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd) -> np.ndarray:
    Etab = get_E_tensor(li, lj, ai, aj, Ra, Rb)
    Etcd = get_E_tensor(lk, ll, ak, al, Rc, Rd)

    aij = ai + aj
    Rp = (ai * Ra + aj * Rb) / aij
    akl = ak + al
    Rq = (ak * Rc + al * Rd) / akl
    Rpq = Rp - Rq
    theta = aij * akl / (aij + akl)
    Rt = get_R_tensor(li+lj+lk+ll, theta, Rpq)

    nfi = n_cart(li)
    nfj = n_cart(lj)
    lij1 = li + lj + 1
    lkl1 = lk + ll + 1
    Rt2 = np.zeros((lij1,lij1,lij1, lkl1,lkl1,lkl1))
    for t, u, v in itertools.product(range(lkl1), range(lkl1), range(lkl1)):
        Rt2[:,:,:,t,u,v] = (-1)**(t+u+v) * Rt[t:t+lij1,u:u+lij1,v:v+lij1]
    fac = 2*np.pi**2.5/(aij*akl*(aij+akl)**.5)
    return fac * np.einsum('abtuv,tuvefg,cdefg->abcd', Etab, Rt2, Etcd)
