import numpy as np
import numba
from basis import jitCGTO, CGTO, n_cart
from coulomb_1e_MD import get_R_tensor, get_E_tensor
from rys_roots import gamma_inc


@numba.njit
def contracted_ERI_jit(bas_i: jitCGTO, bas_j: jitCGTO, bas_k: jitCGTO, bas_l: jitCGTO) -> np.ndarray:
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
                    fac = ci * cj * ck * cl
                    V += fac * primitive_ERI(
                        li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
    return V
#
# without compiling the CGTO class
#
def contracted_ERI(bas_i: CGTO, bas_j: CGTO, bas_k: CGTO, bas_l: CGTO) -> np.ndarray:
    li, lj = bas_i.angular_momentum, bas_j.angular_momentum
    lk, ll = bas_k.angular_momentum, bas_l.angular_momentum
    Ra, Rb = bas_i.coordinates, bas_j.coordinates
    Rc, Rd = bas_k.coordinates, bas_l.coordinates
    norm_ci = bas_i.norm_coefficients
    norm_cj = bas_j.norm_coefficients
    norm_ck = bas_k.norm_coefficients
    norm_cl = bas_l.norm_coefficients
    exps_i, exps_j = bas_i.exponents, bas_j.exponents
    exps_k, exps_l = bas_k.exponents, bas_l.exponents
    return _contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                           norm_ci, norm_cj, norm_ck, norm_cl, Ra, Rb, Rc, Rd)

@numba.njit(cache=True)
def _contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                    coef_i, coef_j, coef_k, coef_l, Ra, Rb, Rc, Rd):
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)
    V = np.zeros((nfi, nfj, nfk, nfl))

    for ai, ci in zip(exps_i, coef_i):
        for aj, cj in zip(exps_j, coef_j):
            for ak, ck in zip(exps_k, coef_k):
                for al, cl in zip(exps_l, coef_l):
                    fac = ci * cj * ck * cl
                    eri = primitive_ERI(
                        li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
                    for i in range(nfi):
                        for j in range(nfj):
                            for k in range(nfk):
                                for l in range(nfl):
                                    V[i,j,k,l] += fac * eri[i,j,k,l]
    return V

@numba.njit('double[:,:,:,::1](int64, int64, int64, int64, double, double, double, double, double[::1], double[::1], double[::1], double[::1])', cache=True)
def primitive_ERI(li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd) -> np.ndarray:
    aij = ai + aj
    Rp = (ai * Ra + aj * Rb) / aij
    akl = ak + al
    Rq = (ak * Rc + al * Rd) / akl
    Rpq = Rp - Rq
    theta = aij * akl / (aij + akl)
    l4 = li + lj + lk + ll
    if l4 == 0:
        Rab = Ra - Rb
        Rcd = Rc - Rd
        theta_ij = ai * aj / aij
        theta_kl = ak * al / akl
        theta_r2 = theta * Rpq.dot(Rpq)
        Kabcd = 2*np.pi**2.5/(aij*akl*(aij+akl)**.5)
        Kabcd *= np.exp(-theta_ij*Rab.dot(Rab) - theta_kl*Rcd.dot(Rcd))
        _gamma_inc = gamma_inc(0, theta_r2)
        return Kabcd * _gamma_inc.reshape(1,1,1,1)

    Rt = get_R_tensor(l4, theta, Rpq)
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)
    lij = li + lj
    lkl = lk + ll
    nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    nf_kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    Rt2 = np.empty((nf_kl, nf_ij))
    kl = 0
    for e in range(lkl+1):
        for f in range(lkl+1-e):
            for g in range(lkl+1-e-f):
                phase = (-1)**(e+f+g)
                ij = 0
                for t in range(lij+1):
                    for u in range(lij+1-t):
                        for v in range(lij+1-t-u):
                            Rt2[kl,ij] = phase * Rt[t+e,u+f,v+g]
                            ij += 1
                kl += 1

    Etab = get_E_tensor(li, lj, ai, aj, Ra, Rb)
    Etcd = get_E_tensor(lk, ll, ak, al, Rc, Rd)
    gcd = np.dot(Rt2.T, Etcd.reshape(nfk*nfl,nf_kl).T)
    eri = np.dot(Etab.reshape(nfi*nfj,nf_ij), gcd)
    eri *= 2*np.pi**2.5/(aij*akl*(aij+akl)**.5)
    return eri.reshape(nfi,nfj,nfk,nfl)
