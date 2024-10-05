import itertools
import numpy as np
from basis import CGTO, n_cart, gto_offsets
from coulomb_1e_MD import get_R_tensor, get_E_tensor, reduced_cart_iter
from rys_roots import gamma_inc

def contracted_ERI(bas_i: CGTO, bas_j: CGTO, bas_k: CGTO, bas_l: CGTO) -> np.ndarray:
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
    aij = ai + aj
    Rp = (ai * Ra + aj * Rb) / aij
    akl = ak + al
    Rq = (ak * Rc + al * Rd) / akl
    Rpq = Rp - Rq
    theta = aij * akl / (aij + akl)
    lij = li + lj
    lkl = lk + ll
    l4 = lij + lkl

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
    nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    nf_kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    Rt2 = np.empty((nf_kl, nf_ij))
    for kl, (e, f, g) in enumerate(reduced_cart_iter(lkl)):
        phase = (-1)**(e+f+g)
        for ij, (t, u, v) in enumerate(reduced_cart_iter(lij)):
            Rt2[kl,ij] = phase * Rt[t+e,u+f,v+g]

    Etab = get_E_tensor(li, lj, ai, aj, Ra, Rb)
    Etcd = get_E_tensor(lk, ll, ak, al, Rc, Rd)
    # Basis contraction can be applied to the gcd array before proceeding to the
    # second np.dot
    gcd = np.dot(Rt2.T, Etcd.reshape(nfk*nfl,nf_kl).T)
    eri = np.dot(Etab.reshape(nfi*nfj,nf_ij), gcd)
    eri *= 2*np.pi**2.5/(aij*akl*(aij+akl)**.5)
    return eri.reshape(nfi,nfj,nfk,nfl)


def get_tensor(f, gtos):
    offsets = gto_offsets(gtos)
    nao = offsets[-1]
    V = np.empty((nao, nao, nao, nao))
    for i, bas_i in enumerate(gtos):
        i0, i1 = offsets[i], offsets[i+1]
        for j, bas_j in enumerate(gtos):
            j0, j1 = offsets[j], offsets[j+1]
            for k, bas_k in enumerate(gtos):
                k0, k1 = offsets[k], offsets[k+1]
                for l, bas_l in enumerate(gtos):
                    l0, l1 = offsets[l], offsets[l+1]
                    V[i0:i1,j0:j1,k0:k1,l0:l1] = f(bas_i, bas_j, bas_k, bas_l)
    return V
