from functools import lru_cache
import numpy as np
from py_qc_book.chap12.analytical_integrals.v5.basis import gto_offsets, num_functions
from py_qc_book.chap12.analytical_integrals.v5.coulomb_1e_MD import get_R_tensor, get_E_tensor
from j8fold import pack_gto_attrs

@lru_cache(20)
def reduced_cart_iter(n):
    '''Nested loops for Cartesian components, subject to x+y+z <= n'''
    return [(x, y, z) for x in range(n+1) for y in range(n+1-x) for z in range(n+1-x-y)]

def primitive_R_tensor(li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd):
    aij = ai + aj
    Rp = (ai * Ra + aj * Rb) / aij
    akl = ak + al
    Rq = (ak * Rc + al * Rd) / akl
    Rpq = Rp - Rq
    theta = aij * akl / (aij + akl)
    lij = li + lj
    lkl = lk + ll
    l4 = lij + lkl
    Rt = get_R_tensor(l4, theta, Rpq)
    nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    nf_kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    Rt2 = np.empty((nf_ij, nf_kl))
    for kl, (e, f, g) in enumerate(reduced_cart_iter(lkl)):
        phase = (-1)**(e+f+g)
        for ij, (t, u, v) in enumerate(reduced_cart_iter(lij)):
            Rt2[ij,kl] = phase * Rt[t+e,u+f,v+g]
    Rt2 *= 2*np.pi**2.5/(aij*akl*(aij+akl)**.5)
    return Rt2

def contract_dm_R(gto_params, shell_idx, Et_dm):
    i, j, k, l = shell_idx
    i0, i1, li, Ra, exps_i, norm_ci = gto_params[i]
    j0, j1, lj, Rb, exps_j, norm_cj = gto_params[j]
    k0, k1, lk, Rc, exps_k, norm_ck = gto_params[k]
    l0, l1, ll, Rd, exps_l, norm_cl = gto_params[l]
    out = []
    for ai in exps_i:
        for aj in exps_j:
            Rt_dm = 0.
            kl = 0
            for ak in exps_k:
                for al in exps_l:
                    Rt = primitive_R_tensor(li, lj, lk, ll, ai, aj, ak, al,
                                            Ra, Rb, Rc, Rd)
                    Rt_dm += np.einsum('pq,q->p', Rt, Et_dm[kl])
                    kl += 1
            out.append(Rt_dm)
    return np.array(out)

def build_j(gtos, dm):
    nao = num_functions(gtos)
    jmat = np.zeros((nao, nao))
    gto_params = pack_gto_attrs(gtos)
    ngtos = len(gtos)

    @lru_cache(ngtos*ngtos)
    def contract_Et_dm(i, j):
        i0, i1, li, Ra, exps_i, norm_ci = gto_params[i]
        j0, j1, lj, Rb, exps_j, norm_cj = gto_params[j]
        Et_dm = []
        for ai, ci in zip(exps_i, norm_ci):
            for aj, cj in zip(exps_j, norm_cj):
                Et = get_E_tensor(li, lj, ai, aj, Ra, Rb)
                Et_dm.append(ci * cj * np.einsum('ijt,ji->t', Et, dm[j0:j1,i0:i1]))
        return Et_dm

    for i in range(ngtos):
        for j in range(ngtos):
            Rt_dm = 0.
            for k in range(ngtos):
                for l in range(ngtos):
                    Et_dm = contract_Et_dm(k, l)
                    Rt_dm += contract_dm_R(gto_params, (i,j,k,l), Et_dm)

            i0, i1, li, Ra, exps_i, norm_ci = gto_params[i]
            j0, j1, lj, Rb, exps_j, norm_cj = gto_params[j]
            ij = 0
            jblock = 0.
            for ai, ci in zip(exps_i, norm_ci):
                for aj, cj in zip(exps_j, norm_cj):
                    Et = get_E_tensor(li, lj, ai, aj, Ra, Rb)
                    R_cicj = ci * cj * Rt_dm[ij]
                    jblock += np.einsum('ijt,t->ij', Et, R_cicj)
                    ij += 1
            jmat[i0:i1,j0:j1] = jblock
    return jmat
