import numpy as np
from py_qc_book.chap12.analytical_integrals.v5.basis import gto_offsets, num_functions
from py_qc_book.chap12.analytical_integrals.v5.eri_OS_rys import contracted_ERI

def dispatch(gto_params, shell_idx):
    i, j, k, l = shell_idx
    i0, i1 = gto_params[i][:2]
    j0, j1 = gto_params[j][:2]
    k0, k1 = gto_params[k][:2]
    l0, l1 = gto_params[l][:2]
    return i0, i1, j0, j1, k0, k1, l0, l1

def get_eri_sub_tensor(gto_params, shell_idx):
    i, j, k, l = shell_idx
    i0, i1, li, Ra, exps_i, norm_ci = gto_params[i]
    j0, j1, lj, Rb, exps_j, norm_cj = gto_params[j]
    k0, k1, lk, Rc, exps_k, norm_ck = gto_params[k]
    l0, l1, ll, Rd, exps_l, norm_cl = gto_params[l]
    di = i1 - i0
    dj = j1 - j0
    dk = k1 - k0
    dl = l1 - l0
    out = np.zeros((di, dj, dk, dl))

    contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                   norm_ci, norm_cj, norm_ck, norm_cl, Ra, Rb, Rc, Rd,
                   out, 0, 0, 0, 0)
    return out

def contract_j_s1(jmat, gto_params, dm, shell_idx):
    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(gto_params, shell_idx)
    sub_eri = get_eri_sub_tensor(gto_params, shell_idx)
    jmat[k0:k1,l0:l1] += np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1])
    return

# i>=j, k>=l
def contract_j_s4(jmat, gto_params, dm, shell_idx):
    i, j, k, l = shell_idx
    if i < j or k < l:
        return

    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(gto_params, shell_idx)
    sub_eri = get_eri_sub_tensor(gto_params, shell_idx)
    if i > j and k > l:
        tmp = np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1] + dm[i0:i1,j0:j1].T)
        jmat[k0:k1,l0:l1] += tmp
        jmat[l0:l1,k0:k1] += tmp.T
    elif i > j: # k == l
        tmp = np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1] + dm[i0:i1,j0:j1].T)
        jmat[k0:k1,l0:l1] += tmp
    elif k > l: # i == j
        tmp = np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1])
        jmat[k0:k1,l0:l1] += tmp
        jmat[l0:l1,k0:k1] += tmp.T
    else: # i == j and k == l
        jmat[k0:k1,l0:l1] += np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1])
    return

# i>=j, k>=l, ij >= kl
def contract_j_s8(jmat, gto_params, dm, shell_idx):
    i, j, k, l = shell_idx
    if i < j or k < l or i < k:
        return
    if i == k:
        return contract_j_s4(jmat, gto_params, dm, shell_idx)

    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(gto_params, shell_idx)
    sub_eri = get_eri_sub_tensor(gto_params, shell_idx)
    if i > j and k > l:
        tmp = np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1] + dm[i0:i1,j0:j1].T)
        jmat[k0:k1,l0:l1] += tmp
        jmat[l0:l1,k0:k1] += tmp.T
        tmp = np.einsum('ijkl,lk->ij', sub_eri, dm[l0:l1,k0:k1] + dm[k0:k1,l0:l1].T)
        jmat[i0:i1,j0:j1] += tmp
        jmat[j0:j1,i0:i1] += tmp.T
    elif i > j: # k == l
        tmp = np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1] + dm[i0:i1,j0:j1].T)
        jmat[k0:k1,l0:l1] += tmp
        tmp = np.einsum('ijkl,lk->ij', sub_eri, dm[l0:l1,k0:k1])
        jmat[i0:i1,j0:j1] += tmp
        jmat[j0:j1,i0:i1] += tmp.T
    elif k > l: # i == j
        tmp = np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1])
        jmat[k0:k1,l0:l1] += tmp
        jmat[l0:l1,k0:k1] += tmp.T
        tmp = np.einsum('ijkl,lk->ij', sub_eri, dm[l0:l1,k0:k1] + dm[k0:k1,l0:l1].T)
        jmat[i0:i1,j0:j1] += tmp
    else: # i == j and k == l
        jmat[k0:k1,l0:l1] += np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1])
        jmat[i0:i1,j0:j1] += np.einsum('ijkl,lk->ij', sub_eri, dm[l0:l1,k0:k1])
    return

def pack_gto_attrs(gtos):
    '''Pack GTO attributs, making them more efficient to load'''
    offsets = gto_offsets(gtos)
    gto_params = []
    for i, bas_i in enumerate(gtos):
        i0 = offsets[i]
        i1 = offsets[i+1]
        li = bas_i.angular_momentum
        Ra = bas_i.coordinates
        exps_i = bas_i.exponents
        norm_ci = bas_i.norm_coefficients
        gto_params.append((i0, i1, li, Ra, exps_i, norm_ci))
    return gto_params

def build_j_8fold(gtos, dm):
    nao = num_functions(gtos)
    jmat = np.zeros((nao, nao))
    gto_params = pack_gto_attrs(gtos)
    ngtos = len(gtos)
    for i in range(ngtos):
        for j in range(i+1):
            for k in range(i+1):
                for l in range(k+1):
                    contract_j_s8(jmat, gto_params, dm, (i,j,k,l))
    return jmat
