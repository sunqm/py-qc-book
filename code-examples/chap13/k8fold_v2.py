'''Consolidate GTO shells into blocks'''

import numpy as np
from py_qc_book.chap12.analytical_integrals.v5.basis import gto_offsets
from py_qc_book.chap13.j8fold_v2 import (
    dispatch, get_eri_sub_tensor, partition_gtos, pack_gto_attrs)

def contract_k_s1(kmat, gto_params, dm, group_idx, block_offsets):
    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(group_idx, block_offsets)
    sub_eri = get_eri_sub_tensor(gto_params, group_idx, block_offsets)
    kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
    return

# i>=j, k>=l
def contract_k_s4(kmat, gto_params, dm, group_idx, block_offsets):
    i, j, k, l = group_idx
    if i < j or k < l:
        return

    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(group_idx, block_offsets)
    sub_eri = get_eri_sub_tensor(gto_params, group_idx, block_offsets)
    if i > j and k > l:
        kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
        kmat[j0:j1,l0:l1] += np.einsum('ijkl,ik->jl', sub_eri, dm[i0:i1,k0:k1])
        kmat[i0:i1,k0:k1] += np.einsum('ijkl,jl->ik', sub_eri, dm[j0:j1,l0:l1])
        kmat[j0:j1,k0:k1] += np.einsum('ijkl,il->jk', sub_eri, dm[i0:i1,l0:l1])
    elif i > j: # k == l
        kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
        kmat[j0:j1,l0:l1] += np.einsum('ijkl,ik->jl', sub_eri, dm[i0:i1,k0:k1])
    elif k > l: # i == j
        kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
        kmat[i0:i1,k0:k1] += np.einsum('ijkl,jl->ik', sub_eri, dm[j0:j1,l0:l1])
    else: # i == j and k == l
        kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
    return

# i>=j, k>=l, ij >= kl
def contract_k_s8(kmat, gto_params, dm, group_idx, block_offsets):
    i, j, k, l = group_idx
    if i < j or k < l or i < k:
        return
    if i == k:
        return contract_k_s4(kmat, gto_params, dm, group_idx, block_offsets)

    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(group_idx, block_offsets)
    sub_eri = get_eri_sub_tensor(gto_params, group_idx, block_offsets)
    if i > j and k > l:
        kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
        kmat[j0:j1,l0:l1] += np.einsum('ijkl,ik->jl', sub_eri, dm[i0:i1,k0:k1])
        kmat[i0:i1,k0:k1] += np.einsum('ijkl,jl->ik', sub_eri, dm[j0:j1,l0:l1])
        kmat[j0:j1,k0:k1] += np.einsum('ijkl,il->jk', sub_eri, dm[i0:i1,l0:l1])
        kmat[l0:l1,i0:i1] += np.einsum('ijkl,kj->li', sub_eri, dm[k0:k1,j0:j1])
        kmat[l0:l1,j0:j1] += np.einsum('ijkl,ki->lj', sub_eri, dm[k0:k1,i0:i1])
        kmat[k0:k1,i0:i1] += np.einsum('ijkl,lj->ki', sub_eri, dm[l0:l1,j0:j1])
        kmat[k0:k1,j0:j1] += np.einsum('ijkl,li->kj', sub_eri, dm[l0:l1,i0:i1])
    elif i > j: # k == l
        kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
        kmat[j0:j1,l0:l1] += np.einsum('ijkl,ik->jl', sub_eri, dm[i0:i1,k0:k1])
        kmat[l0:l1,i0:i1] += np.einsum('ijkl,kj->li', sub_eri, dm[k0:k1,j0:j1])
        kmat[l0:l1,j0:j1] += np.einsum('ijkl,ki->lj', sub_eri, dm[k0:k1,i0:i1])
    elif k > l: # i == j
        kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
        kmat[i0:i1,k0:k1] += np.einsum('ijkl,jl->ik', sub_eri, dm[j0:j1,l0:l1])
        kmat[l0:l1,i0:i1] += np.einsum('ijkl,kj->li', sub_eri, dm[k0:k1,j0:j1])
        kmat[k0:k1,i0:i1] += np.einsum('ijkl,lj->ki', sub_eri, dm[l0:l1,j0:j1])
    else: # i == j and k == l
        kmat[i0:i1,l0:l1] += np.einsum('ijkl,jk->il', sub_eri, dm[j0:j1,k0:k1])
        kmat[l0:l1,i0:i1] += np.einsum('ijkl,kj->li', sub_eri, dm[k0:k1,j0:j1])
    return

def build_k_8fold(gtos, dm, block_size=8):
    offsets = gto_offsets(gtos)
    nao = offsets[-1]
    kmat = np.zeros((nao, nao))

    gto_params = pack_gto_attrs(gtos)
    block_offsets = (partition_gtos(gtos, block_size), offsets)
    nblks = len(block_offsets[0]) - 1

    for i in range(nblks):
        for j in range(i+1):
            for k in range(i+1):
                for l in range(k+1):
                    contract_k_s8(kmat, gto_params, dm, (i,j,k,l), block_offsets)
    return kmat
