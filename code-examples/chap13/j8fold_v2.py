'''Consolidate GTO shells into blocks'''

import numpy as np
from py_qc_book.chap12.analytical_integrals.v5.basis import n_cart, gto_offsets
from py_qc_book.chap12.analytical_integrals.v5.eri_OS_rys import contracted_ERI
from py_qc_book.chap13.j8fold import pack_gto_attrs

def dispatch(group_idx, block_offsets):
    i, j, k, l = group_idx
    block, offsets = block_offsets
    return (offsets[block[i]], offsets[block[i+1]],
            offsets[block[j]], offsets[block[j+1]],
            offsets[block[k]], offsets[block[k+1]],
            offsets[block[l]], offsets[block[l+1]])

def get_eri_sub_tensor(gto_params, group_idx, block_offsets):
    i, j, k, l = group_idx
    block, offsets = block_offsets
    ish0, ish1 = block[i], block[i+1]
    jsh0, jsh1 = block[j], block[j+1]
    ksh0, ksh1 = block[k], block[k+1]
    lsh0, lsh1 = block[l], block[l+1]
    di = offsets[ish1] - offsets[ish0]
    dj = offsets[jsh1] - offsets[jsh0]
    dk = offsets[ksh1] - offsets[ksh0]
    dl = offsets[lsh1] - offsets[lsh0]
    out = np.zeros((di, dj, dk, dl))

    ioff = offsets[ish0]
    joff = offsets[jsh0]
    koff = offsets[ksh0]
    loff = offsets[lsh0]

    for i in range(ish0, ish1):
        i0, i1, li, Ra, exps_i, norm_ci = gto_params[i]
        i0 -= ioff
        for j in range(jsh0, jsh1):
            j0, j1, lj, Rb, exps_j, norm_cj = gto_params[j]
            j0 -= joff
            for k in range(ksh0, ksh1):
                k0, k1, lk, Rc, exps_k, norm_ck = gto_params[k]
                k0 -= koff
                for l in range(lsh0, lsh1):
                    l0, l1, ll, Rd, exps_l, norm_cl = gto_params[l]
                    l0 -= loff
                    contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                                   norm_ci, norm_cj, norm_ck, norm_cl, Ra, Rb, Rc, Rd,
                                   out, i0, j0, k0, l0)
    return out

def contract_j_s1(jmat, gto_params, dm, group_idx, block_offsets):
    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(group_idx, block_offsets)
    sub_eri = get_eri_sub_tensor(gto_params, group_idx, block_offsets)
    jmat[k0:k1,l0:l1] += np.einsum('ijkl,ji->kl', sub_eri, dm[j0:j1,i0:i1])
    return

# i>=j, k>=l
def contract_j_s4(jmat, gto_params, dm, group_idx, block_offsets):
    i, j, k, l = group_idx
    if i < j or k < l:
        return

    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(group_idx, block_offsets)
    sub_eri = get_eri_sub_tensor(gto_params, group_idx, block_offsets)
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
def contract_j_s8(jmat, gto_params, dm, group_idx, block_offsets):
    i, j, k, l = group_idx
    if i < j or k < l or i < k:
        return
    if i == k:
        return contract_j_s4(jmat, gto_params, dm, group_idx, block_offsets)

    i0, i1, j0, j1, k0, k1, l0, l1 = dispatch(group_idx, block_offsets)
    sub_eri = get_eri_sub_tensor(gto_params, group_idx, block_offsets)
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

def partition_gtos(gtos, block_size):
    '''Partition GTOs into small blocks. Return the shell offsets of each block
    '''
    def get_block_dims(dims):
        if not dims:
            return []

        s = 0
        for i, d in enumerate(dims):
            s += d
            if s >= block_size:
                break
        sub_blocks = get_block_dims(dims[i+1:])
        sub_blocks.append(i+1)
        return sub_blocks

    dims = [n_cart(b.angular_momentum) for b in gtos]
    block_dims = np.array(get_block_dims(dims))[::-1]
    return np.append(0, np.cumsum(block_dims))

def build_j_8fold(gtos, dm, block_size=8):
    offsets = gto_offsets(gtos)
    nao = offsets[-1]
    jmat = np.zeros((nao, nao))

    gto_params = pack_gto_attrs(gtos)
    block_offsets = (partition_gtos(gtos, block_size), offsets)
    nblks = len(block_offsets[0]) - 1

    for i in range(nblks):
        for j in range(i+1):
            for k in range(i+1):
                for l in range(k+1):
                    contract_j_s8(jmat, gto_params, dm, (i,j,k,l), block_offsets)
    return jmat
