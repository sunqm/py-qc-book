import itertools
from tempfile import NamedTemporaryFile
import numpy as np
import scipy.linalg
from py_qc_book.chap09.io_utils import iterate_with_prefetch
from py_qc_book.chap12.analytical_integrals.v5.basis import (
    gto_offsets, num_functions)
from py_qc_book.chap12.analytical_integrals.v5.eri_OS_rys import contracted_ERI
from py_qc_book.chap13.j8fold_v2 import partition_gtos, pack_gto_attrs

class CDERI:
    segment_max_size = 500000000 # 4 GB
    forward = True

    def __init__(self, gtos, aux_gtos):
        self.gtos = gtos
        self.aux_gtos = aux_gtos
        self.nao = num_functions(gtos)
        self.naux = num_functions(aux_gtos)
        self.blocksize = 200

        mask = np.tril(schwarz_halfcond(gtos) > 1e-15)
        self.idx = np.where(mask)

        c = np.linalg.cholesky(eval_int2c2e(aux_gtos))
        self.datafile = []
        block_size = CDERI.segment_max_size // self.nao
        for gto_pair in itertools.pairwise(partition_gtos(gtos, block_size)):
            int3c2e = eval_compress_int3c2e(gtos, aux_gtos, gto_pair, mask)
            # Using NamedTemporaryFile for automatic deletion
            chunk = np.memmap(NamedTemporaryFile(), mode='w+', shape=int3c2e.shape, dtype=float)
            chunk[:] = scipy.linalg.solve_triangular(c, int3c2e, lower=True)
            self.datafile.append(chunk)

    def __iter__(self):
        blocks = list(itertools.pairwise(
             np.append(range(0, self.naux, self.blocksize), self.blocksize)))
        if not CDERI.forward:
            blocks = reversed(blocks)
        # Flip the iteration direction to maximize FS cache utilization
        CDERI.forward = not CDERI.forward

        # buf0 and buf1 are reused to reduce memory allocation overhead
        buf0 = np.zeros((self.blocksize, self.nao, self.nao))
        buf1 = np.zeros((self.blocksize, self.nao, self.nao))
        def unpack(block):
            start, end = block
            dat = [d[start:end] for d in self.datafile]
            return decompress(dat, self.idx, buf0)

        for _, data in iterate_with_prefetch(blocks, unpack):
            buf0, buf1 = buf1, buf0
            yield data

def pairwise(start, stop, step):
    return itertools.pairwise(np.append(range(start, stop, step), stop))

def eval_int2c2e(aux_gtos):
    naux = num_functions(aux_gtos)
    aux_gto_params = pack_gto_attrs(aux_gtos)

    # the fictitious s-type GTO
    lj = ll = 0
    exps_j = exps_l = np.zeros(1)
    norm_cj = norm_cl = np.ones(1)

    v = np.zeros((naux, 1, naux, 1))
    for params_i, params_k in itertools.product(
            aux_gto_params, aux_gto_params):
        i0, i1, li, Ra, exps_i, norm_ci = params_i
        k0, k1, lk, Rc, exps_k, norm_ck = params_k
        contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                       norm_ci, norm_cj, norm_ck, norm_cl, Ra, Ra, Rc, Rc,
                       v, i0, 0, k0, 0)
    return v[:,0,:,0]

def eval_compress_int3c2e(gtos, aux_gtos, gto_pair, mask):
    offsets = gto_offsets(gtos)
    gto0, gto1 = gto_pair
    j3c = []
    for shell0, shell1 in pairwise(gto0, gto1, 10):
        i0, i1 = offsets[shell0], offsets[shell1]
        dat = eval_int3c2e_block(aux_gtos, gtos[shell0:shell1], gtos)
        j3c.append(dat[:,mask[i0:i1]])
    return np.hstack(j3c)

def eval_int3c2e_block(aux_gtos, gtos_k, gtos_l):
    naux = num_functions(aux_gtos)
    nao_k = num_functions(gtos_k)
    nao_l = num_functions(gtos_l)
    gto_k_params = pack_gto_attrs(gtos_k)
    gto_l_params = pack_gto_attrs(gtos_l)
    aux_gto_params = pack_gto_attrs(aux_gtos)

    # the fictitious s-type GTO
    lj = 0
    exps_j = np.zeros(1)
    norm_cj = np.ones(1)

    v = np.zeros((naux, 1, nao_k, nao_l))
    for params_i, params_k, params_l in itertools.product(
            aux_gto_params, gto_k_params, gto_l_params):
        i0, i1, li, Ra, exps_i, norm_ci = params_i
        k0, k1, lk, Rc, exps_k, norm_ck = params_k
        l0, l1, ll, Rd, exps_l, norm_cl = params_l
        contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                       norm_ci, norm_cj, norm_ck, norm_cl, Ra, Ra, Rc, Rd,
                       v, i0, 0, k0, l0)
    return v[:,0]

def schwarz_halfcond(gtos):
    '''sqrt((ij|ij)) for all GTOs'''
    gto_params = pack_gto_attrs(gtos)
    nao = num_functions(gtos)
    v = np.zeros((nao, nao))
    for params_i, params_j in itertools.product(gto_params, gto_params):
        i0, i1, li, Ra, exps_i, norm_ci = params_i
        j0, j1, lj, Rb, exps_j, norm_cj = params_j
        out = np.zeros((i1-i0, j1-j0, i1-i0, j1-j0))
        contracted_ERI(li, lj, li, lj, exps_i, exps_j, exps_i, exps_j,
                       norm_ci, norm_cj, norm_ci, norm_cj, Ra, Rb, Ra, Rb,
                       out, 0, 0, 0, 0)
        v[i0:i1,j0:j1] = np.einsum('ijij->ij', out) ** .5
    return v

def decompress(cderi_blocks, idx, out):
    naux = cderi_blocks[0].shape[0]
    p0 = p1 = 0
    for d in cderi_blocks:
        p0, p1 = p1, p1 + d.shape[1]
        idx0 = idx[0][p0:p1]
        idx1 = idx[1][p0:p1]
        out[:naux,idx0,idx1] = out[:naux,idx1,idx0] = d
    return out[:naux]

def build_jk(gtos, aux_gtos, dm, mo_occupied):
    cderi = CDERI(gtos, aux_gtos)
    nao = num_functions(gtos)
    jmat, kmat = np.zeros((nao,nao)), np.zeros((nao,nao))
    for sub_cderi in iter(cderi):
        tmp = np.einsum('jt,pij->tpi', mo_occupied, sub_cderi, optimize=True)
        kmat += np.einsum('tpi,tpj->ij', tmp, tmp, optimize=True) * 2
        jmat += np.einsum('pij,ji,pkl->kl', sub_cderi, dm, sub_cderi, optimize=True)
    return jmat, kmat

if __name__ == "__main__":
    from py_qc_book.chap12.analytical_integrals.v5.basis import Molecule
    from py_qc_book.chap13.simple_scf import RHF
    xyz = '''
    H 0 0 0
    H 1 0 0'''
    mol = Molecule.from_xyz(xyz)
    gtos = mol.assign_basis({'H': '6-31G'})
    aux_gtos = mol.assign_basis({'H': '6-311++G**'})

    model = RHF(mol, basis_set={'H': '6-31G'})
    wfn = model.get_initial_guess()
    dm = wfn.density_matrices
    mo_occupied = wfn.orbitals[:,:1]
    j, k = build_jk(gtos, aux_gtos, dm, mo_occupied)

    ref = model.get_jk(wfn)
    print(abs(j - ref[0]).max())
    print(abs(k - ref[1]).max())
