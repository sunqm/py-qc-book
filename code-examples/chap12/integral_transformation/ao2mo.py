import tempfile
import itertools
from contextlib import contextmanager
import numpy as np
import h5py
from py_qc_book.chap09.io_utils import background, iterate_with_prefetch
from py_qc_book.chap12.analytical_integrals.v5.basis import gto_offsets
from py_qc_book.chap12.analytical_integrals.v5.eri_OS_rys import (
    contracted_ERI, pack_gto_attrs)

IO_BUF_SIZE = 1e7 # 800 MB

def transform_outcore(gtos, mo):
    '''Outcore integral transfromation for two-electron integrals:
    einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo, mo, mo, mo)
    '''
    n_shells = len(gtos)
    offsets = gto_offsets(gtos)
    gto_params = pack_gto_attrs(gtos)
    nao, nmo = mo.shape
    ao_idx1, ao_idx2 = np.tril_indices(nao)
    mo_idx1, mo_idx2 = np.tril_indices(nmo)
    nao_pair = len(ao_idx1)
    nmo_pair = len(mo_idx1)

    def fill(shell_k):
        k0, lk, Rc, exps_k, norm_ck = gto_params[shell_k]
        k1 = offsets[shell_k+1]
        buf = np.empty((nao, nao, k1-k0, k1))
        for ((i0, li, Ra, exps_i, norm_ci),
             (j0, lj, Rb, exps_j, norm_cj)) in itertools.product(*(gto_params,)*2):
            if i0 < j0: continue # symmetry between ij in (ij|kl)
            for l0, ll, Rd, exps_l, norm_cl in gto_params[:shell_k+1]:
                contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                               norm_ci, norm_cj, norm_ck, norm_cl, Ra, Rb, Rc, Rd,
                               buf, i0, j0, 0, l0)
        buf[ao_idx2, ao_idx1] = buf[ao_idx1, ao_idx2]
        return buf

    @contextmanager
    def temp_h5dat():
        with tempfile.TemporaryDirectory(dir='.') as tmpdir:
            with h5py.File(f'{tmpdir}/eri.h5', 'w') as f:
                yield f.create_dataset('eri', (nmo_pair, nao_pair), dtype='f8',
                                       chunks=(320,320))

    with temp_h5dat() as dataset:
        p1 = 0
        def save_data(buf):
            nonlocal p1
            dat = np.hstack(buf)
            p0, p1 = p1, p1 + dat.shape[1]
            dataset[:,p0:p1] = dat

        with background(save_data) as write:
            buf = []
            for shell_k in range(n_shells):
                eri = fill(shell_k)
                k0, k1 = offsets[shell_k], offsets[shell_k+1]
                for k, l0 in enumerate(range(k0, k1)):
                    dat = eri[:,:,k,:l0+1]
                    dat = np.einsum('pqs,pi,qj->ijs', dat, mo, mo, optimize=True)
                    dat = dat[mo_idx1, mo_idx2] # symmetry between ij in (ij|kl)
                    buf.append(dat)
                    if sum(x.size for x in buf) > IO_BUF_SIZE:
                        write(buf)
                        buf = []
            if buf: # sync the remaining data in buf
                write(buf)

        block_size = 320
        tasks = range(0, nmo_pair, block_size)
        def loader(i0):
            i1 = min(i0 + block_size, nmo_pair)
            return dataset[i0:i1]

        out = np.empty((nmo_pair, nmo_pair))
        for i0, buf in iterate_with_prefetch(tasks, loader):
            di = buf.shape[0]
            dat = np.empty((di, nao, nao))
            dat[:, ao_idx1, ao_idx2] = buf
            dat[:, ao_idx2, ao_idx1] = buf
            dat = np.einsum('xrs,rk,sl->xkl', dat, mo, mo, optimize=True)
            out[i0:i0+di] = dat[:, mo_idx1, mo_idx2]
    return out

if __name__ == '__main__':
    from py_qc_book.chap12.analytical_integrals.v5.basis import Molecule, n_cart

    benzene = '''
    C        0.0000     1.3990     0.0000
    C       -1.2095     0.7442     0.0000
    C       -1.2095    -0.7442     0.0000
    C        0.0000    -1.3990     0.0000
    C        1.2095    -0.7442     0.0000
    C        1.2095     0.7442     0.0000
    H        0.0000     2.4939     0.0000
    H       -2.1994     1.3083     0.0000
    H       -2.1994    -1.3083     0.0000
    H        0.0000    -2.4939     0.0000
    H        2.1994    -1.3083     0.0000
    H        2.1994     1.3083     0.0000
    '''
    gtos = Molecule.from_xyz(benzene).assign_basis({'C': '6-31G*', 'H': '6-31G'})
    nao = sum(n_cart(b.angular_momentum) for b in gtos)
    mo = np.eye(nao)
    eri_mo = transform_outcore(gtos, mo)
