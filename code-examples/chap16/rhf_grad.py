import numpy as np
import jax
import jax.numpy as jnp
import pyscf
from py_qc_book.chap16.krylov import solve_krylov

from jax.config import config
config.update('jax_enable_x64', True)
# In case your CUDA version is too old.
config.update('jax_platform_name', 'cpu')

def grad_hf_energy(mol, mo_energy, mo_coeff, nocc):
    mo_o = mo_coeff[:,:nocc]
    dm = einsum('pi,qi->pq', mo_o, mo_o) * 2
    dme = einsum('pi,i,qi->pq', mo_o, mo_energy[:nocc], mo_o) * 2
    eri1 = mol.intor('int2e_ip1')
    j = einsum('xpqrs,sr->xpq', eri1, dm)
    k = einsum('xpqrs,qr->xps', eri1, dm)
    vhf = j - k * .5
    s1 = mol.intor('int1e_ipovlp')

    aoslices = mol.aoslice_by_atom()
    de = np.zeros((mol.natm, 3))
    for k in range(mol.natm):
        p0, p1 = aoslices[k,2:]
        h1 = grad_hcore(mol, k)
        de[k] += np.einsum('xij,ji->x', h1, dm)
        de[k] += np.einsum('xij,ji->x', -vhf[:,p0:p1], dm[:,p0:p1]) * 2
        de[k] -= np.einsum('xij,ji->x', -s1[:,p0:p1], dme[:,p0:p1]) * 2

    coords = mol.atom_coords()
    Z = mol.atom_charges()
    de_nuc = jax.grad(nuclear_repulsion_energy)(coords, Z)
    return de + de_nuc

def grad_hcore(mol, atom_id):
    # derivatives of T + V on bra
    hcore_partial = mol.intor('int1e_ipkin') + mol.intor('int1e_ipnuc')
    with mol.with_rinv_at_nucleus(atom_id):
        v = mol.intor('int1e_iprinv') * -mol.atom_charge(atom_id)
    p0, p1 = mol.aoslice_by_atom()[atom_id,2:4]
    v[:,p0:p1] -= hcore_partial[:,p0:p1]
    return v + v.transpose(0, 2, 1)

def nuclear_repulsion_energy(coords, z):
    rr = coords[:,None,:] - coords
    zz = z[:,None] * z
    # Note zeros are not differentiable for norm function.
    # So we exclude these elements before calling norm
    tril = np.tril_indices(len(z), -1)
    d = jnp.linalg.norm(rr[tril], axis=1)
    return (zz[tril] / d).sum()

def einsum(*args):
    return np.einsum(*args, optimize=True)

def solve_mo1(mol, mo_energy, mo_coeff, nocc):
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    mo_o = mo_coeff[:,:nocc]
    mo_v = mo_coeff[:,nocc:]
    e_o = mo_energy[:nocc]
    e_v = mo_energy[nocc:]

    eri = mol.intor('int2e')
    def matvec(x):
        dm = einsum('pi,xij,qj->xpq', mo_v, x.reshape(-1,nvir,nocc), mo_o) * 2
        dm = dm + dm.transpose(0, 2, 1)
        vhf = einsum('pi,xpq,qj->xij', mo_v, get_vhf(eri, dm), mo_o)
        vhf /= e_v[:,None] - e_o
        return vhf.ravel()

    f1, s1 = _fock_partial_deriv(mol, mo_coeff, nocc)
    s1 = einsum('pi,xpq,qj->xij', mo_coeff, s1, mo_coeff)
    f1 = einsum('pi,xpq,qj->xij', mo_v, f1, mo_o)
    b = (f1 - s1[:,nocc:,:nocc] * e_o) / (e_o - e_v[:,None])
    mo1_vo = solve_krylov(matvec, b.ravel()).reshape(-1,nvir,nocc)

    mo1 = -.5 * s1
    mo1[:,nocc:,:nocc] = mo1_vo
    # mo1.T + s1 + mo1 = 0
    mo1[:,:nocc,nocc:] = -s1[:,:nocc,nocc:] - mo1_vo.transpose(0, 2, 1)
    mo1 = einsum('pq,xqi->xpi', mo_coeff, mo1)
    return mo1.reshape(mol.natm, 3, nao, nmo)

def get_vhf(eri, dm):
    j = einsum('pqrs,xqp->xrs', eri, dm)
    k = einsum('pqrs,xqr->xps', eri, dm)
    return j - k * .5

def _fock_partial_deriv(mol, mo_coeff, nocc):
    nao, nmo = mo_coeff.shape
    mo_o = mo_coeff[:,:nocc]

    eri1 = mol.intor('int2e_ip1')
    eri = mol.intor('int2e')
    dm = einsum('pi,qi->pq', mo_o, mo_o) * 2
    j = einsum('xpqrs,sr->xpq', eri1, dm)
    k = einsum('xpqrs,qr->xps', eri1, dm)
    vhf1 = j - k * .5
    s1 = mol.intor('int1e_ipovlp')

    aoslices = mol.aoslice_by_atom()
    s1_buf = []
    f1_buf = []
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1ao = np.zeros((3, nao, nao))
        s1ao[:,p0:p1] -= s1[:,p0:p1]
        s1ao[:,:,p0:p1] -= s1[:,p0:p1].transpose(0,2,1)
        s1_buf.append(s1ao)

        j = einsum('xpqrs,qp->xrs', eri1[:,p0:p1], dm[:,p0:p1])
        k = einsum('xpqrs,sp->xrq', eri1[:,p0:p1], dm[:,p0:p1])
        v1 = -(j - k * .5)
        v1[:,p0:p1] -= vhf1[:,p0:p1]
        v1 = v1 + v1.transpose(0,2,1)
        s1_oo = einsum('pi,xpq,qj->xij', mo_o, s1ao, mo_o)
        s1_oo = einsum('pi,xij,qj->xpq', mo_o, s1_oo, mo_o) * 2
        fock1 = grad_hcore(mol, ia) + v1 - get_vhf(eri, s1_oo)
        f1_buf.append(fock1)
    return np.vstack(f1_buf), np.vstack(s1_buf)

if __name__ == '__main__':
    mol = pyscf.M(atom='N 0. 0 0; N 1.5 0 0', basis='ccpvdz')
    mf = mol.RHF().run()
    nocc = mol.nelectron // 2
    de = grad_hf_energy(mol, mf.mo_energy, mf.mo_coeff, nocc)

    mo1 = solve_mo1(mol, mf.mo_energy, mf.mo_coeff, nocc)
    nao, nmo = mf.mo_coeff.shape
    mo1 = mo1.reshape(mol.natm, 3, nao, nmo)
    dm = einsum('pi,qi->pq', mo1[0,0], mf.mo_coeff) * 2
    dm = dm + dm.T

    def finite_diff(mol, delta=1e-2):
        mol = mol.copy()
        coords = mol.atom_coords()
        coords[0,0] += .5 * delta
        mol.set_geom_(coords, unit='Bohr')
        mf = mol.RHF(verbose=0).run()
        ehf0 = mf.e_tot
        dm0 = einsum('pi,qi->pq', mf.mo_coeff, mf.mo_coeff) * 2

        coords[0,0] -= delta
        mol.set_geom_(coords, unit='Bohr')
        mf = mol.RHF(verbose=0).run()
        ehf1 = mf.e_tot
        dm1 = einsum('pi,qi->pq', mf.mo_coeff, mf.mo_coeff) * 2
        return (ehf0 - ehf1)/delta, (dm0 - dm1)/delta

    de_fd, dm_fd = finite_diff(mol, 1e-2)
    print(abs(de_fd - de[0,0]).max(), abs(dm_fd - dm).max())

    de_fd, dm_fd = finite_diff(mol, .5e-2)
    print(abs(de_fd - de[0,0]).max(), abs(dm_fd - dm).max())
