import tempfile
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import pyscf
from py_qc_book.chap13.diis import DIIS
from py_qc_book.chap16.krylov import solve_krylov
from py_qc_book.chap16.rhf_grad import grad_hcore, grad_hf_energy, solve_mo1

from jax.config import config
config.update('jax_enable_x64', True)
# In case your CUDA version is too old.
config.update('jax_platform_name', 'cpu')

einsum = jnp.einsum

def _Mole_flatten(mol):
    if hasattr(mol, 'coords'):
        coords = mol.coords
    else:
        coords = mol.atom_coords()
    children = (coords,)
    # id(mol) in metadata is changed by _tree_unflatten when computing hessian.
    # This may break the PyTreeDef type comparison. The current pytree class
    # registration cannot be used for high-order derivatives.
    metadata = (mol, ['coords'])
    return children, metadata

def _tree_unflatten(metadata, children):
    obj, keys = metadata
    # Create a copy of the original object
    out = object.__new__(obj.__class__)
    out.__dict__.update(obj.__dict__)
    for k, a in zip(keys, children):
        setattr(out, k, a)
    return out

jax.tree_util.register_pytree_node(
    pyscf.gto.mole.Mole, _Mole_flatten, _tree_unflatten)

jax.tree_util.register_pytree_node(
    pyscf.scf.hf.RHF,
    lambda mf: ((mf.mol,), (mf, ['mol'])), _tree_unflatten)

def update_CCD_amplitudes(H, t2, level_shift=0):
    nvir, nocc = t2.shape[1:3]
    fock = H['fock']
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]
    e_o = foo.diagonal()
    e_v = fvv.diagonal() + level_shift

    Fvv = fvv - .5 * einsum('klcd,bdkl->bc', H['oovv'], t2)
    Foo = foo + .5 * einsum('klcd,cdjl->kj', H['oovv'], t2)
    Fvv = Fvv - jnp.diag(e_v)
    Foo = Foo - jnp.diag(e_o)

    t2out = .25 * H['vvoo']
    t2out = t2out - einsum('bkcj,acik->abij', H['vovo'], t2)
    t2out = t2out + .5 * einsum('bc,acij->abij', Fvv, t2)
    t2out = t2out - .5 * einsum('kj,abik->abij', Foo, t2)
    t2out = t2out + .5 * einsum('klcd,acik,bdjl->abij', H['oovv'], t2, t2)
    t2out = t2out - t2out.transpose(0,1,3,2)
    t2out = t2out - t2out.transpose(1,0,2,3)
    oooo = .5 * einsum('klcd,cdij->ijkl', H['oovv'], t2)
    oooo = oooo + H['oooo']
    t2out = t2out + .5 * einsum('ijkl,abkl->abij', oooo, t2)
    t2out = t2out + .5 * einsum('abcd,cdij->abij', H['vvvv'], t2)
    t2out = t2out / (e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None])
    return t2out

def get_CCD_corr_energy(H, t2):
    return .25 * einsum('ijab,abij->', H['oovv'], t2)

def mp2(H):
    nocc = H['oooo'].shape[0]
    fock = H['fock']
    e_o = fock.diagonal()[:nocc]
    e_v = fock.diagonal()[nocc:]
    t2 = H['vvoo'] / (e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None])
    e = get_CCD_corr_energy(H, t2)
    return H['e_hf'] + e, t2

def CCD_solve(mf: pyscf.scf.hf.RHF, max_cycle=20, conv_tol=1e-5):
    H = mo_integrals(mf)
    e_ccd, t2 = mp2(H) # initial guess
    print(f'E(MP2)={e_ccd}')
    e_hf = H['e_hf']
    for cycle in range(max_cycle):
        t2, t2_prev = update_CCD_amplitudes(H, t2), t2
        e_ccd, e_prev = get_CCD_corr_energy(H, t2) + e_hf, e_ccd
        print(f'{cycle=}, E(CCD)={e_ccd}, dE={e_ccd-e_prev}')
        if jnp.abs(t2 - t2_prev).max() < conv_tol:
            break
    return e_ccd

@jax.custom_jvp
def mo_integrals(mf: pyscf.scf.hf.RHF):
    '''MO integrals in physists notation <pq||rs>'''
    mol = mf.mol
    orb = mf.mo_coeff
    eri = mol.intor('int2e', aosym='s1')
    eri = einsum('pqrs,pi,qj,rk,sl->ijkl', eri, orb, orb, orb, orb)
    hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    hcore = einsum('pq,pi,qj->ij', hcore, orb, orb)
    return _mo_integrals_common(mol, hcore, eri, mf.e_tot)

@mo_integrals.defjvp
def mo_integrals_jvp(primals, tangents):
    mf, = tangents
    disp = mf.mol.coords
    mol = primals[0].mol
    mo1 = solve_mo1(mol, mf.mo_energy, mf.mo_coeff, mol.nelectron//2)
    mo1 = einsum('ixpq,ix->pq', mo1, disp)
    eri = _eri_deriv(mol, mf.mo_coeff, mo1, disp)
    hcore = _hcore_deriv(mol, mf.mo_coeff, mo1, disp)
    e_hf1 = grad_hf_energy(mol, mf.mo_energy, mf.mo_coeff, mol.nelectron//2)
    e_hf = einsum('ix,ix->', e_hf1, disp)
    return mo_integrals(*primals), _mo_integrals_common(mol, hcore, eri, e_hf)

def _mo_integrals_common(mol, hcore, eri, e_hf):
    nmo = hcore.shape[0]
    no = mol.nelectron
    i2 = jnp.eye(2)
    eri = einsum('pqrs,ab,cd->parcqbsd', eri, i2, i2).reshape([nmo*2]*4)
    eri = eri - eri.transpose(1,0,2,3)
    H = {}
    H['vvoo'] = vvoo = eri[no:,no:,:no,:no]
    H['oovv'] = vvoo.transpose(2,3,0,1)
    H['vovo'] = eri[no:,:no,no:,:no]
    H['oooo'] = eri[:no,:no,:no,:no]
    H['vvvv'] = eri[no:,no:,no:,no:]
    hcore = einsum('pq,ab->paqb', hcore, i2).reshape([nmo*2]*2)
    H['fock'] = hcore + einsum('ipiq->pq', eri[:no,:,:no,:])
    H['e_hf'] = e_hf
    return H

def _hcore_deriv(mol, mo0, mo1, disp):
    hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    hcore = einsum('pq,pi,qj->ij', hcore, mo1, mo0)
    hcore = hcore + hcore.T
    for i in range(mol.natm):
        h1 = grad_hcore(mol, i)
        hcore = hcore + einsum('x,xpq,pi,qj->ij', disp[i], h1, mo0, mo0)
    return hcore

def _eri_deriv(mol, mo0, mo1, disp):
    eri1 = mol.intor('int2e_ip1', aosym='s1')
    eri1_partial = einsum('xpqrs,qj,rk,sl->xpjkl', eri1, mo0, mo0, mo0)
    eri = mol.intor('int2e', aosym='s1')
    eri = einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo1, mo0, mo0, mo0)
    aoslices = mol.aoslice_by_atom()[:,2:4]
    for i in range(mol.natm):
        p0, p1 = aoslices[i]
        eri = eri + einsum('x,xpjkl,pi->ijkl', disp[i], -eri1_partial[:,p0:p1], mo0[p0:p1])
    # symmetrize eri
    eri = eri + eri.transpose(1,0,2,3)
    eri = eri + eri.transpose(2,3,0,1)
    return eri

if __name__ == '__main__':
    mol = pyscf.M(atom='N 0. 0 0; N 3. 0 0', unit='Bohr', basis='cc-pvdz')
    mf = mol.RHF().run()
    grad = jax.grad(CCD_solve, argnums=0)
    dmf = grad(mf)
    print(dmf.mol.coords)
