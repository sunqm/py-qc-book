from functools import lru_cache
import numpy as np
import scipy.special
from basis import CGTO, n_cart, iter_cart_xyz

def ft_contracted_overlap(bas_i: CGTO, bas_j: CGTO, kv: np.ndarray) -> np.ndarray:
    li, lj = bas_i.angular_momentum, bas_j.angular_momentum
    norm_ci = bas_i.norm_coefficients
    norm_cj = bas_j.norm_coefficients
    Ra, Rb = bas_i.coordinates, bas_j.coordinates
    S = 0
    for ai, ci in zip(bas_i.exponents, norm_ci):
        for aj, cj in zip(bas_j.exponents, norm_cj):
            S += ci * cj * ft_primitive_overlap(li, lj, ai, aj, Ra, Rb, kv)
    return S

def ft_primitive_overlap(li, lj, ai, aj, Ra, Rb, kv):
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra - 1j/(2*aij) * kv
    theta_ij = ai * aj / aij

    @lru_cache(1000)
    def get_S(i: int, j: int):
        if i < 0 or j < 0:
            return 0
        if j > 0:
            return get_S(i+1, j-1) + Rab * get_S(i, j-1)
        if i > 1:
            return Rpa * get_S(i-1, j) + (i-1)/(2*aij) * get_S(i-2, j)
        if i == 1:
            return Rpa * get_S(i-1, j)
        return (np.pi/aij)**.5 * np.exp(-theta_ij*Rab**2 - kv**2/(4*aij) - 1j*kv*Rp)

    nfi = n_cart(li)
    nfj = n_cart(lj)
    nk = kv.shape[0]
    ft = np.zeros((nfi, nfj, nk), dtype=np.complex128)
    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            ft[i,j] = get_S(ix,jx)[:,0] * get_S(iy,jy)[:,1] * get_S(iz,jz)[:,2]
    return ft

def pw_3d_grids(n_r, n_theta, n_phi):
    x, w_x = scipy.special.roots_legendre(n_r)
    t, w_t = scipy.special.roots_legendre(n_phi)
    theta, w_theta = scipy.special.roots_legendre(n_theta)
    r = np.log(2/(1-x)) / np.log(2)
    w_r = w_x /np.log(2) / (1-x)
    theta = theta * np.pi + np.pi
    w_theta = w_theta * np.pi
    phi = np.arccos(t)

    kv = np.zeros((w_r.size, w_t.size, w_theta.size, 3))
    kv[:,:,:,0] = np.einsum('i,j,k->ijk', r, np.sin(phi), np.cos(theta)) # x
    kv[:,:,:,1] = np.einsum('i,j,k->ijk', r, np.sin(phi), np.sin(theta)) # y
    kv[:,:,:,2] = np.einsum('i,j->ij', r, np.cos(phi))[:,:,np.newaxis] # z

    weights = np.einsum('i,j,k->ijk', r**2 * w_r, w_t, w_theta)
    return kv.reshape(-1,3), weights.ravel()

def primitive_ERI(li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd) -> np.ndarray:
    kv, weights = pw_3d_grids(20, 20, 20)
    Fij = ft_primitive_overlap(li, lj, ai, aj, Ra, Rb, kv)
    Fkl = ft_primitive_overlap(lk, ll, al, ak, Rd, Rc, -kv)
    coul_ft = .5/np.pi**2 / np.einsum('gx,gx->g', kv, kv)
    return np.einsum('g,g,ijg,klg->ijkl', weights, coul_ft, Fij, Fkl)
