from functools import lru_cache
from typing import List
import numpy as np
from basis import CGTO, n_cart, iter_cart_xyz
from rys_roots import gamma_inc as _gamma_inc


def contracted_ERI(bas_i, bas_j, bas_k, bas_l) -> np.ndarray:
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

@lru_cache(100)
def gamma_inc(n, x):
    return _gamma_inc(n, x)[n]

def primitive_ERI(li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd) -> np.ndarray:
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab.dot(Rab))

    akl = ak + al
    Rcd = Rc - Rd
    Rq = (ak * Rc + al * Rd) / akl
    theta_kl = ak * al / akl
    Kcd = np.exp(-theta_kl * Rcd.dot(Rcd))

    Rpq = Rp - Rq
    theta = aij * akl / (aij + akl)
    theta_r2 = theta * Rpq.dot(Rpq)
    Kabcd = 2*np.pi**2.5/(aij*akl*(aij+akl)**.5) * Kab * Kcd

    Xab, Yab, Zab = Rab
    Xcd, Ycd, Zcd = Rcd
    Xpq, Ypq, Zpq = Rpq
    Xpa, Ypa, Zpa = Rp - Ra

    @lru_cache()
    def vrr(n, ix, iy, iz):
        if iz > 0:
            val = Zpa*vrr(n, ix, iy, iz-1)
            val -= theta/aij*Zpq*vrr(n+1, ix, iy, iz-1)
            if iz > 1:
                val += (iz-1)*.5/aij * (vrr(n, ix, iy, iz-2) - theta/aij*vrr(n+1, ix, iy, iz-2))
            return val

        if iy > 0:
            val = Ypa*vrr(n, ix, iy-1, iz)
            val -= theta/aij*Ypq*vrr(n+1, ix, iy-1, iz)
            if iy > 1:
                val += (iy-1)*.5/aij * (vrr(n, ix, iy-2, iz) - theta/aij*vrr(n+1, ix, iy-2, iz))
            return val

        if ix > 0:
            val = Xpa*vrr(n, ix-1, iy, iz)
            val -= theta/aij*Xpq*vrr(n+1, ix-1, iy, iz)
            if ix > 1:
                val += (ix-1)*.5/aij * (vrr(n, ix-2, iy, iz) - theta/aij*vrr(n+1, ix-2, iy, iz))
            return val

        return Kabcd * gamma_inc(n, theta_r2)

    @lru_cache()
    def trr(ix, iy, iz, kx, ky, kz):
        if kz > 0:
            val = -(aj*Zab+al*Zcd)/akl * trr(ix, iy, iz, kx, ky, kz-1)
            val -= aij/akl * trr(ix, iy, iz+1, kx, ky, kz-1)
            if kz > 1:
                val += (kz-1)*.5/akl * trr(ix, iy, iz, kx, ky, kz-2)
            if iz > 0:
                val += iz*.5/akl * trr(ix, iy, iz-1, kx, ky, kz-1)
            return val

        if ky > 0:
            val = -(aj*Yab+al*Ycd)/akl * trr(ix, iy, iz, kx, ky-1, kz)
            val -= aij/akl * trr(ix, iy+1, iz, kx, ky-1, kz)
            if ky > 1:
                val += (ky-1)*.5/akl * trr(ix, iy, iz, kx, ky-2, kz)
            if iy > 0:
                val += iy*.5/akl * trr(ix, iy-1, iz, kx, ky-1, kz)
            return val

        if kx > 0:
            val = -(aj*Xab+al*Xcd)/akl * trr(ix, iy, iz, kx-1, ky, kz)
            val -= aij/akl * trr(ix+1, iy, iz, kx-1, ky, kz)
            if kx > 1:
                val += (kx-1)*.5/akl * trr(ix, iy, iz, kx-2, ky, kz)
            if ix > 0:
                val += ix*.5/akl * trr(ix-1, iy, iz, kx-1, ky, kz)
            return val

        return vrr(0, ix, iy, iz)

    @lru_cache()
    def hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly, lz):
        if lz > 0:
            return hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz+1, lx, ly, lz-1) + Zcd * hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly, lz-1)
        if ly > 0:
            return hrr(ix, iy, iz, jx, jy, jz, kx, ky+1, kz, lx, ly-1, lz) + Ycd * hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly-1, lz)
        if lx > 0:
            return hrr(ix, iy, iz, jx, jy, jz, kx+1, ky, kz, lx-1, ly, lz) + Xcd * hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx-1, ly, lz)
        if jz > 0:
            return hrr(ix, iy, iz+1, jx, jy, jz-1, kx, ky, kz, lx, ly, lz) + Zab * hrr(ix, iy, iz, jx, jy, jz-1, kx, ky, kz, lx, ly, lz)
        if jy > 0:
            return hrr(ix, iy+1, iz, jx, jy-1, jz, kx, ky, kz, lx, ly, lz) + Yab * hrr(ix, iy, iz, jx, jy-1, jz, kx, ky, kz, lx, ly, lz)
        if jx > 0:
            return hrr(ix+1, iy, iz, jx-1, jy, jz, kx, ky, kz, lx, ly, lz) + Xab * hrr(ix, iy, iz, jx-1, jy, jz, kx, ky, kz, lx, ly, lz)
        return trr(ix, iy, iz, kx, ky, kz)

    ixyz = iter_cart_xyz(li)
    jxyz = iter_cart_xyz(lj)
    kxyz = iter_cart_xyz(lk)
    lxyz = iter_cart_xyz(ll)
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)
    eri = np.empty((nfi, nfj, nfk, nfl))
    for i, (ix, iy, iz) in enumerate(ixyz):
        for j, (jx, jy, jz) in enumerate(jxyz):
            for k, (kx, ky, kz) in enumerate(kxyz):
                for l, (lx, ly, lz) in enumerate(lxyz):
                    eri[i,j,k,l] = hrr(ix,iy,iz,jx,jy,jz,kx,ky,kz,lx,ly,lz)
    return eri

def _primitive_ERI(li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd) -> np.ndarray:
    '''This version has smaller numerical stability issues. It will be used for
    the code generator in the final version of the optimization.
    '''
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab.dot(Rab))

    akl = ak + al
    Rcd = Rc - Rd
    Rq = (ak * Rc + al * Rd) / akl
    theta_kl = ak * al / akl
    Kcd = np.exp(-theta_kl * Rcd.dot(Rcd))

    Rpq = Rp - Rq
    theta = aij * akl / (aij + akl)
    theta_r2 = theta * Rpq.dot(Rpq)
    Kabcd = 2*np.pi**2.5/(aij*akl*(aij+akl)**.5) * Kab * Kcd

    Xab, Yab, Zab = Rab
    Xcd, Ycd, Zcd = Rcd
    Xpq, Ypq, Zpq = Rpq
    Xpa, Ypa, Zpa = Rp - Ra
    Xqc, Yqc, Zqc = Rq - Rc

    @lru_cache()
    def vrr(n, ix, iy, iz):
        if iz > 0:
            val = Zpa*vrr(n, ix, iy, iz-1)
            val -= theta/aij*Zpq*vrr(n+1, ix, iy, iz-1)
            if iz > 1:
                val += (iz-1)*.5/aij * (vrr(n, ix, iy, iz-2) - theta/aij*vrr(n+1, ix, iy, iz-2))
            return val

        if iy > 0:
            val = Ypa*vrr(n, ix, iy-1, iz)
            val -= theta/aij*Ypq*vrr(n+1, ix, iy-1, iz)
            if iy > 1:
                val += (iy-1)*.5/aij * (vrr(n, ix, iy-2, iz) - theta/aij*vrr(n+1, ix, iy-2, iz))
            return val

        if ix > 0:
            val = Xpa*vrr(n, ix-1, iy, iz)
            val -= theta/aij*Xpq*vrr(n+1, ix-1, iy, iz)
            if ix > 1:
                val += (ix-1)*.5/aij * (vrr(n, ix-2, iy, iz) - theta/aij*vrr(n+1, ix-2, iy, iz))
            return val

        return Kabcd * gamma_inc(n, theta_r2)

    @lru_cache()
    def trr(n, ix, iy, iz, kx, ky, kz):
        if kz > 0:
            val = Zqc * trr(n, ix, iy, iz, kx, ky, kz-1)
            val += theta/akl*Zpq * trr(n+1, ix, iy, iz, kx, ky, kz-1)
            if kz > 1:
                val += (kz-1)*.5/akl * (trr(n, ix, iy, iz, kx, ky, kz-2)
                    - theta/akl * trr(n+1, ix, iy, iz, kx, ky, kz-2))
            if iz > 0:
                val += iz*.5/(aij+akl) * trr(n+1, ix, iy, iz-1, kx, ky, kz-1)
            return val

        if ky > 0:
            val = Yqc * trr(n, ix, iy, iz, kx, ky-1, kz)
            val += theta/akl*Ypq * trr(n+1, ix, iy, iz, kx, ky-1, kz)
            if ky > 1:
                val += (ky-1)*.5/akl * (trr(n, ix, iy, iz, kx, ky-2, kz)
                    - theta/akl * trr(n+1, ix, iy, iz, kx, ky-2, kz))
            if iy > 0:
                val += iy*.5/(aij+akl) * trr(n+1, ix, iy-1, iz, kx, ky-1, kz)
            return val

        if kx > 0:
            val = Xqc * trr(n, ix, iy, iz, kx-1, ky, kz)
            val += theta/akl*Xpq * trr(n+1, ix, iy, iz, kx-1, ky, kz)
            if kx > 1:
                val += (kx-1)*.5/akl * (trr(n, ix, iy, iz, kx-2, ky, kz)
                    - theta/akl * trr(n+1, ix, iy, iz, kx-2, ky, kz))
            if ix > 0:
                val += ix*.5/(aij+akl) * trr(n+1, ix-1, iy, iz, kx-1, ky, kz)
            return val

        return vrr(n, ix, iy, iz)

    @lru_cache()
    def hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly, lz):
        if lz > 0:
            return hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz+1, lx, ly, lz-1) + Zcd * hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly, lz-1)
        if ly > 0:
            return hrr(ix, iy, iz, jx, jy, jz, kx, ky+1, kz, lx, ly-1, lz) + Ycd * hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx, ly-1, lz)
        if lx > 0:
            return hrr(ix, iy, iz, jx, jy, jz, kx+1, ky, kz, lx-1, ly, lz) + Xcd * hrr(ix, iy, iz, jx, jy, jz, kx, ky, kz, lx-1, ly, lz)
        if jz > 0:
            return hrr(ix, iy, iz+1, jx, jy, jz-1, kx, ky, kz, lx, ly, lz) + Zab * hrr(ix, iy, iz, jx, jy, jz-1, kx, ky, kz, lx, ly, lz)
        if jy > 0:
            return hrr(ix, iy+1, iz, jx, jy-1, jz, kx, ky, kz, lx, ly, lz) + Yab * hrr(ix, iy, iz, jx, jy-1, jz, kx, ky, kz, lx, ly, lz)
        if jx > 0:
            return hrr(ix+1, iy, iz, jx-1, jy, jz, kx, ky, kz, lx, ly, lz) + Xab * hrr(ix, iy, iz, jx-1, jy, jz, kx, ky, kz, lx, ly, lz)
        return trr(0, ix, iy, iz, kx, ky, kz)

    ixyz = iter_cart_xyz(li)
    jxyz = iter_cart_xyz(lj)
    kxyz = iter_cart_xyz(lk)
    lxyz = iter_cart_xyz(ll)
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)
    eri = np.empty((nfi, nfj, nfk, nfl))
    for i, (ix, iy, iz) in enumerate(ixyz):
        for j, (jx, jy, jz) in enumerate(jxyz):
            for k, (kx, ky, kz) in enumerate(kxyz):
                for l, (lx, ly, lz) in enumerate(lxyz):
                    eri[i,j,k,l] = hrr(ix,iy,iz,jx,jy,jz,kx,ky,kz,lx,ly,lz)
    return eri
