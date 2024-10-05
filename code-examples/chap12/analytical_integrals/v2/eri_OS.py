from functools import lru_cache
from typing import List
import numpy as np
from basis import CGTO, n_cart, iter_cart_xyz
from rys_roots import gamma_inc


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

    lij = li + lj
    lkl = lk + ll
    n = lij + lkl
    _gamma_inc = gamma_inc(n, theta_r2)

    Xab, Yab, Zab = Rab
    Xcd, Ycd, Zcd = Rcd
    Xpq, Ypq, Zpq = Rpq
    Xpa, Ypa, Zpa = Rp - Ra

    vrr = np.empty((n+1, n+1,n+1,n+1))
    vrr[:,0,0,0] = Kabcd * _gamma_inc
    for ix in range(n+1):
        for iy in range(n+1):
            for iz in range(n+1):
                for ni in range(n):
                    if ix > 0:
                        val = Xpa*vrr[ni, ix-1, iy, iz]
                        val -= theta/aij*Xpq*vrr[ni+1, ix-1, iy, iz]
                        if ix > 1:
                            val += (ix-1)*.5/aij * (vrr[ni, ix-2, iy, iz] - theta/aij*vrr[ni+1, ix-2, iy, iz])
                        vrr[ni,ix,iy,iz] = val
                    elif iy > 0:
                        val = Ypa*vrr[ni, ix, iy-1, iz]
                        val -= theta/aij*Ypq*vrr[ni+1, ix, iy-1, iz]
                        if iy > 1:
                            val += (iy-1)*.5/aij * (vrr[ni, ix, iy-2, iz] - theta/aij*vrr[ni+1, ix, iy-2, iz])
                        vrr[ni,ix,iy,iz] = val
                    elif iz > 0:
                        val = Zpa*vrr[ni, ix, iy, iz-1]
                        val -= theta/aij*Zpq*vrr[ni+1, ix, iy, iz-1]
                        if iz > 1:
                            val += (iz-1)*.5/aij * (vrr[ni, ix, iy, iz-2] - theta/aij*vrr[ni+1, ix, iy, iz-2])
                        vrr[ni,ix,iy,iz] = val

    trr = np.empty((n+1,n+1,n+1, lkl+1,lkl+1,lkl+1))
    trr[:,:,:,0,0,0] = vrr[0]
    for kx in range(lkl+1):
        for ky in range(lkl+1):
            for kz in range(lkl+1):
                for ix in range(n+1-kx):
                    for iy in range(n+1-ky):
                        for iz in range(n+1-kz):
                            if kx > 0:
                                val = -(aj*Xab+al*Xcd)/akl * trr[ix, iy, iz, kx-1, ky, kz]
                                val -= aij/akl * trr[ix+1, iy, iz, kx-1, ky, kz]
                                if kx > 1:
                                    val += (kx-1)*.5/akl * trr[ix, iy, iz, kx-2, ky, kz]
                                if ix > 0:
                                    val += ix*.5/akl * trr[ix-1, iy, iz, kx-1, ky, kz]
                                trr[ix,iy,iz,kx,ky,kz] = val
                            elif ky > 0:
                                val = -(aj*Yab+al*Ycd)/akl * trr[ix, iy, iz, kx, ky-1, kz]
                                val -= aij/akl * trr[ix, iy+1, iz, kx, ky-1, kz]
                                if ky > 1:
                                    val += (ky-1)*.5/akl * trr[ix, iy, iz, kx, ky-2, kz]
                                if iy > 0:
                                    val += iy*.5/akl * trr[ix, iy-1, iz, kx, ky-1, kz]
                                trr[ix,iy,iz,kx,ky,kz] = val
                            elif kz > 0:
                                val = -(aj*Zab+al*Zcd)/akl * trr[ix, iy, iz, kx, ky, kz-1]
                                val -= aij/akl * trr[ix, iy, iz+1, kx, ky, kz-1]
                                if kz > 1:
                                    val += (kz-1)*.5/akl * trr[ix, iy, iz, kx, ky, kz-2]
                                if iz > 0:
                                    val += iz*.5/akl * trr[ix, iy, iz-1, kx, ky, kz-1]
                                trr[ix,iy,iz,kx,ky,kz] = val

    # Split hrr into two separated loops
    hrr = np.zeros((lij+1,lij+1,lij+1, lkl+1,lkl+1,lkl+1, ll+1,ll+1,ll+1))
    hrr[:,:,:,:,:,:,0,0,0] = trr[:lij+1,:lij+1,:lij+1]
    for lx in range(ll+1):
        for ly in range(ll+1):
            for lz in range(ll+1):
                for kx in range(lkl+1-lx):
                    for ky in range(lkl+1-ly):
                        for kz in range(lkl+1-lz):
                            if lx > 0:
                                hrr[:,:,:, kx, ky, kz, lx, ly, lz] = hrr[:,:,:, kx+1, ky, kz, lx-1, ly, lz] + Xcd * hrr[:,:,:, kx, ky, kz, lx-1, ly, lz]
                            elif ly > 0:
                                hrr[:,:,:, kx, ky, kz, lx, ly, lz] = hrr[:,:,:, kx, ky+1, kz, lx, ly-1, lz] + Ycd * hrr[:,:,:, kx, ky, kz, lx, ly-1, lz]
                            elif lz > 0:
                                hrr[:,:,:, kx, ky, kz, lx, ly, lz] = hrr[:,:,:, kx, ky, kz+1, lx, ly, lz-1] + Zcd * hrr[:,:,:, kx, ky, kz, lx, ly, lz-1]

    eri = np.zeros((lij+1,lij+1,lij+1, lj+1,lj+1,lj+1, lk+1,lk+1,lk+1, ll+1,ll+1,ll+1))
    eri[:,:,:,0,0,0] = hrr[:,:,:,:lk+1,:lk+1,:lk+1]
    for jx in range(lj+1):
        for jy in range(lj+1):
            for jz in range(lj+1):
                for ix in range(lij+1-jx):
                    for iy in range(lij+1-jy):
                        for iz in range(lij+1-jz):
                            if jx > 0:
                                eri[ix, iy, iz, jx, jy, jz] = eri[ix+1, iy, iz, jx-1, jy, jz] + Xab * eri[ix, iy, iz, jx-1, jy, jz]
                            elif jy > 0:
                                eri[ix, iy, iz, jx, jy, jz] = eri[ix, iy+1, iz, jx, jy-1, jz] + Yab * eri[ix, iy, iz, jx, jy-1, jz]
                            elif jz > 0:
                                eri[ix, iy, iz, jx, jy, jz] = eri[ix, iy, iz+1, jx, jy, jz-1] + Zab * eri[ix, iy, iz, jx, jy, jz-1]

    ix, iy, iz = np.array(iter_cart_xyz(li)).T
    jx, jy, jz = np.array(iter_cart_xyz(lj)).T
    kx, ky, kz = np.array(iter_cart_xyz(lk)).T
    lx, ly, lz = np.array(iter_cart_xyz(ll)).T
    eri = eri[ix,iy,iz]
    eri = eri[:,jx,jy,jz]
    eri = eri[:,:,kx,ky,kz]
    eri = eri[:,:,:,lx,ly,lz]
    return eri
