from functools import lru_cache
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
    if n == 0:
        return Kabcd * _gamma_inc.reshape(1,1,1,1)

    Xab, Yab, Zab = Rab
    Xcd, Ycd, Zcd = Rcd
    Xpq, Ypq, Zpq = Rpq
    Xpa, Ypa, Zpa = Rp - Ra

    vrr = np.empty((n+1, n+1,n+1,n+1))
    vrr[:,0,0,0] = Kabcd * _gamma_inc
    ix = 0
    iy = 0
    if n > 0:
        iz = 1
        for ni in range(n+1-ix-iy-iz):
            val = Zpa*vrr[ni, ix, iy, iz-1]
            val -= theta/aij*Zpq*vrr[ni+1, ix, iy, iz-1]
            vrr[ni,ix,iy,iz] = val

    for iz in range(2, n+1-ix-iy):
        for ni in range(n+1-ix-iy-iz):
            val = Zpa*vrr[ni, ix, iy, iz-1]
            val -= theta/aij*Zpq*vrr[ni+1, ix, iy, iz-1]
            val += (iz-1)*.5/aij * (vrr[ni, ix, iy, iz-2] - theta/aij*vrr[ni+1, ix, iy, iz-2])
            vrr[ni,ix,iy,iz] = val

    if n > 0:
        iy = 1
        for iz in range(n+1-ix-iy):
            for ni in range(n+1-ix-iy-iz):
                val = Ypa*vrr[ni, ix, iy-1, iz]
                val -= theta/aij*Ypq*vrr[ni+1, ix, iy-1, iz]
                vrr[ni,ix,iy,iz] = val
    for iy in range(2, n+1-ix):
        for iz in range(n+1-ix-iy):
            for ni in range(n+1-ix-iy-iz):
                val = Ypa*vrr[ni, ix, iy-1, iz]
                val -= theta/aij*Ypq*vrr[ni+1, ix, iy-1, iz]
                val += (iy-1)*.5/aij * (vrr[ni, ix, iy-2, iz] - theta/aij*vrr[ni+1, ix, iy-2, iz])
                vrr[ni,ix,iy,iz] = val

    if n > 0:
        ix = 1
        for iy in range(n+1-ix):
            for iz in range(n+1-ix-iy):
                for ni in range(n+1-ix-iy-iz):
                    val = Xpa*vrr[ni, ix-1, iy, iz]
                    val -= theta/aij*Xpq*vrr[ni+1, ix-1, iy, iz]
                    vrr[ni,ix,iy,iz] = val
    for ix in range(2, n+1):
        for iy in range(n+1-ix):
            for iz in range(n+1-ix-iy):
                for ni in range(n+1-ix-iy-iz):
                    val = Xpa*vrr[ni, ix-1, iy, iz]
                    val -= theta/aij*Xpq*vrr[ni+1, ix-1, iy, iz]
                    val += (ix-1)*.5/aij * (vrr[ni, ix-2, iy, iz] - theta/aij*vrr[ni+1, ix-2, iy, iz])
                    vrr[ni,ix,iy,iz] = val

    trr = np.empty((n+1,n+1,n+1, lkl+1,lkl+1,lkl+1))
    trr[:,:,:,0,0,0] = vrr[0]
    kx = 0
    ky = 0
    if lkl > 0:
        kz = 1
        ksum = kx + ky + kz
        for ix in range(n+1-ksum):
            for iy in range(n+1-ksum-ix):
                iz = 0
                val = -(aj*Zab+al*Zcd)/akl * trr[ix, iy, iz, kx, ky, kz-1]
                val -= aij/akl * trr[ix, iy, iz+1, kx, ky, kz-1]
                trr[ix,iy,iz,kx,ky,kz] = val
                for iz in range(1, n+1-ksum-ix-iy):
                    val = -(aj*Zab+al*Zcd)/akl * trr[ix, iy, iz, kx, ky, kz-1]
                    val -= aij/akl * trr[ix, iy, iz+1, kx, ky, kz-1]
                    val += iz*.5/akl * trr[ix, iy, iz-1, kx, ky, kz-1]
                    trr[ix,iy,iz,kx,ky,kz] = val

    for kz in range(2, lkl+1-kx-ky):
        ksum = kx + ky + kz
        for ix in range(n+1-ksum):
            for iy in range(n+1-ksum-ix):
                iz = 0
                val = -(aj*Zab+al*Zcd)/akl * trr[ix, iy, iz, kx, ky, kz-1]
                val -= aij/akl * trr[ix, iy, iz+1, kx, ky, kz-1]
                val += (kz-1)*.5/akl * trr[ix, iy, iz, kx, ky, kz-2]
                trr[ix,iy,iz,kx,ky,kz] = val
                for iz in range(1, n+1-ksum-ix-iy):
                    val = -(aj*Zab+al*Zcd)/akl * trr[ix, iy, iz, kx, ky, kz-1]
                    val -= aij/akl * trr[ix, iy, iz+1, kx, ky, kz-1]
                    val += (kz-1)*.5/akl * trr[ix, iy, iz, kx, ky, kz-2]
                    val += iz*.5/akl * trr[ix, iy, iz-1, kx, ky, kz-1]
                    trr[ix,iy,iz,kx,ky,kz] = val

    if lkl > 0:
        ky = 1
        for kz in range(lkl+1-kx-ky):
            ksum = kx + ky + kz
            for ix in range(n+1-ksum):
                iy = 0
                for iz in range(n+1-ksum-ix-iy):
                    val = -(aj*Yab+al*Ycd)/akl * trr[ix, iy, iz, kx, ky-1, kz]
                    val -= aij/akl * trr[ix, iy+1, iz, kx, ky-1, kz]
                    trr[ix,iy,iz,kx,ky,kz] = val
                for iy in range(1, n+1-ksum-ix):
                    for iz in range(n+1-ksum-ix-iy):
                        val = -(aj*Yab+al*Ycd)/akl * trr[ix, iy, iz, kx, ky-1, kz]
                        val -= aij/akl * trr[ix, iy+1, iz, kx, ky-1, kz]
                        val += iy*.5/akl * trr[ix, iy-1, iz, kx, ky-1, kz]
                        trr[ix,iy,iz,kx,ky,kz] = val

    for ky in range(2, lkl+1-kx):
        for kz in range(lkl+1-kx-ky):
            ksum = kx + ky + kz
            for ix in range(n+1-ksum):
                iy = 0
                for iz in range(n+1-ksum-ix-iy):
                    val = -(aj*Yab+al*Ycd)/akl * trr[ix, iy, iz, kx, ky-1, kz]
                    val -= aij/akl * trr[ix, iy+1, iz, kx, ky-1, kz]
                    val += (ky-1)*.5/akl * trr[ix, iy, iz, kx, ky-2, kz]
                    trr[ix,iy,iz,kx,ky,kz] = val
                for iy in range(1, n+1-ksum-ix):
                    for iz in range(n+1-ksum-ix-iy):
                        val = -(aj*Yab+al*Ycd)/akl * trr[ix, iy, iz, kx, ky-1, kz]
                        val -= aij/akl * trr[ix, iy+1, iz, kx, ky-1, kz]
                        val += (ky-1)*.5/akl * trr[ix, iy, iz, kx, ky-2, kz]
                        val += iy*.5/akl * trr[ix, iy-1, iz, kx, ky-1, kz]
                        trr[ix,iy,iz,kx,ky,kz] = val

    if lkl > 0:
        kx = 1
        for ky in range(lkl+1-kx):
            for kz in range(lkl+1-kx-ky):
                ksum = kx + ky + kz
                ix = 0
                for iy in range(n+1-ksum-ix):
                    for iz in range(n+1-ksum-ix-iy):
                        val = -(aj*Xab+al*Xcd)/akl * trr[ix, iy, iz, kx-1, ky, kz]
                        val -= aij/akl * trr[ix+1, iy, iz, kx-1, ky, kz]
                        trr[ix,iy,iz,kx,ky,kz] = val
                for ix in range(1, n+1-ksum):
                    for iy in range(n+1-ksum-ix):
                        for iz in range(n+1-ksum-ix-iy):
                            val = -(aj*Xab+al*Xcd)/akl * trr[ix, iy, iz, kx-1, ky, kz]
                            val -= aij/akl * trr[ix+1, iy, iz, kx-1, ky, kz]
                            val += ix*.5/akl * trr[ix-1, iy, iz, kx-1, ky, kz]
                            trr[ix,iy,iz,kx,ky,kz] = val

    for kx in range(2, lkl+1):
        for ky in range(lkl+1-kx):
            for kz in range(lkl+1-kx-ky):
                ksum = kx + ky + kz
                ix = 0
                for iy in range(n+1-ksum-ix):
                    for iz in range(n+1-ksum-ix-iy):
                        val = -(aj*Xab+al*Xcd)/akl * trr[ix, iy, iz, kx-1, ky, kz]
                        val -= aij/akl * trr[ix+1, iy, iz, kx-1, ky, kz]
                        val += (kx-1)*.5/akl * trr[ix, iy, iz, kx-2, ky, kz]
                        trr[ix,iy,iz,kx,ky,kz] = val
                for ix in range(1, n+1-ksum):
                    for iy in range(n+1-ksum-ix):
                        for iz in range(n+1-ksum-ix-iy):
                            val = -(aj*Xab+al*Xcd)/akl * trr[ix, iy, iz, kx-1, ky, kz]
                            val -= aij/akl * trr[ix+1, iy, iz, kx-1, ky, kz]
                            val += (kx-1)*.5/akl * trr[ix, iy, iz, kx-2, ky, kz]
                            val += ix*.5/akl * trr[ix-1, iy, iz, kx-1, ky, kz]
                            trr[ix,iy,iz,kx,ky,kz] = val

    hrr = np.zeros((lij+1,lij+1,lij+1, lkl+1,lkl+1,lkl+1, ll+1,ll+1,ll+1))
    hrr[:,:,:,:,:,:,0,0,0] = trr[:lij+1,:lij+1,:lij+1]
    lx = 0
    ly = 0
    for lz in range(1, ll+1-ly):
        lsum = lx + ly + lz
        for kx in range(lkl+1-lsum):
            for ky in range(lkl+1-lsum-kx):
                for kz in range(max(0, lk-kx-ky), lkl+1-lsum-kx-ky):
                    hrr[:,:,:, kx, ky, kz, lx, ly, lz] = hrr[:,:,:, kx, ky, kz+1, lx, ly, lz-1] + Zcd * hrr[:,:,:, kx, ky, kz, lx, ly, lz-1]
    for ly in range(1, ll+1-lx):
        for lz in range(ll+1-lx-ly):
            lsum = lx + ly + lz
            for kx in range(lkl+1-lsum):
                for ky in range(lkl+1-lsum-kx):
                    for kz in range(max(0, lk-kx-ky), lkl+1-lsum-kx-ky):
                        hrr[:,:,:, kx, ky, kz, lx, ly, lz] = hrr[:,:,:, kx, ky+1, kz, lx, ly-1, lz] + Ycd * hrr[:,:,:, kx, ky, kz, lx, ly-1, lz]
    for lx in range(1, ll+1):
        for ly in range(ll+1-lx):
            for lz in range(ll+1-lx-ly):
                lsum = lx + ly + lz
                for kx in range(lkl+1-lsum):
                    for ky in range(lkl+1-lsum-kx):
                        for kz in range(max(0, lk-kx-ky), lkl+1-lsum-kx-ky):
                            hrr[:,:,:, kx, ky, kz, lx, ly, lz] = hrr[:,:,:, kx+1, ky, kz, lx-1, ly, lz] + Xcd * hrr[:,:,:, kx, ky, kz, lx-1, ly, lz]

    kxyz = iter_cart_xyz(lk)
    lxyz = iter_cart_xyz(ll)
    nfk = len(kxyz)
    nfl = len(lxyz)
    eri = np.zeros((lij+1,lij+1,lij+1, lj+1,lj+1,lj+1, nfk, nfl))
    for k, (kx, ky, kz) in enumerate(kxyz):
        for l, (lx, ly, lz) in enumerate(lxyz):
            eri[:,:,:,0,0,0,k,l] = hrr[:,:,:,kx,ky,kz,lx,ly,lz]
    jx = 0
    jy = 0
    for jz in range(1, lj+1-jx-jy):
        jsum = jx + jy + jz
        for ix in range(lij+1-jsum):
            for iy in range(lij+1-jsum-ix):
                for iz in range(max(0, li-ix-iy), lij+1-jsum-ix-iy):
                    eri[ix, iy, iz, jx, jy, jz] = eri[ix, iy, iz+1, jx, jy, jz-1] + Zab * eri[ix, iy, iz, jx, jy, jz-1]
    for jy in range(1, lj+1-jx):
        for jz in range(lj+1-jx-jy):
            jsum = jx + jy + jz
            for ix in range(lij+1-jsum):
                for iy in range(lij+1-jsum-ix):
                    for iz in range(max(0, li-ix-iy), lij+1-jsum-ix-iy):
                        eri[ix, iy, iz, jx, jy, jz] = eri[ix, iy+1, iz, jx, jy-1, jz] + Yab * eri[ix, iy, iz, jx, jy-1, jz]
    for jx in range(1, lj+1):
        for jy in range(lj+1-jx):
            for jz in range(lj+1-jx-jy):
                jsum = jx + jy + jz
                for ix in range(lij+1-jsum):
                    for iy in range(lij+1-jsum-ix):
                        for iz in range(max(0, li-ix-iy), lij+1-jsum-ix-iy):
                            eri[ix, iy, iz, jx, jy, jz] = eri[ix+1, iy, iz, jx-1, jy, jz] + Xab * eri[ix, iy, iz, jx-1, jy, jz]

    ix, iy, iz = np.array(iter_cart_xyz(li)).T
    jx, jy, jz = np.array(iter_cart_xyz(lj)).T
    eri = eri[ix,iy,iz]
    eri = eri[:,jx,jy,jz]
    return eri
