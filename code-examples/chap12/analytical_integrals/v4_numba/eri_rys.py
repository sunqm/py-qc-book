import numpy as np
import numba
from basis import CGTO, jitCGTO, n_cart
from rys_roots import rys_roots_weights

@numba.njit
def contracted_ERI_jit(bas_i: jitCGTO, bas_j: jitCGTO, bas_k: jitCGTO, bas_l: jitCGTO) -> np.ndarray:
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
    out = np.zeros((nfi, nfj, nfk, nfl))

    for ai, ci in zip(bas_i.exponents, norm_ci):
        for aj, cj in zip(bas_j.exponents, norm_cj):
            for ak, ck in zip(bas_k.exponents, norm_ck):
                for al, cl in zip(bas_l.exponents, norm_cl):
                    fac = ci * cj * ck * cl
                    out += fac * primitive_ERI(
                        li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
    return out
#
# without compiling the CGTO class
#
def contracted_ERI(bas_i: CGTO, bas_j: CGTO, bas_k: CGTO, bas_l: CGTO) -> np.ndarray:
    li, lj = bas_i.angular_momentum, bas_j.angular_momentum
    lk, ll = bas_k.angular_momentum, bas_l.angular_momentum
    Ra, Rb = bas_i.coordinates, bas_j.coordinates
    Rc, Rd = bas_k.coordinates, bas_l.coordinates
    norm_ci = bas_i.norm_coefficients
    norm_cj = bas_j.norm_coefficients
    norm_ck = bas_k.norm_coefficients
    norm_cl = bas_l.norm_coefficients
    exps_i, exps_j = bas_i.exponents, bas_j.exponents
    exps_k, exps_l = bas_k.exponents, bas_l.exponents
    return _contracted_coul_2e(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                               norm_ci, norm_cj, norm_ck, norm_cl, Ra, Rb, Rc, Rd)

@numba.njit(cache=True)
def _contracted_coul_2e(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                        coef_i, coef_j, coef_k, coef_l, Ra, Rb, Rc, Rd):
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)
    out = np.zeros((nfi, nfj, nfk, nfl))
    p_eri = np.zeros((nfi, nfj, nfk, nfl))

    for ai, ci in zip(exps_i, coef_i):
        for aj, cj in zip(exps_j, coef_j):
            for ak, ck in zip(exps_k, coef_k):
                for al, cl in zip(exps_l, coef_l):
                    fac = ci * cj * ck * cl
                    eri = primitive_ERI(
                        li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
                    for i in range(nfi):
                        for j in range(nfj):
                            for k in range(nfk):
                                for l in range(nfl):
                                    out[i,j,k,l] += fac * eri[i,j,k,l]
    return out

@numba.njit('double[:,:,:,::1](int64, int64, int64, int64, double, double, double, double, double[::1], double[::1], double[::1], double[::1])', cache=True)
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
    l4 = lij + lkl
    nroots = (l4 + 2) // 2
    rt, wt = rys_roots_weights(nroots, theta_r2)
    wt *= Kabcd
    if l4 == 0:
        return wt.reshape(1,1,1,1)

    theta_aij = theta / aij
    theta_akl = theta / akl
    fac_aij = .5/aij
    fac_akl = .5/akl
    fac_a1 = .5/(aij+ akl)

    Xpq, Ypq, Zpq = Rpq
    Xpa, Ypa, Zpa = Rp - Ra
    Xqc, Yqc, Zqc = Rq - Rc
    Xtheta_aij = Xpq * theta_aij
    Ytheta_aij = Ypq * theta_aij
    Ztheta_aij = Zpq * theta_aij
    Xtheta_akl = Xpq * theta_akl
    Ytheta_akl = Ypq * theta_akl
    Ztheta_akl = Zpq * theta_akl

    I2dx = np.zeros((l4+1, lkl+1, nroots))
    I2dy = np.zeros((l4+1, lkl+1, nroots))
    I2dz = np.zeros((l4+1, lkl+1, nroots))
    for n in range(nroots):
        I2dx[0,0,n] = 1.
        I2dy[0,0,n] = 1.
        I2dz[0,0,n] = wt[n]

    if l4 > 0:
        for n in range(nroots):
            I2dx[1,0,n] = (Xpa - Xtheta_aij*rt[n]) * I2dx[0,0,n]
            I2dy[1,0,n] = (Ypa - Ytheta_aij*rt[n]) * I2dy[0,0,n]
            I2dz[1,0,n] = (Zpa - Ztheta_aij*rt[n]) * I2dz[0,0,n]
        for i in range(1, l4):
            for n in range(nroots):
                I2dx[i+1,0,n] = (Xpa - Xtheta_aij*rt[n]) * I2dx[i,0,n] + i*fac_aij*(1-theta_aij*rt[n]) * I2dx[i-1,0,n]
                I2dy[i+1,0,n] = (Ypa - Ytheta_aij*rt[n]) * I2dy[i,0,n] + i*fac_aij*(1-theta_aij*rt[n]) * I2dy[i-1,0,n]
                I2dz[i+1,0,n] = (Zpa - Ztheta_aij*rt[n]) * I2dz[i,0,n] + i*fac_aij*(1-theta_aij*rt[n]) * I2dz[i-1,0,n]

    if lkl > 0:
        for n in range(nroots):
            I2dx[0,1,n] = (Xqc + Xtheta_akl*rt[n]) * I2dx[0,0,n]
            I2dy[0,1,n] = (Yqc + Ytheta_akl*rt[n]) * I2dy[0,0,n]
            I2dz[0,1,n] = (Zqc + Ztheta_akl*rt[n]) * I2dz[0,0,n]
        for i in range(1, l4):
            for n in range(nroots):
                I2dx[i,1,n] = (Xqc + Xtheta_akl*rt[n]) * I2dx[i,0,n] + i*fac_a1*rt[n] * I2dx[i-1,0,n]
                I2dy[i,1,n] = (Yqc + Ytheta_akl*rt[n]) * I2dy[i,0,n] + i*fac_a1*rt[n] * I2dy[i-1,0,n]
                I2dz[i,1,n] = (Zqc + Ztheta_akl*rt[n]) * I2dz[i,0,n] + i*fac_a1*rt[n] * I2dz[i-1,0,n]

    for k in range(1, lkl):
        for n in range(nroots):
            I2dx[0,k+1,n] = (Xqc + Xtheta_akl*rt[n]) * I2dx[0,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dx[0,k-1,n]
            I2dy[0,k+1,n] = (Yqc + Ytheta_akl*rt[n]) * I2dy[0,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dy[0,k-1,n]
            I2dz[0,k+1,n] = (Zqc + Ztheta_akl*rt[n]) * I2dz[0,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dz[0,k-1,n]
        for i in range(1, l4-k):
            for n in range(nroots):
                I2dx[i,k+1,n] = (Xqc + Xtheta_akl*rt[n]) * I2dx[i,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dx[i,k-1,n] + i*fac_a1*rt[n] * I2dx[i-1,k,n]
                I2dy[i,k+1,n] = (Yqc + Ytheta_akl*rt[n]) * I2dy[i,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dy[i,k-1,n] + i*fac_a1*rt[n] * I2dy[i-1,k,n]
                I2dz[i,k+1,n] = (Zqc + Ztheta_akl*rt[n]) * I2dz[i,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dz[i,k-1,n] + i*fac_a1*rt[n] * I2dz[i-1,k,n]

    I4dx = np.zeros((lij+1, lj+1, lkl+1, ll+1, nroots))
    I4dy = np.zeros((lij+1, lj+1, lkl+1, ll+1, nroots))
    I4dz = np.zeros((lij+1, lj+1, lkl+1, ll+1, nroots))
    for i in range(lij+1):
        for k in range(lkl+1):
            for n in range(nroots):
                I4dx[i,0,k,0,n] = I2dx[i,k,n]
                I4dy[i,0,k,0,n] = I2dy[i,k,n]
                I4dz[i,0,k,0,n] = I2dz[i,k,n]

    Xab, Yab, Zab = Rab
    Xcd, Ycd, Zcd = Rcd

    for i in range(lij+1):
        for l in range(1, ll+1):
            for k in range(lkl+1-l):
                for n in range(nroots):
                    I4dx[i,0,k,l,n] = I4dx[i,0,k+1,l-1,n] + Xcd * I4dx[i,0,k,l-1,n]
                    I4dy[i,0,k,l,n] = I4dy[i,0,k+1,l-1,n] + Ycd * I4dy[i,0,k,l-1,n]
                    I4dz[i,0,k,l,n] = I4dz[i,0,k+1,l-1,n] + Zcd * I4dz[i,0,k,l-1,n]

    for j in range(lj):
        for i in range(lij-j):
            for k in range(lk+1):
                for l in range(ll+1):
                    for n in range(nroots):
                        I4dx[i,j+1,k,l,n] = I4dx[i+1,j,k,l,n] + Xab * I4dx[i,j,k,l,n]
                        I4dy[i,j+1,k,l,n] = I4dy[i+1,j,k,l,n] + Yab * I4dy[i,j,k,l,n]
                        I4dz[i,j+1,k,l,n] = I4dz[i+1,j,k,l,n] + Zab * I4dz[i,j,k,l,n]

    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)
    eri = np.empty((nfi,nfj,nfk,nfl))
    i = 0
    for ix in range(li, -1, -1):
        for iy in range(li-ix, -1, -1):
            iz = li - ix - iy
            j = 0
            for jx in range(lj, -1, -1):
                for jy in range(lj-jx, -1, -1):
                    jz = lj - jx - jy
                    k = 0
                    for kx in range(lk, -1, -1):
                        for ky in range(lk-kx, -1, -1):
                            kz = lk - kx - ky
                            l = 0
                            for lx in range(ll, -1, -1):
                                for ly in range(ll-lx, -1, -1):
                                    lz = ll - lx - ly
                                    Ix = I4dx[ix,jx,kx,lx]
                                    Iy = I4dy[iy,jy,ky,ly]
                                    Iz = I4dz[iz,jz,kz,lz]
                                    val = 0
                                    for n in range(nroots):
                                        val += Ix[n] * Iy[n] * Iz[n]
                                    eri[i,j,k,l] = val
                                    l += 1
                            k += 1
                    j += 1
            i += 1
    return eri
