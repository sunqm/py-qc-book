import os
import itertools
import ctypes
import numpy as np
import numba
from py_qc_book.chap12.analytical_integrals.v5.basis import (
    CGTO, jitCGTO, n_cart, gto_offsets)
from py_qc_book.chap12.analytical_integrals.v5.rys_roots import (
    rys_roots_weights, gamma_inc_for_c)

liberi_OS = ctypes.CDLL(os.path.abspath(f'{__file__}/../liberi_OS.so'))
run_eri_unrolled = liberi_OS.run_eri_unrolled
run_eri_unrolled.argtypes = [
    ctypes.c_void_p, # eri
    ctypes.c_int,    # li
    ctypes.c_int,    # lj
    ctypes.c_int,    # lk
    ctypes.c_int,    # ll
    ctypes.c_double, # ai
    ctypes.c_double, # aj
    ctypes.c_double, # ak
    ctypes.c_double, # al
    ctypes.c_void_p, # Ra
    ctypes.c_void_p, # Rb
    ctypes.c_void_p, # Rc
    ctypes.c_void_p, # Rd
    ctypes.c_void_p, # gamma_inc_fn
]
run_eri_unrolled.restypes = ctypes.c_int
gamma_inc_fn = gamma_inc_for_c.address

@numba.njit
def contracted_eri_jit(bas_i: jitCGTO, bas_j: jitCGTO, bas_k: jitCGTO, bas_l: jitCGTO,
                       out, i0, j0, k0, l0) -> np.ndarray:
    li = bas_i.angular_momentum
    lj = bas_j.angular_momentum
    lk = bas_k.angular_momentum
    ll = bas_l.angular_momentum
    Ra = bas_i.coordinates
    Rb = bas_j.coordinates
    Rc = bas_k.coordinates
    Rd = bas_l.coordinates
    norm_ci = bas_i.norm_coefficients
    norm_cj = bas_j.norm_coefficients
    norm_ck = bas_k.norm_coefficients
    norm_cl = bas_l.norm_coefficients
    exps_i = bas_i.exponents
    exps_j = bas_j.exponents
    exps_k = bas_k.exponents
    exps_l = bas_l.exponents
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)

    if li + lj + lk + ll <= 5:
        buf = np.zeros((nfi, nfj, nfk, nfl))
        _buf = buf.ctypes.data
        _Ra = Ra.ctypes.data
        _Rb = Rb.ctypes.data
        _Rc = Rc.ctypes.data
        _Rd = Rd.ctypes.data
        for ai, ci in zip(exps_i, norm_ci):
            for aj, cj in zip(exps_j, norm_cj):
                for ak, ck in zip(exps_k, norm_ck):
                    for al, cl in zip(exps_l, norm_cl):
                        fac = ci * cj * ck * cl
                        err = run_eri_unrolled(
                            _buf, li, lj, lk, ll, ai, aj, ak, al,
                            _Ra, _Rb, _Rc, _Rd, gamma_inc_fn)
                        for i in range(nfi):
                            for j in range(nfj):
                                for k in range(nfk):
                                    for l in range(nfl):
                                        out[i0+i,j0+j,k0+k,l0+l] += fac * buf[i,j,k,l]
    else:
        for ai, ci in zip(exps_i, norm_ci):
            for aj, cj in zip(exps_j, norm_cj):
                for ak, ck in zip(exps_k, norm_ck):
                    for al, cl in zip(exps_l, norm_cl):
                        fac = ci * cj * ck * cl
                        buf = primitive_ERI(
                            li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
                        for i in range(nfi):
                            for j in range(nfj):
                                for k in range(nfk):
                                    for l in range(nfl):
                                        out[i0+i,j0+j,k0+k,l0+l] += fac * buf[i,j,k,l]

@numba.njit
def get_eri_tensor_jit(gtos):
    cum = 0
    offsets = [cum]
    for b in gtos:
        cum += n_cart(b.angular_momentum)
        offsets.append(cum)
    nao = offsets[-1]
    out = np.zeros((nao, nao, nao, nao))
    for i, bas_i in enumerate(gtos):
        i0 = offsets[i]
        for j, bas_j in enumerate(gtos):
            j0 = offsets[j]
            for k, bas_k in enumerate(gtos):
                k0 = offsets[k]
                for l, bas_l in enumerate(gtos):
                    l0 = offsets[l]
                    contracted_eri_jit(bas_i, bas_j, bas_k, bas_l,
                                              out, i0, j0, k0, l0)
    return out

@numba.njit
def contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                   coef_i, coef_j, coef_k, coef_l, Ra, Rb, Rc, Rd,
                   out, i0, j0, k0, l0):
    nfi = n_cart(li)
    nfj = n_cart(lj)
    nfk = n_cart(lk)
    nfl = n_cart(ll)
    if li + lj + lk + ll <= 5:
        buf = np.zeros((nfi, nfj, nfk, nfl))
        _buf = buf.ctypes.data
        _Ra = Ra.ctypes.data
        _Rb = Rb.ctypes.data
        _Rc = Rc.ctypes.data
        _Rd = Rd.ctypes.data
        for ai, ci in zip(exps_i, coef_i):
            for aj, cj in zip(exps_j, coef_j):
                for ak, ck in zip(exps_k, coef_k):
                    for al, cl in zip(exps_l, coef_l):
                        fac = ci * cj * ck * cl
                        err = run_eri_unrolled(
                            _buf, li, lj, lk, ll, ai, aj, ak, al,
                            _Ra, _Rb, _Rc, _Rd, gamma_inc_fn)
                        for i in range(nfi):
                            for j in range(nfj):
                                for k in range(nfk):
                                    for l in range(nfl):
                                        out[i0+i,j0+j,k0+k,l0+l] += fac * buf[i,j,k,l]
    else:
        for ai, ci in zip(exps_i, coef_i):
            for aj, cj in zip(exps_j, coef_j):
                for ak, ck in zip(exps_k, coef_k):
                    for al, cl in zip(exps_l, coef_l):
                        fac = ci * cj * ck * cl
                        buf = primitive_ERI(
                            li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
                        for i in range(nfi):
                            for j in range(nfj):
                                for k in range(nfk):
                                    for l in range(nfl):
                                        out[i0+i,j0+j,k0+k,l0+l] += fac * buf[i,j,k,l]

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
    l_4 = lij + lkl
    nroots = (l_4 + 2) // 2
    rt, wt = rys_roots_weights(nroots, theta_r2)
    wt *= Kabcd
    if lij == lkl == 0:
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

    I4dx = np.zeros((lij+1, lj+1, lkl+1, ll+1, nroots))
    I4dy = np.zeros((lij+1, lj+1, lkl+1, ll+1, nroots))
    I4dz = np.zeros((lij+1, lj+1, lkl+1, ll+1, nroots))
    I2dx = I4dx[:,0,:,0]
    I2dy = I4dy[:,0,:,0]
    I2dz = I4dz[:,0,:,0]
    for n in range(nroots):
        I2dx[0,0,n] = 1.
        I2dy[0,0,n] = 1.
        I2dz[0,0,n] = wt[n]

    if l_4 > 0:
        for n in range(nroots):
            I2dx[1,0,n] = (Xpa - Xtheta_aij*rt[n]) * I2dx[0,0,n]
            I2dy[1,0,n] = (Ypa - Ytheta_aij*rt[n]) * I2dy[0,0,n]
            I2dz[1,0,n] = (Zpa - Ztheta_aij*rt[n]) * I2dz[0,0,n]
        for i in range(1, lij):
            for n in range(nroots):
                I2dx[i+1,0,n] = (Xpa - Xtheta_aij*rt[n]) * I2dx[i,0,n] + i*fac_aij*(1-theta_aij*rt[n]) * I2dx[i-1,0,n]
                I2dy[i+1,0,n] = (Ypa - Ytheta_aij*rt[n]) * I2dy[i,0,n] + i*fac_aij*(1-theta_aij*rt[n]) * I2dy[i-1,0,n]
                I2dz[i+1,0,n] = (Zpa - Ztheta_aij*rt[n]) * I2dz[i,0,n] + i*fac_aij*(1-theta_aij*rt[n]) * I2dz[i-1,0,n]

    if lkl > 0:
        for n in range(nroots):
            I2dx[0,1,n] = (Xqc + Xtheta_akl*rt[n]) * I2dx[0,0,n]
            I2dy[0,1,n] = (Yqc + Ytheta_akl*rt[n]) * I2dy[0,0,n]
            I2dz[0,1,n] = (Zqc + Ztheta_akl*rt[n]) * I2dz[0,0,n]
        for i in range(1, lij+1):
            for n in range(nroots):
                I2dx[i,1,n] = (Xqc + Xtheta_akl*rt[n]) * I2dx[i,0,n] + i*fac_a1*rt[n] * I2dx[i-1,0,n]
                I2dy[i,1,n] = (Yqc + Ytheta_akl*rt[n]) * I2dy[i,0,n] + i*fac_a1*rt[n] * I2dy[i-1,0,n]
                I2dz[i,1,n] = (Zqc + Ztheta_akl*rt[n]) * I2dz[i,0,n] + i*fac_a1*rt[n] * I2dz[i-1,0,n]

    for k in range(1, lkl):
        for n in range(nroots):
            I2dx[0,k+1,n] = (Xqc + Xtheta_akl*rt[n]) * I2dx[0,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dx[0,k-1,n]
            I2dy[0,k+1,n] = (Yqc + Ytheta_akl*rt[n]) * I2dy[0,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dy[0,k-1,n]
            I2dz[0,k+1,n] = (Zqc + Ztheta_akl*rt[n]) * I2dz[0,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dz[0,k-1,n]
        for i in range(1, lij+1):
            for n in range(nroots):
                I2dx[i,k+1,n] = (Xqc + Xtheta_akl*rt[n]) * I2dx[i,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dx[i,k-1,n] + i*fac_a1*rt[n] * I2dx[i-1,k,n]
                I2dy[i,k+1,n] = (Yqc + Ytheta_akl*rt[n]) * I2dy[i,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dy[i,k-1,n] + i*fac_a1*rt[n] * I2dy[i-1,k,n]
                I2dz[i,k+1,n] = (Zqc + Ztheta_akl*rt[n]) * I2dz[i,k,n] + k*fac_akl*(1-theta_akl*rt[n]) * I2dz[i,k-1,n] + i*fac_a1*rt[n] * I2dz[i-1,k,n]

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

def pack_gto_attrs(gtos):
    '''Pack GTO attributs, making them more efficient to load'''
    offsets = gto_offsets(gtos)
    gto_params = []
    for i, bas_i in enumerate(gtos):
        i0 = offsets[i]
        li = bas_i.angular_momentum
        Ra = bas_i.coordinates
        exps_i = bas_i.exponents
        norm_ci = bas_i.norm_coefficients
        gto_params.append((i0, li, Ra, exps_i, norm_ci))
    return gto_params

def get_eri_tensor(gtos):
    offsets = gto_offsets(gtos)
    gto_params = pack_gto_attrs(gtos)
    nao = offsets[-1]
    out = np.zeros((nao, nao, nao, nao))
    for ((i0, li, Ra, exps_i, norm_ci),
         (j0, lj, Rb, exps_j, norm_cj),
         (k0, lk, Rc, exps_k, norm_ck),
         (l0, ll, Rd, exps_l, norm_cl)) in itertools.product(*(gto_params,)*4):
        contracted_ERI(li, lj, lk, ll, exps_i, exps_j, exps_k, exps_l,
                       norm_ci, norm_cj, norm_ck, norm_cl, Ra, Rb, Rc, Rd,
                       out, i0, j0, k0, l0)
    return out
