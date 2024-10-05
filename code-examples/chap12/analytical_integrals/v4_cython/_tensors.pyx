#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
#cython: cdivision=True
#cython: language_level=3

import numpy as np
import scipy.linalg.lapack as lapack_lite
from libc.math cimport sqrt, exp, erf, M_PI

cdef inline int n_cart(int l):
    return (l + 1) * (l + 2) // 2

cdef double[::1] downward(int m, double t):
    cdef double half = .5
    cdef double b = m + half
    cdef double e = half * exp(-t)
    cdef double x = e
    cdef double f = e
    cdef double bi = b
    cdef double prec = 1e-15
    while x > prec * e:
        bi += 1
        x *= t / bi
        f += x
    f /= b
    cdef double[::1] out = np.empty(m+1)
    out[m] = f
    cdef int i
    for i in range(m):
        b -= 1
        f = (e + t * f) / b
        out[m-i-1] = f
    return out

cdef double[::1] upward(int m, double t):
    cdef double half = .5
    cdef double tt = sqrt(t)
    cdef double f = sqrt(M_PI)/2 / tt * erf(tt)
    cdef double e = exp(-t)
    cdef double b = half / t
    cdef double[::1] out = np.empty(m+1)
    out[0] = f
    cdef int i
    for i in range(m):
        f = b * ((2*i+1) * f - e)
        out[i+1] = f
    return out

cdef double[::1] gamma_inc(int m, double t):
    assert m >= 0
    assert t >= 0
    if (t < m + 1.5):
        return downward(m, t)
    else:
        return upward(m, t)

cdef inline double poly_value1(double *a, int order, double x):
    cdef int i
    cdef double p = a[order]
    for i in range(1, order+1):
        p = p * x + a[order-i]
    return p

cdef double[:,::1] schmidt_orth(double[::1] moments, int nroots):
    cdef int n1 = nroots + 1
    cdef double[:,::1] cs = np.zeros((n1, n1))
    cdef int j, k, m
    cdef double fac, dot
    for j in range(n1):
        fac = moments[j+j]
        for k in range(j):
            dot = 0.
            for m in range(k+1):
                dot += cs[k,m] * moments[j+m]
            for m in range(k+1):
                cs[j,m] -= dot * cs[k,m]
            fac -= dot * dot

        if fac <= 0:
            raise RuntimeError(f'schmidt_orth fail. nroots={nroots} fac={fac}')
        fac = fac**-.5
        cs[j,j] = fac
        for k in range(j):
            cs[j,k] *= fac
    return cs

cdef find_roots(double[:,::1] cs, int nroots):
    #return np.roots(cs[nroots,::-1])
    cdef double[::1] roots = np.empty(nroots)
    cdef double dum
    if nroots == 1:
        roots[0] = -cs[1,0] / cs[1,1]
        return roots
    elif nroots == 2:
        dum = sqrt(cs[2,1] * cs[2,1] - 4. * cs[2,0] * cs[2,2])
        roots[0] = .5 * (-cs[2,1] - dum) / cs[2,2]
        roots[1] = .5 * (-cs[2,1] + dum) / cs[2,2]
        return roots

    cdef double[:,::1] A = np.zeros((nroots, nroots))
    cdef double norm = -1./cs[nroots,nroots]
    cdef int m
    for m in range(nroots-1):
        A[m+1,m] = 1.
    for m in range(nroots):
        A[0,m] = cs[nroots,nroots-1-m] * norm
    roots = lapack_lite.dgeev(A, 0, 0)[0]
    return roots

cdef rys_roots_weights(int nroots, double x):
    cdef double[::1] moments = gamma_inc(nroots*2, x)
    if moments[0] < 1e-16:
        return np.zeros(nroots), np.zeros(nroots)

    # Find polynomials roots
    cdef double[:,::1] cs = schmidt_orth(moments, nroots)
    roots = find_roots(cs, nroots)
    cdef double[::1] _roots = roots

    # rtp.T.dot(diag(weights)).dot(rtp) = identity
    weights = np.zeros(nroots)
    cdef double[::1] _weights = weights
    cdef double root, dum, poly
    for i in range(nroots):
        root = _roots[i]
        dum = 1 / moments[0]
        for j in range(1, nroots):
            poly = poly_value1(&cs[j,0], j, root)
            dum += poly * poly
        _weights[i] = 1 / dum
    return roots, weights

cdef inline double square(double *r):
    return r[0]*r[0] + r[1]*r[1] + r[2]*r[2]

def get_E_tensor(int li, int lj, double ai, double aj, double[:] Ra, double[:] Rb):
    cdef double aij = ai + aj
    cdef double theta_ij = ai * aj / aij
    cdef double[3] Rpa, Rpb, Kab
    cdef double Rab, Rp
    cdef int m
    for m in range(3):
        Rab = Ra[m] - Rb[m]
        Rp = (ai * Ra[m] + aj * Rb[m]) / aij
        Rpa[m] = Rp - Ra[m]
        Rpb[m] = Rp - Rb[m]
        Kab[m] = exp(-theta_ij * Rab**2)

    cdef int lij = li + lj
    E_cart = np.empty((3, li+1, lj+1, lij+1))
    cdef double[:,:,::1] Ex = E_cart[0]
    cdef double[:,:,::1] Ey = E_cart[1]
    cdef double[:,:,::1] Ez = E_cart[2]
    Ex[0,0,0] = Kab[0]
    Ey[0,0,0] = Kab[1]
    Ez[0,0,0] = Kab[2]

    Xpa, Ypa, Zpa = Rpa
    Xpb, Ypb, Zpb = Rpb

    cdef int i, j, t
    i = 0
    j = 0
    for t in range(1, lij+1):
        Ex[0,0,t] = 0.
        Ey[0,0,t] = 0.
        Ez[0,0,t] = 0.

    for i in range(1, li+1):
        Ex[i,0,0] = Xpa * Ex[i-1,0,0] + Ex[i-1,0,1]
        Ey[i,0,0] = Ypa * Ey[i-1,0,0] + Ey[i-1,0,1]
        Ez[i,0,0] = Zpa * Ez[i-1,0,0] + Ez[i-1,0,1]
        for t in range(1, lij+1):
            Ex[i,0,t] = i*Ex[i-1,0,t-1] / (2*aij*t)
            Ey[i,0,t] = i*Ey[i-1,0,t-1] / (2*aij*t)
            Ez[i,0,t] = i*Ez[i-1,0,t-1] / (2*aij*t)

    for j in range(1, lj+1):
        Ex[0,j,0] = Xpb * Ex[0,j-1,0] + Ex[0,j-1,1]
        Ey[0,j,0] = Ypb * Ey[0,j-1,0] + Ey[0,j-1,1]
        Ez[0,j,0] = Zpb * Ez[0,j-1,0] + Ez[0,j-1,1]
        for t in range(1, lij+1):
            Ex[0,j,t] = j*Ex[0,j-1,t-1] / (2*aij*t)
            Ey[0,j,t] = j*Ey[0,j-1,t-1] / (2*aij*t)
            Ez[0,j,t] = j*Ez[0,j-1,t-1] / (2*aij*t)
        for i in range(1, li+1):
            Ex[i,j,0] = Xpb * Ex[i,j-1,0] + Ex[i,j-1,1]
            Ey[i,j,0] = Ypb * Ey[i,j-1,0] + Ey[i,j-1,1]
            Ez[i,j,0] = Zpb * Ez[i,j-1,0] + Ez[i,j-1,1]
            for t in range(1, lij+1):
                Ex[i,j,t] = (i*Ex[i-1,j,t-1] + j*Ex[i,j-1,t-1]) / (2*aij*t)
                Ey[i,j,t] = (i*Ey[i-1,j,t-1] + j*Ey[i,j-1,t-1]) / (2*aij*t)
                Ez[i,j,t] = (i*Ez[i-1,j,t-1] + j*Ez[i,j-1,t-1]) / (2*aij*t)

    cdef int nfi = n_cart(li)
    cdef int nfj = n_cart(lj)
    cdef int nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    Et = np.empty((nfi, nfj, nf_ij))
    cdef double[:,:,::1] _Et = Et

    cdef int u, v, n
    cdef int ix, iy, iz
    cdef int jx, jy, jz
    cdef double val_tu
    i = 0
    for ix in range(li, -1, -1):
        for iy in range(li-ix, -1, -1):
            iz = li - ix - iy
            j = 0
            for jx in range(lj, -1, -1):
                for jy in range(lj-jx, -1, -1):
                    jz = lj - jx - jy
                    n = 0
                    for t in range(lij+1):
                        for u in range(lij+1-t):
                            for v in range(lij+1-t-u):
                                _Et[i,j,n] = Ex[ix,jx,t] * Ey[iy,jy,u] * Ez[iz,jz,v]
                                n += 1
                    j += 1
            i += 1
    return Et

def get_R_tensor(int l, double a, double[:] rpq):
    cdef double rx = rpq[0]
    cdef double ry = rpq[1]
    cdef double rz = rpq[2]
    cdef double r2 = rx * rx + ry * ry + rz * rz
    Rt = np.zeros((l+1, l+1, l+1, l+1))
    cdef double [:,:,:,::1] _Rt = Rt
    cdef int t, u, v, m
    cdef double[::1] _gamma_inc = gamma_inc(l, a*r2)

    for m in range(l+1):
        _Rt[m,0,0,0] = (-2*a)**m * _gamma_inc[m]

    if l == 0:
        return Rt[0]

    # t = u = 0
    for m in range(l):
        _Rt[m,0,0,1] = rz * _Rt[m+1,0,0,0]
    for v in range(1, l):
        for m in range(l-v):
            _Rt[m,0,0,v+1] = v * _Rt[m+1,0,0,v-1] + rz * _Rt[m+1,0,0,v]

    # t = 0, u = 1
    for v in range(l+1):
        for m in range(l-v):
            _Rt[m,0,1,v] = ry * _Rt[m+1,0,0,v]
    # u > 1
    for u in range(1, l):
        for v in range(l+1):
            for m in range(l-u-v):
                _Rt[m,0,u+1,v] = u * _Rt[m+1,0,u-1,v] + ry * _Rt[m+1,0,u,v]

    # t = 1
    for u in range(l+1):
        for v in range(l+1-u):
            for m in range(l-u-v):
                _Rt[m,1,u,v] = rx * _Rt[m+1,0,u,v]
    # t > 1
    for t in range(1, l):
        for u in range(l+1-t):
            for v in range(l+1-t-u):
                for m in range(l-t-u-v):
                    _Rt[m,t+1,u,v] = t * _Rt[m+1,t-1,u,v] + rx * _Rt[m+1,t,u,v]
    return Rt[0]


def primitive_ERI_MD(int li, int lj, int lk, int ll,
                    double ai, double aj, double ak, double al,
                    double[::1] Ra, double[::1] Rb, double[::1] Rc, double[::1] Rd):
    cdef double aij = ai + aj
    cdef double akl = ak + al
    cdef double theta = aij * akl / (aij + akl)
    #cdef double Rpq[3]
    cdef double Rp, Rq
    cdef int m
    Rpq = np.empty(3)
    cdef double[::1] _Rpq = Rpq
    for m in range(3):
        Rp = (ai * Ra[m] + aj * Rb[m]) / aij
        Rq = (ak * Rc[m] + al * Rd[m]) / akl
        _Rpq[m] = Rp - Rq
    cdef int lij = li + lj
    cdef int lkl = lk + ll
    cdef int l4 = lij + lkl

    cdef double[3] Rab, Rcd
    cdef double theta_ij, theta_kl, theta_r2, Kabcd
    if l4 == 0:
        for m in range(3):
            Rab[m] = Ra[m] - Rb[m]
            Rcd[m] = Rc[m] - Rd[m]
        theta_ij = ai * aj / aij
        theta_kl = ak * al / akl
        theta_r2 = theta * square(&_Rpq[0])
        Kabcd = 2*M_PI**2.5/(aij*akl*(aij+akl)**.5)
        Kabcd *= exp(-theta_ij * square(Rab) - theta_kl * square(Rcd))
        eri = np.empty((1,1,1,1))
        eri[0,0,0,0] = Kabcd * gamma_inc(l4, theta_r2)[0]
        return eri

    cdef double[:,:,::1] Rt = get_R_tensor(l4, theta, Rpq)
    cdef int nfi = n_cart(li)
    cdef int nfj = n_cart(lj)
    cdef int nfk = n_cart(lk)
    cdef int nfl = n_cart(ll)
    cdef int nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    cdef int nf_kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    cdef int e, f, g, t, u, v, ij, kl
    cdef double phase

    cdef double fac = 2*M_PI**2.5/(aij*akl*(aij+akl)**.5)
    cdef double[:,::1] Rt2 = np.empty((nf_kl, nf_ij))
    kl = 0
    for e in range(lkl+1):
        for f in range(lkl+1-e):
            for g in range(lkl+1-e-f):
                phase = (-1)**(e+f+g) * fac
                ij = 0
                for t in range(lij+1):
                    for u in range(lij+1-t):
                        for v in range(lij+1-t-u):
                            Rt2[kl,ij] = phase * Rt[t+e,u+f,v+g]
                            ij += 1
                kl += 1

    Etab = get_E_tensor(li, lj, ai, aj, Ra, Rb)
    Etcd = get_E_tensor(lk, ll, ak, al, Rc, Rd)
    gcd = np.dot(Rt2.T, Etcd.reshape(nfk*nfl,nf_kl).T)
    eri = np.dot(Etab.reshape(nfi*nfj,nf_ij), gcd)
    return eri.reshape(nfi,nfj,nfk,nfl)

def primitive_ERI_OS(int li, int lj, int lk, int ll,
                     double ai, double aj, double ak, double al,
                     double[::1] Ra, double[::1] Rb, double[::1] Rc, double[::1] Rd):
    cdef double aij = ai + aj
    cdef double akl = ak + al
    cdef double theta_ij = ai * aj / aij
    cdef double theta_kl = ak * al / akl
    cdef double[3] Rab, Rcd, Rpq, Rpa
    cdef int m
    cdef double Rp, Rq
    for m in range(3):
        Rab[m] = Ra[m] - Rb[m]
        Rcd[m] = Rc[m] - Rd[m]
        Rp = (ai * Ra[m] + aj * Rb[m]) / aij
        Rq = (ak * Rc[m] + al * Rd[m]) / akl
        Rpq[m] = Rp - Rq
        Rpa[m] = Rp - Ra[m]
    cdef double theta = aij * akl / (aij + akl)
    cdef double theta_r2 = theta * square(Rpq)
    cdef double Kabcd = 2*M_PI**2.5/(aij*akl*(aij+akl)**.5)
    Kabcd *= exp(-theta_ij * square(Rab) - theta_kl * square(Rcd))

    cdef int lij = li + lj
    cdef int lkl = lk + ll
    cdef int n = lij + lkl
    cdef double[::1] _gamma_inc = gamma_inc(n, theta_r2)

    if n == 0:
        out = np.empty((1,1,1,1))
        out[0,0,0,0] = Kabcd * _gamma_inc[0]
        return out

    cdef double Xab = Rab[0]
    cdef double Yab = Rab[1]
    cdef double Zab = Rab[2]
    cdef double Xcd = Rcd[0]
    cdef double Ycd = Rcd[1]
    cdef double Zcd = Rcd[2]
    cdef double Xpq = Rpq[0]
    cdef double Ypq = Rpq[1]
    cdef double Zpq = Rpq[2]
    cdef double Xpa = Rpa[0]
    cdef double Ypa = Rpa[1]
    cdef double Zpa = Rpa[2]

    cdef int i, j, k, l, ij, kl
    cdef int ix, iy, iz
    cdef int jx, jy, jz
    cdef int kx, ky, kz
    cdef int lx, ly, lz
    cdef int ni, jsum, ksum, lsum
    cdef double val
    cdef double[:,:,:,::1] vrr = np.empty((n+1, n+1,n+1,n+1))
    for ni in range(n+1):
        vrr[ni,0,0,0] = Kabcd * _gamma_inc[ni]

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
            val += (iz-1)/(2*aij) * (vrr[ni, ix, iy, iz-2] - theta/aij*vrr[ni+1, ix, iy, iz-2])
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
                val += (iy-1)/(2*aij) * (vrr[ni, ix, iy-2, iz] - theta/aij*vrr[ni+1, ix, iy-2, iz])
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
                    val += (ix-1)/(2*aij) * (vrr[ni, ix-2, iy, iz] - theta/aij*vrr[ni+1, ix-2, iy, iz])
                    vrr[ni,ix,iy,iz] = val

    cdef double[:,:,:,:,:,::1] trr = np.empty((n+1,n+1,n+1, lkl+1,lkl+1,lkl+1))
    for ix in range(n+1):
        for iy in range(n+1-ix):
            for iz in range(n+1-ix-iy):
                trr[ix,iy,iz,0,0,0] = vrr[0,ix,iy,iz]
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

    cdef int nfi = n_cart(li)
    cdef int nfj = n_cart(lj)
    cdef int nfk = n_cart(lk)
    cdef int nfl = n_cart(ll)

    cdef double[:,:,:,:,:,::1] hrr = np.empty((lkl+1,lkl+1,lkl+1, ll+1,ll+1,ll+1))
    cdef double[:,:,:,:,:,:,::1] eri = np.empty((lij+1,lij+1,lij+1, lj+1,lj+1,lj+1, nfk*nfl))
    for ix in range(lij+1):
        for iy in range(lij+1-ix):
            for iz in range(lij+1-ix-iy):
                for kx in range(lkl+1):
                    for ky in range(lkl+1-kx):
                        for kz in range(lkl+1-kx-ky):
                            hrr[kx,ky,kz,0,0,0] = trr[ix,iy,iz,kx,ky,kz]
                lx = 0
                ly = 0
                for lz in range(1, ll+1):
                    lsum = lx + ly + lz
                    for kx in range(lkl+1-lsum):
                        for ky in range(lkl+1-lsum-kx):
                            for kz in range(lkl+1-lsum-kx-ky):
                                hrr[kx, ky, kz, lx, ly, lz] = hrr[kx, ky, kz+1, lx, ly, lz-1] + Zcd * hrr[kx, ky, kz, lx, ly, lz-1]
                for ly in range(1, ll+1-lx):
                    for lz in range(ll+1-lx-ly):
                        lsum = lx + ly + lz
                        for kx in range(lkl+1-lsum):
                            for ky in range(lkl+1-lsum-kx):
                                for kz in range(lkl+1-lsum-kx-ky):
                                    hrr[kx, ky, kz, lx, ly, lz] = hrr[kx, ky+1, kz, lx, ly-1, lz] + Ycd * hrr[kx, ky, kz, lx, ly-1, lz]
                for lx in range(1, ll+1):
                    for ly in range(ll+1-lx):
                        for lz in range(ll+1-lx-ly):
                            lsum = lx + ly + lz
                            for kx in range(lkl+1-lsum):
                                for ky in range(lkl+1-lsum-kx):
                                    for kz in range(lkl+1-lsum-kx-ky):
                                        hrr[kx, ky, kz, lx, ly, lz] = hrr[kx+1, ky, kz, lx-1, ly, lz] + Xcd * hrr[kx, ky, kz, lx-1, ly, lz]

                kl = 0
                for kx in range(lk, -1, -1):
                    for ky in range(lk-kx, -1, -1):
                        kz = lk - kx - ky
                        for lx in range(ll, -1, -1):
                            for ly in range(ll-lx, -1, -1):
                                lz = ll - lx - ly
                                eri[ix,iy,iz,0,0,0,kl] = hrr[kx,ky,kz,lx,ly,lz]
                                kl += 1

    jx = 0
    jy = 0
    for jz in range(1, lj+1-jx-jy):
        jsum = jx + jy + jz
        for ix in range(lij+1-jsum):
            for iy in range(lij+1-jsum-ix):
                for iz in range(lij+1-jsum-ix-iy):
                    for kl in range(nfk*nfl):
                        eri[ix, iy, iz, jx, jy, jz, kl] = eri[ix, iy, iz+1, jx, jy, jz-1, kl] + Zab * eri[ix, iy, iz, jx, jy, jz-1, kl]

    for jy in range(1, lj+1-jx):
        for jz in range(lj+1-jx-jy):
            jsum = jx + jy + jz
            for ix in range(lij+1-jsum):
                for iy in range(lij+1-jsum-ix):
                    for iz in range(lij+1-jsum-ix-iy):
                        for kl in range(nfk*nfl):
                            eri[ix, iy, iz, jx, jy, jz, kl] = eri[ix, iy+1, iz, jx, jy-1, jz, kl] + Yab * eri[ix, iy, iz, jx, jy-1, jz, kl]

    for jx in range(1, lj+1):
        for jy in range(lj+1-jx):
            for jz in range(lj+1-jx-jy):
                jsum = jx + jy + jz
                for ix in range(lij+1-jsum):
                    for iy in range(lij+1-jsum-ix):
                        for iz in range(lij+1-jsum-ix-iy):
                            for kl in range(nfk*nfl):
                                eri[ix, iy, iz, jx, jy, jz, kl] = eri[ix+1, iy, iz, jx-1, jy, jz, kl] + Xab * eri[ix, iy, iz, jx-1, jy, jz, kl]

    out = np.empty((nfi,nfj,nfk*nfl))
    cdef double[:,:,::1] _out = out
    i = 0
    for ix in range(li, -1, -1):
        for iy in range(li-ix, -1, -1):
            iz = li - ix - iy
            j = 0
            for jx in range(lj, -1, -1):
                for jy in range(lj-jx, -1, -1):
                    jz = lj - jx - jy
                    for k in range(nfk*nfl):
                        _out[i,j,k] = eri[ix,iy,iz,jx,jy,jz,k]
                    j += 1
            i += 1
    return out.reshape(nfi, nfj, nfk, nfl)

def primitive_ERI_rys(int li, int lj, int lk, int ll,
                      double ai, double aj, double ak, double al,
                      double[::1] Ra, double[::1] Rb, double[::1] Rc, double[::1] Rd):
    cdef double aij = ai + aj
    cdef double akl = ak + al
    cdef double theta_ij = ai * aj / aij
    cdef double theta_kl = ak * al / akl
    cdef double[3] Rab, Rcd, Rpq, Rpa, Rqc
    cdef int m
    cdef double Rp, Rq
    for m in range(3):
        Rab[m] = Ra[m] - Rb[m]
        Rcd[m] = Rc[m] - Rd[m]
        Rp = (ai * Ra[m] + aj * Rb[m]) / aij
        Rq = (ak * Rc[m] + al * Rd[m]) / akl
        Rpq[m] = Rp - Rq
        Rpa[m] = Rp - Ra[m]
        Rqc[m] = Rq - Rc[m]

    cdef double theta = aij * akl / (aij + akl)
    cdef double theta_r2 = theta * square(Rpq)
    cdef double Kabcd = 2*M_PI**2.5/(aij*akl*(aij+akl)**.5)
    Kabcd *= exp(-theta_ij * square(Rab) - theta_kl * square(Rcd))

    cdef int lij = li + lj
    cdef int lkl = lk + ll
    cdef int l4 = lij + lkl
    cdef int nroots = (l4 + 2) // 2
    rt, wt = rys_roots_weights(nroots, theta_r2)
    for m in range(nroots):
        wt[m] *= Kabcd

    if l4 == 0:
        return wt.reshape(1,1,1,1)

    cdef double[::1] _rt = rt
    cdef double[::1] _wt = wt
    cdef double theta_aij = theta / aij
    cdef double theta_akl = theta / akl
    cdef double fac_aij = .5/aij
    cdef double fac_akl = .5/akl
    cdef double fac_a1 = .5/(aij+ akl)
    cdef double Xpq = Rpq[0]
    cdef double Ypq = Rpq[1]
    cdef double Zpq = Rpq[2]
    cdef double Xpa = Rpa[0]
    cdef double Ypa = Rpa[1]
    cdef double Zpa = Rpa[2]
    cdef double Xqc = Rqc[0]
    cdef double Yqc = Rqc[1]
    cdef double Zqc = Rqc[2]
    cdef double Xtheta_aij = Xpq * theta_aij
    cdef double Ytheta_aij = Ypq * theta_aij
    cdef double Ztheta_aij = Zpq * theta_aij
    cdef double Xtheta_akl = Xpq * theta_akl
    cdef double Ytheta_akl = Ypq * theta_akl
    cdef double Ztheta_akl = Zpq * theta_akl

    trr = np.empty((3,l4+1,lkl+1,nroots))
    cdef double[:,:,::1] I2dx = trr[0]
    cdef double[:,:,::1] I2dy = trr[1]
    cdef double[:,:,::1] I2dz = trr[2]
    cdef int i, j, k, l, n
    
    for n in range(nroots):
        I2dx[0,0,n] = 1.
        I2dy[0,0,n] = 1.
        I2dz[0,0,n] = _wt[n]

    if l4 > 0:
        for n in range(nroots):
            I2dx[1,0,n] = (Xpa - Xtheta_aij*_rt[n]) * I2dx[0,0,n]
            I2dy[1,0,n] = (Ypa - Ytheta_aij*_rt[n]) * I2dy[0,0,n]
            I2dz[1,0,n] = (Zpa - Ztheta_aij*_rt[n]) * I2dz[0,0,n]
        for i in range(1, l4):
            for n in range(nroots):
                I2dx[i+1,0,n] = (Xpa - Xtheta_aij*_rt[n]) * I2dx[i,0,n] + i*fac_aij*(1-theta_aij*_rt[n]) * I2dx[i-1,0,n]
                I2dy[i+1,0,n] = (Ypa - Ytheta_aij*_rt[n]) * I2dy[i,0,n] + i*fac_aij*(1-theta_aij*_rt[n]) * I2dy[i-1,0,n]
                I2dz[i+1,0,n] = (Zpa - Ztheta_aij*_rt[n]) * I2dz[i,0,n] + i*fac_aij*(1-theta_aij*_rt[n]) * I2dz[i-1,0,n]

    if lkl > 0:
        for n in range(nroots):
            I2dx[0,1,n] = (Xqc + Xtheta_akl*_rt[n]) * I2dx[0,0,n]
            I2dy[0,1,n] = (Yqc + Ytheta_akl*_rt[n]) * I2dy[0,0,n]
            I2dz[0,1,n] = (Zqc + Ztheta_akl*_rt[n]) * I2dz[0,0,n]
        for i in range(1, l4):
            for n in range(nroots):
                I2dx[i,1,n] = (Xqc + Xtheta_akl*_rt[n]) * I2dx[i,0,n] + i*fac_a1*_rt[n] * I2dx[i-1,0,n]
                I2dy[i,1,n] = (Yqc + Ytheta_akl*_rt[n]) * I2dy[i,0,n] + i*fac_a1*_rt[n] * I2dy[i-1,0,n]
                I2dz[i,1,n] = (Zqc + Ztheta_akl*_rt[n]) * I2dz[i,0,n] + i*fac_a1*_rt[n] * I2dz[i-1,0,n]

    for k in range(1, lkl):
        for n in range(nroots):
            I2dx[0,k+1,n] = (Xqc + Xtheta_akl*_rt[n]) * I2dx[0,k,n] + k*fac_akl*(1-theta_akl*_rt[n]) * I2dx[0,k-1,n]
            I2dy[0,k+1,n] = (Yqc + Ytheta_akl*_rt[n]) * I2dy[0,k,n] + k*fac_akl*(1-theta_akl*_rt[n]) * I2dy[0,k-1,n]
            I2dz[0,k+1,n] = (Zqc + Ztheta_akl*_rt[n]) * I2dz[0,k,n] + k*fac_akl*(1-theta_akl*_rt[n]) * I2dz[0,k-1,n]
        for i in range(1, l4-k):
            for n in range(nroots):
                I2dx[i,k+1,n] = (Xqc + Xtheta_akl*_rt[n]) * I2dx[i,k,n] + k*fac_akl*(1-theta_akl*_rt[n]) * I2dx[i,k-1,n] + i*fac_a1*_rt[n] * I2dx[i-1,k,n]
                I2dy[i,k+1,n] = (Yqc + Ytheta_akl*_rt[n]) * I2dy[i,k,n] + k*fac_akl*(1-theta_akl*_rt[n]) * I2dy[i,k-1,n] + i*fac_a1*_rt[n] * I2dy[i-1,k,n]
                I2dz[i,k+1,n] = (Zqc + Ztheta_akl*_rt[n]) * I2dz[i,k,n] + k*fac_akl*(1-theta_akl*_rt[n]) * I2dz[i,k-1,n] + i*fac_a1*_rt[n] * I2dz[i-1,k,n]

    hrr = np.zeros((3, lij+1, lj+1, lkl+1, ll+1, nroots))
    cdef double[:,:,:,:,::1] I4dx = hrr[0]
    cdef double[:,:,:,:,::1] I4dy = hrr[1]
    cdef double[:,:,:,:,::1] I4dz = hrr[2]

    for i in range(lij+1):
        for k in range(lkl+1):
            for n in range(nroots):
                I4dx[i,0,k,0,n] = I2dx[i,k,n]
                I4dy[i,0,k,0,n] = I2dy[i,k,n]
                I4dz[i,0,k,0,n] = I2dz[i,k,n]
    cdef double Xab = Rab[0]
    cdef double Yab = Rab[1]
    cdef double Zab = Rab[2]
    cdef double Xcd = Rcd[0]
    cdef double Ycd = Rcd[1]
    cdef double Zcd = Rcd[2]

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

    cdef int nfi = n_cart(li)
    cdef int nfj = n_cart(lj)
    cdef int nfk = n_cart(lk)
    cdef int nfl = n_cart(ll)
    cdef int ix, iy, iz
    cdef int jx, jy, jz
    cdef int kx, ky, kz
    cdef int lx, ly, lz
    cdef double val

    eri = np.empty((nfi,nfj,nfk,nfl))
    cdef double[:,:,:,::1] _eri = eri
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
                                    val = 0
                                    for n in range(nroots):
                                        val += I4dx[ix,jx,kx,lx,n] * I4dy[iy,jy,ky,ly,n] * I4dz[iz,jz,kz,lz,n]
                                    _eri[i,j,k,l] = val
                                    l += 1
                            k += 1
                    j += 1
            i += 1
    return eri

def _contracted_ERI(bas_i, bas_j, bas_k, bas_l, intor):
    cdef int li = bas_i.angular_momentum
    cdef int lj = bas_j.angular_momentum
    cdef int lk = bas_k.angular_momentum
    cdef int ll = bas_l.angular_momentum
    Ra = bas_i.coordinates
    Rb = bas_j.coordinates
    Rc = bas_k.coordinates
    Rd = bas_l.coordinates
    cdef double[::1] norm_ci = bas_i.norm_coefficients
    cdef double[::1] norm_cj = bas_j.norm_coefficients
    cdef double[::1] norm_ck = bas_k.norm_coefficients
    cdef double[::1] norm_cl = bas_l.norm_coefficients
    cdef double[::1] exps_i = bas_i.exponents
    cdef double[::1] exps_j = bas_j.exponents
    cdef double[::1] exps_k = bas_k.exponents
    cdef double[::1] exps_l = bas_l.exponents
    cdef int nfi = n_cart(li)
    cdef int nfj = n_cart(lj)
    cdef int nfk = n_cart(lk)
    cdef int nfl = n_cart(ll)

    V = np.zeros((nfi, nfj, nfk, nfl))
    cdef double[:,:,:,::1] _V = V

    cdef int npi = len(exps_i)
    cdef int npj = len(exps_j)
    cdef int npk = len(exps_k)
    cdef int npl = len(exps_l)
    cdef int ip, jp, kp, lp
    cdef int i, j, k, l
    cdef double ai, aj, ak, al
    cdef double ci, cj, ck, cl
    cdef double fac
    cdef double[:,:,:,::1] _eri

    for ip in range(npi):
        ai = exps_i[ip]
        ci = norm_ci[ip]
        for jp in range(npj):
            aj = exps_j[jp]
            cj = norm_cj[jp]
            for kp in range(npk):
                ak = exps_k[kp]
                ck = norm_ck[kp]
                for lp in range(npl):
                    al = exps_l[lp]
                    cl = norm_cl[lp]
                    fac = ci * cj * ck * cl
                    _eri = intor(li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
                    for i in range(nfi):
                        for j in range(nfj):
                            for k in range(nfk):
                                for l in range(nfl):
                                    _V[i,j,k,l] += fac * _eri[i,j,k,l]
    return V
