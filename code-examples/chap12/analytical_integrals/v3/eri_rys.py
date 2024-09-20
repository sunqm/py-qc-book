import numpy as np
from basis import CGTO, n_cart, iter_cart_xyz
from rys_roots import rys_roots_weights

def contracted_ERI(bas_i: CGTO, bas_j: CGTO, bas_k: CGTO, bas_l: CGTO) -> np.ndarray:
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
                    out += ci*cj*ck*cl * primitive_ERI(
                        li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd)
    return out

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

    Rpq = Rp[:,None] - Rq[:,None]
    Rpa = Rp[:,None] - Ra[:,None]
    Rqc = Rq[:,None] - Rc[:,None]

    trr = np.empty((lij+1,lkl+1,3,nroots))
    trr[0,0,0] = 1.
    trr[0,0,1] = 1.
    trr[0,0,2] = wt

    if lij > 0:
        trr[1,0] = (Rpa - Rpq*theta/aij*rt) * trr[0,0]
        for i in range(1, lij):
            val = (Rpa - Rpq*theta/aij*rt) * trr[i,0]
            val += i*.5/aij*(1-theta/aij*rt) * trr[i-1,0]
            trr[i+1,0] = val

    if lkl > 0:
        trr[0,1] = (Rqc + Rpq*theta/akl*rt) * trr[0,0]
        for i in range(1, lij+1):
            trr[i,1] = (Rqc + Rpq*theta/akl*rt) * trr[i,0] + i*.5/(aij+akl)*rt * trr[i-1,0]

    for k in range(1, lkl):
        val = (Rqc + Rpq*theta/akl*rt) * trr[0,k]
        val += k*.5/akl*(1-theta/akl*rt) * trr[0,k-1]
        trr[0,k+1] = val
        for i in range(1, lij+1):
            val = (Rqc + Rpq*theta/akl*rt) * trr[i,k]
            val += k*.5/akl*(1-theta/akl*rt) * trr[i,k-1]
            val += i*.5/(aij+akl)*rt * trr[i-1,k]
            trr[i,k+1] = val

    I4dx, I4dy, I4dz = np.zeros((3, lij+1, lj+1, lkl+1, ll+1, nroots))
    I4dx[:,0,:,0] = trr[:lij+1,:lkl+1,0]
    I4dy[:,0,:,0] = trr[:lij+1,:lkl+1,1]
    I4dz[:,0,:,0] = trr[:lij+1,:lkl+1,2]
    Xab, Yab, Zab = Rab
    Xcd, Ycd, Zcd = Rcd

    for l in range(1, ll+1):
        for k in range(lkl+1-l):
            I4dx[:,0,k,l] = I4dx[:,0,k+1,l-1] + Xcd * I4dx[:,0,k,l-1]
            I4dy[:,0,k,l] = I4dy[:,0,k+1,l-1] + Ycd * I4dy[:,0,k,l-1]
            I4dz[:,0,k,l] = I4dz[:,0,k+1,l-1] + Zcd * I4dz[:,0,k,l-1]

    for j in range(lj):
        for i in range(lij-j):
            I4dx[i,j+1,:lk+1,:ll+1] = I4dx[i+1,j,:lk+1,:ll+1] + Xab * I4dx[i,j,:lk+1,:ll+1]
            I4dy[i,j+1,:lk+1,:ll+1] = I4dy[i+1,j,:lk+1,:ll+1] + Yab * I4dy[i,j,:lk+1,:ll+1]
            I4dz[i,j+1,:lk+1,:ll+1] = I4dz[i+1,j,:lk+1,:ll+1] + Zab * I4dz[i,j,:lk+1,:ll+1]

    nfi = len(iter_cart_xyz(li))
    nfj = len(iter_cart_xyz(lj))
    nfk = len(iter_cart_xyz(lk))
    nfl = len(iter_cart_xyz(ll))
    eri = np.empty((nfi,nfj,nfk,nfl))
    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            for k, (kx, ky, kz) in enumerate(iter_cart_xyz(lk)):
                for l, (lx, ly, lz) in enumerate(iter_cart_xyz(ll)):
                    eri[i,j,k,l] = np.einsum('n,n,n->',
                                             I4dx[ix,jx,kx,lx],
                                             I4dy[iy,jy,ky,ly],
                                             I4dz[iz,jz,kz,lz])
    return eri
