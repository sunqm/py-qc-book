from itertools import product
import numpy as np
from basis import CGTO, n_cart, iter_cart_xyz
from rys_roots import rys_roots_weights
import eri_OS

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
    l_4 = lij + lkl
    nroots = (l_4 + 2) // 2
    rt, wt = rys_roots_weights(nroots, theta_r2)
    wt *= Kabcd

    Rab = Rab[:,None]
    Rcd = Rcd[:,None]
    Rpq = Rpq[:,None]
    Rpa = Rp[:,None] - Ra[:,None]
    Rqc = Rq[:,None] - Rc[:,None]

    trr = np.zeros((l_4+1,lkl+1,3,nroots))
    for k in range(lkl+1):
        if k == 0:
            for i in range(l_4+1-k):
                if i == 0:
                    trr[0,0] = 1.
                elif i == 1:
                    trr[1,k] = (Rpa - Rpq*theta/aij*rt) * trr[0,k]
                else:
                    val = (Rpa - Rpq*theta/aij*rt) * trr[i-1,0]
                    val += (i-1)*.5/aij*(1-theta/aij*rt) * trr[i-2,0]
                    trr[i,0] = val
        elif k == 1:
            for i in range(l_4+1-k):
                if i == 0:
                    trr[i,1] = (Rqc + Rpq*theta/akl*rt) * trr[i,0]
                else:
                    trr[i,1] = (Rqc + Rpq*theta/akl*rt) * trr[i,0] + i*.5/(aij+akl)*rt * trr[i-1,0]
        else:
            for i in range(l_4+1-k):
                if i == 0:
                    val = (Rqc + Rpq*theta/akl*rt) * trr[i,k-1]
                    val += (k-1)*.5/akl*(1-theta/akl*rt) * trr[i,k-2]
                    trr[i,k] = val
                else:
                    val = (Rqc + Rpq*theta/akl*rt) * trr[i,k-1]
                    val += (k-1)*.5/akl*(1-theta/akl*rt) * trr[i,k-2]
                    val += i*.5/(aij+akl)*rt * trr[i-1,k-1]
                    trr[i,k] = val

    hrr = np.zeros((lij+1, lj+1, lkl+1, ll+1, 3, nroots))
    for j in range(lj+1):
        for i in range(lij+1-j):
            for l in range(ll+1):
                for k in range(lkl+1-l):
                    if j > 0:
                        hrr[i,j,k,l] = hrr[i+1,j-1,k,l] + Rab*hrr[i,j-1,k,l]
                    elif l > 0:
                        hrr[i,j,k,l] = hrr[i,j,k+1,l-1] + Rcd*hrr[i,j,k,l-1]
                    else:
                        hrr[i,0,k,0] = trr[i,k]

    nfi = len(iter_cart_xyz(li))
    nfj = len(iter_cart_xyz(lj))
    nfk = len(iter_cart_xyz(lk))
    nfl = len(iter_cart_xyz(ll))
    eri = np.empty((nfi,nfj,nfk,nfl))
    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            for k, (kx, ky, kz) in enumerate(iter_cart_xyz(lk)):
                for l, (lx, ly, lz) in enumerate(iter_cart_xyz(ll)):
                    eri[i,j,k,l] = np.einsum('n,n,n,n->', wt,
                                             hrr[ix,jx,kx,lx,0],
                                             hrr[iy,jy,ky,ly,1],
                                             hrr[iz,jz,kz,lz,2])
    return eri
