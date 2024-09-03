import numpy as np
from py_qc_book.chap12.analytical_integrals.v5.basis import n_cart, iter_cart_xyz

def eval_gtos(gtos, grids):
    '''GTO value on grids up to the first order derivative.

    Returns a tensor with shape [4,nao,ngrids] . The leading dimension
    represents for the zeroth order, and the three components of the first order
    derivatives of GTO values on grids.
    '''
    ngrids = grids.shape[0]
    ao_value = []
    for gto in gtos:
        l = gto.angular_momentum
        r = gto.coordinates
        dx, dy, dz = (grids - r).T
        r2 = dx * dx + dy * dy + dz * dz
        ao = np.zeros((4, n_cart(l), ngrids))
        ao_value.append(ao)

        for e, c in zip(gto.exponents, gto.coefficients):
            fac = c * np.exp(-e * r2)
            gtox = dx**np.arange(0, l+2)[:,np.newaxis]
            gtoy = dy**np.arange(0, l+2)[:,np.newaxis]
            gtoz = dz**np.arange(0, l+2)[:,np.newaxis]
            for n, (lx, ly, lz) in enumerate(iter_cart_xyz(l)):
                ao[0,n] += fac * gtox[lx] * gtoy[ly] * gtoz[lz]
                # d/dx
                ao[1,n] -= 2 * e * fac * gtox[lx+1] * gtoy[ly] * gtoz[lz]
                if lx > 0:
                    ao[1,n] += fac * lx * gtox[lx-1] * gtoy[ly] * gtoz[lz]
                # d/dy
                ao[2,n] -= 2 * e * fac * gtox[lx] * gtoy[ly+1] * gtoz[lz]
                if ly > 0:
                    ao[2,n] += fac * ly * gtox[lx] * gtoy[ly-1] * gtoz[lz]
                # d/dz
                ao[3,n] -= 2 * e * fac * gtox[lx] * gtoy[ly] * gtoz[lz+1]
                if lz > 0:
                    ao[3,n] += fac * lz * gtox[lx] * gtoy[ly] * gtoz[lz-1]
    return np.concatenate(ao_value, axis=1)

