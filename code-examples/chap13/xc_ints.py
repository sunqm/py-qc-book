'''
Integrals for XC functionals
'''

import numpy as np

def eval_xc(density, xc_code, order=0):
    '''Returns the raw derivatives from XC libraries

    exc: the energy
    vrho: first partial derivative in terms of the density
    vsigma: first partial derivative in terms of sigma
    vtau: first partial derivative in terms of the kinetic energy density
    v2rho2, v2rhosigma, v2sigma2, v2rhotau, v2sigmatau, v2tau2:
        second order partial derivatives
    '''
    # Use pyscf libxc module to mimic the functionality of a XC functional library
    from pyscf.dft import libxc
    assert density.ndim == 3
    exc, vxc, fxc, kxc = libxc.eval_xc(density, xc_code, deriv=order)
    xctype = xc_family(xc_code)

    out = {}
    out['exc'] = exc * density[0]
    if order > 0:
        out['vrho'] = vxc[0].T
        if 'GGA' in xctype:
            out['vsigma'] = vxc[1].T
        if 'MGGA' in xctype:
            out['vtau'] = vxc[3].T
    if order > 1:
        out['v2rho2'] = fxc[0].T
        if 'GGA' in xctype:
            out['v2rhosigma'] = fxc[1].T
            out['v2sigma2'] = fxc[2].T
        if 'MGGA' in xctype:
            out['v2rhotau'] = fxc[6].T
            out['v2sigmatau'] = fxc[9].T
            out['v2tau2'] = fxc[4].T
    if order > 2:
        raise NotImplementedError
    return out

def xc_family(xc_code):
    '''Returns the family (LDA, GGA, MGGA) of a XC functional.
    '''
    from pyscf.dft import libxc
    return libxc.xc_type(xc_code)

def xc_deriv_tensor1(density, xc_code):
    '''
    Transform libxc functional derivatives to the derivative tensor
    corresponding to density tensor:
    [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
     [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].
    each element "density" is a N-size vector. The tensor shape is
    [2,1,N] for LDA, [2,4,N] for GGA, and [2,5,N] for MGGA.
    The first order derivative tensor in return has the shape
    [2,1,N] for LDA, [2,4,N] for GGA, and [2,5,N] for MGGA.
    '''
    einsum = np.einsum
    density = np.asarray(density)
    raw_xc = eval_xc(xc_code, density, spin=1, deriv=1)
    xctype = xc_family(xc_code)

    ngrids = density.shape[-1]
    if xctype == 'LDA':
        vp = raw_xc['vrho'].reshape(2, 1, ngrids)
    else:
        if xctype == 'GGA':
            vp = np.empty((2, 4, ngrids))
        else:
            vp = np.empty((2, 5, ngrids))
        vp[:,0] = raw_xc['vrho']
        vp[:,1:4] = einsum('abg,bxg->axg', conversion_2(raw_xc['vsigma']),
                           density[:,1:4])
        if xctype == 'MGGA':
            vp[:,4] = raw_xc['vtau']
    return vp

def xc_deriv_tensor2(density, xc_code):
    '''
    Transform libxc functional derivatives to the derivative tensor
    corresponding to density tensor:
    [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
     [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].
    each element "density" is a N-size vector. The tensor shape is
    [2,1,N] for LDA, [2,4,N] for GGA, and [2,5,N] for MGGA.
    The second order derivative tensor in return has the shape
    [2,1,2,1,N] for LDA, [2,4,2,4,N] for GGA, and [2,5,2,5,N] for MGGA.
    '''
    einsum = np.einsum
    density = np.asarray(density)
    raw_xc = eval_xc(xc_code, density, spin=1, deriv=2)
    xctype = xc_family(xc_code)

    ngrids = density.shape[-1]
    if xctype == 'LDA':
        vp = conversion_1(raw_xc['v2rho2']).reshape(2,1,2,1, ngrids)
    else:
        if xctype == 'GGA':
            vp = np.empty((2, 4, 2, 4, ngrids))
        else:
            vp = np.empty((2, 5, 2, 5, ngrids))
        vp[:,0,:,0] = conversion_1(raw_xc['v2rho2'])

        # First transforms to
        #[[uu_uu, ud_ud, ud_dd],
        # [ud_uu, ud_ud, ud_dd],
        # [dd_uu, dd_ud, dd_dd]]
        qgg = raw_xc['v2sigma2'][[[0, 1, 2],
                                  [1, 3, 4],
                                  [2, 4, 5]]]
        # Expands to
        #[[[uu_uu*2, ud_ud], [ud_ud, ud_dd*2]],
        # [[ud_uu*2, ud_ud], [ud_ud, ud_dd*2]],
        # [[dd_uu*2, dd_ud], [dd_ud, dd_dd*2]]]
        qgg = conversion_2(qgg, axis=1)
        # Expands to
        #[[[[uu_uu*2*2, ud_ud*2], [ud_ud*2, ud_dd*2*2]],
        #  [[ud_uu*2  , ud_ud  ], [ud_ud  , ud_dd*2  ]]],
        # [[[ud_uu*2  , ud_ud  ], [ud_ud  , ud_dd*2  ]],
        #  [[dd_uu*2*2, dd_ud*2], [dd_ud*2, dd_dd*2*2]]]]
        qgg = conversion_2(qgg, axis=0)
        vp[:,1:4,:,1:4] = einsum('abcdg,bxg,dyg->axcyg', qgg, density[:,1:4], density[:,1:4])
        qg = conversion_2(raw_xc['vsigma'])
        for i in range(1, 4):
            vp[:,i,:,i] += qg

        qrg = conversion_2(raw_xc['v2rhosigma'].reshape(2,3,ngrids), axis=1)
        qrg = einsum('rabg,bxg->raxg', qrg, density[:,1:4])
        vp[:,0,:,1:4] = qrg
        vp[:,1:4,:,0] = qrg.transpose(1,2,0,3)

        if xctype == 'MGGA':
            qgt = conversion_2(raw_xc['v2sigmatau'].reshape(3,2,ngrids), axis=0)
            qgt = einsum('abtg,axg->bxtg', qgt, density[:,1:4])
            vp[:,1:4,:,4] = qgt
            vp[:,4,:,1:4] = qgt.transpose(2,0,1,3)

            qrt = raw_xc['v2rhotau'].reshape(2,2,ngrids)
            vp[:,0,:,4] = qrt
            vp[:,4,:,0] = qrt.transpose(1,0,2)

            vp[:,4,:,4] = conversion_1(raw_xc['v2tau2'])
    return vp

def conversion_1(v, axis=0):
    '''[u_u, u_d, d_d] -> [[u_u, u_d], [d_u, d_d]]'''
    assert v.shape[axis] == 3
    return v[(slice(None),) * axis + ([[0,1],[1,2]],)]

def conversion_2(v, axis=0):
    '''[uu, ud, dd] -> [[uu*2, ud], [du, dd*2]]'''
    assert v.shape[axis] == 3
    v = v[(slice(None),) * axis + ([[0,1],[1,2]],)]
    v[(slice(None),)*axis + (0,0)] *= 2
    v[(slice(None),)*axis + (1,1)] *= 2
    return v

def get_vxc(gtos, dm, xc_code, grids, weights):
    from py_qc_book.chap13.eval_gto import eval_gtos
    einsum = np.einsum
    ao = eval_gtos(gtos, grids)
    nao, ngrids = ao.shape[1:]

    rho_ij = np.zeros((5, nao, nao, ngrids))
    xctype = xc_family(xc_code)
    match xctype:
        case 'LDA':
            rho_ij[0] = einsum('xig,xjg->ijg', ao[:1], ao[:1])
        case 'GGA':
            rho_ij[0:4] = einsum('xig,jg->xijg', ao[:4], ao[0])
            rho_ij[1:4] += einsum('ig,xjg->xijg', ao[0], ao[1:4])
        case 'MGGA':
            rho_ij[0:4] = einsum('xig,jg->xijg', ao[:4], ao[0])
            rho_ij[1:4] += einsum('ig,xjg->xijg', ao[0], ao[1:4])
            rho_ij[4] = einsum('xig,xjg->ijg', ao[1:4], ao[1:4]) * .5

    density = einsum('xijg,sji->sxg', rho_ij, dm)
    xc_tensor = xc_deriv_tensor1(density, xc_code)
    vxc = einsum('g,xijg,axg->aij', weights, rho_ij, xc_tensor)
    return vxc
