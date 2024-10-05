import math
import numpy as np
import numba

@numba.njit(cache=True)
def gamma_inc(m, t):
#             _ 1           2
#            /     2 m  -t u
# F (t)  =   |    u    e      du,
#  m        _/  0
#
    assert m >= 0
    assert t >= 0
    # downward is alaways more accurate than upward, but ~3x slower
    if (t < m + 1.5):
        return downward(m, t)
    else:
        return upward(m, t)

@numba.njit(cache=True)
def downward(m, t, prec=1e-15):
#
# F[m] = int u^{2m} e^{-t u^2} du
#      = 1/(2m+1) int e^{-t u^2} d u^{2m+1}
#      = 1/(2m+1) [e^{-t u^2} u^{2m+1}]_0^1 + (2t)/(2m+1) int u^{2m+2} e^{-t u^2} du
#      = 1/(2m+1) e^{-t} + (2t)/(2m+1) F[m+1]
#      = 1/(2m+1) e^{-t} + (2t)/(2m+1)(2m+3) e^{-t} + (2t)^2/(2m+1)(2m+3) F[m+2]
#
    half = .5
    b = m + half
    e = half * np.exp(-t)
    x = e
    f = e
    while x > prec * e:
        b += 1
        x *= t / b
        f += x

    b = m + half
    f /= b
    out = np.empty(m+1)
    out[-1] = f
    for i in range(m):
        b -= 1
        f = (e + t * f) / b
        out[m-i-1] = f
    return out

@numba.njit(cache=True)
def upward(m, t):
#
# F[m] = int u^{2m} e^{-t u^2} du
#      = -1/2t int u^{2m-1} d e^{-t u^2}
#      = -1/2t [e^{-t u^2} * u^{2m-1}]_0^1 + (2m-1)/2t int u^{2m-2} e^{-t u^2} du
#      = 1/2t (-e^{-t} + (2m-1) F[m-1])
#
    half = .5
    tt = np.sqrt(t)
    f = np.sqrt(np.pi)/2 / tt * math.erf(tt)
    e = np.exp(-t)
    b = half / t
    out = np.empty(m+1)
    out[0] = f
    for i in range(m):
        f = b * ((2*i+1) * f - e)
        out[i+1] = f
    return out

@numba.njit(cache=True)
def schmidt_orth(moments, nroots):
    cs = np.zeros((nroots+1, nroots+1))
    for j in range(nroots+1):
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

@numba.njit(inline='always')
def poly_value1(a, order, x):
    p = a[order]
    for i in range(1, order+1):
        p = p * x + a[order-i]
    return p

@numba.njit(cache=True)
def rys_roots_weights(nroots, x):
    moments = gamma_inc(nroots*2, x)
    if moments[0] < 1e-16:
        return np.zeros(nroots), np.zeros(nroots)

    # Find polynomials roots
    cs = schmidt_orth(moments, nroots)
    roots = np.roots(cs[nroots,::-1])

    # Solve rtp.T.dot(diag(weights)).dot(rtp) = identity
    weights = np.zeros(nroots)
    for i in range(nroots):
        root = roots[i]
        dum = 1 / moments[0]
        for j in range(1, nroots):
            poly = poly_value1(cs[j,:j+1], j, root)
            dum += poly * poly
        weights[i] = 1 / dum
    return roots, weights
