import math
import mpmath as mp
import numpy as np

DECIMALS = 30
mp.mp.dps = DECIMALS
mp.mp.pretty = True

def boys(m, t):
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

def downward(m, t, prec=.1**DECIMALS):
    #
    # F[m] = int u^{2m} e^{-t u^2} du
    #      = 1/(2m+1) int e^{-t u^2} d u^{2m+1}
    #      = 1/(2m+1) [e^{-t u^2} u^{2m+1}]_0^1 + (2t)/(2m+1) int u^{2m+2} e^{-t u^2} du
    #      = 1/(2m+1) e^{-t} + (2t)/(2m+1) F[m+1]
    #      = 1/(2m+1) e^{-t} + (2t)/(2m+1)(2m+3) e^{-t} + (2t)^2/(2m+1)(2m+3) F[m+2]
    #
    e = mp.mpf('.5') * mp.exp(-t)
    x = e
    s = e
    b = m + mp.mpf('1.5')
    while x > prec:
        x *= t / b
        s += x
        b += 1

    b = m + mp.mpf('.5')
    f = s / b
    out = [f]
    for i in range(m):
        b -= 1
        f = (e + t * f) / b
        out.append(f)
    return np.array(out[::-1])

def upward(m, t):
    #
    # F[m] = int u^{2m} e^{-t u^2} du
    #      = -1/2t int u^{2m-1} d e^{-t u^2}
    #      = -1/2t [e^{-t u^2} * u^{2m-1}]_0^1 + (2m-1)/2t int u^{2m-2} e^{-t u^2} du
    #      = 1/2t (-e^{-t} + (2m-1) F[m-1])
    #
    tt = mp.sqrt(t)
    f = mp.sqrt(mp.pi)/2 / tt * mp.erf(tt)
    e = mp.exp(-t)
    b = mp.mpf('.5') / t
    out = [f]
    for i in range(m):
        f = b * ((2*i+1) * f - e)
        out.append(f)
    return np.array(out)

def schmidt_orth(moments, nroots):
    r'''
    Returns the coefficients of the orthogonal polynomials

    \sum_{kl} c_{i,k} c_{j,l} moments_{k+l} = \delta_{ij}
    '''
    n1 = nroots + 1
    cs = np.array([mp.mpf(0)] * n1**2).reshape(n1, n1)
    for j in range(n1):
        fac = moments[j+j]
        for k in range(j):
            dot = cs[k,:j+1].dot(moments[j:j+j+1])
            cs[j,:j] -= dot * cs[k,:j]
            fac -= dot * dot

        if fac <= 0:
            raise RuntimeError(f'schmidt_orth fail. {nroots=} {fac=}')
        fac = fac**mp.mpf('-.5')
        cs[j,j] = fac
        cs[j,:j] *= fac
    return cs

def find_polyroots(cs, nroots):
    assert len(cs) == nroots + 1
    if nroots == 1:
        return np.array([-cs[0] / cs[1]])
    #roots = mp.polyroots(cs[::-1]) # convergence issue ccasionally
    A = mp.matrix(nroots)
    for m in range(nroots-1):
        A[m+1,m] = mp.mpf(1)
    for m in range(nroots):
        A[0,m] = -cs[nroots-1-m] / cs[nroots]
    roots = mp.eig(A, left=False, right=False)
    return np.array([x.real for x in roots])

def rys_roots_weights(nroots, x):
    moments = boys(nroots*2, mp.mpf(x))
    cs = schmidt_orth(moments, nroots)
    roots = find_polyroots(cs[nroots], nroots)

    rtp = roots[:,None]**np.arange(nroots+1)
    # Solve rtp.T.dot(diag(weights)).dot(rtp) = identity
    weights = 1. / (rtp.dot(cs.T)**2).sum(axis=1)
    return roots, weights
