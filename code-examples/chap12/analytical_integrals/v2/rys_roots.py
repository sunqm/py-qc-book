import math
import numpy as np

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
    out = [f]
    for i in range(m):
        b -= 1
        f = (e + t * f) / b
        out.append(f)
    return np.array(out)[::-1]

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
    out = [f]
    for i in range(m):
        f = b * ((2*i+1) * f - e)
        out.append(f)
    return np.array(out)

def schmidt_orth(moments, nroots):
    '''
    Returns the coefficients of the orthogonal polynomials

    \sum_{kl} c_{i,k} c_{j,l} moments_{k+l} = \delta_{ij}
    '''
    s = np.zeros((nroots+1, nroots+1))
    for j in range(nroots+1):
        for i in range(nroots+1):
            s[i,j] = moments[i+j]
    return np.linalg.inv(np.linalg.cholesky(s))

def rys_roots_weights(nroots, x):
    moments = gamma_inc(nroots*2, x)
    if moments[0] < 1e-16:
        return np.zeros(nroots), np.zeros(nroots)

    # Find polynomials roots
    cs = schmidt_orth(moments, nroots)
    roots = np.roots(cs[nroots,::-1])

    rtp = roots[:,None]**np.arange(nroots+1)
    # Solve rtp.T.dot(diag(weights)).dot(rtp) = identity
    weights = 1. / (rtp.dot(cs.T)**2).sum(axis=1)
    return roots, weights
