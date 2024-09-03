from os import path
import tempfile
import pickle
import numpy as np
import scipy.special
import numba
import rys_roots

MAX_RYS_ROOTS = 12
INTERVAL = 3.
DEGREE = 11

def tabulate_chebfit():
    n = DEGREE + 1
    cheb_nodes = np.cos(np.pi / n * (np.arange(n) + .5))

    intervals = [(a, a+INTERVAL) for a in np.arange(0., 50., INTERVAL)]
    cs = []
    for nrys in range(1, MAX_RYS_ROOTS+1):
        cs_i = []
        for a, b in intervals:
            # Map chebyshev nodes to the sample points between interval (a, b)
            xs = cheb_nodes*(b-a)/2 + (b+a)/2
            rws = np.empty((n,2,nrys))
            for i, x in enumerate(xs):
                r, w = rys_roots.rys_roots_weights(nrys, x)
                rws[i,0] = r
                rws[i,1] = w
            cs_i.append(
                np.polynomial.chebyshev.chebfit(
                    cheb_nodes, rws.reshape(n,2*nrys), DEGREE))
        cs.append(np.array(cs_i))
    return cs

db_file = path.join(tempfile.gettempdir(), 'chebfit_tab.pkl')
chebfit_tab = None
def polynomial_approx(nroots, t):
    global chebfit_tab
    if chebfit_tab is None:
        if path.exists(db_file):
            with open(db_file, 'rb') as f:
                chebfit_tab = pickle.load(f)
        else:
            chebfit_tab = tabulate_chebfit()
            with open(db_file, 'wb') as f:
                pickle.dump(chebfit_tab, f)
    return _cheb_eval(nroots, t, chebfit_tab[nroots-1])

@numba.njit
def _cheb_eval(nroots, t, chebfit_tab):
    interval_id = int(t // INTERVAL)
    c = chebfit_tab[interval_id]
    a = interval_id * INTERVAL
    b = a + INTERVAL
    x = (t - (a+b)/2) / ((b-a)/2)

    rws = np.empty(2*nroots)
    for n in range(2*nroots):
        x2 = 2*x
        c0 = c[DEGREE-1,n]
        c1 = c[DEGREE,n]
        for i in range(2, DEGREE + 1):
            tmp = c0
            c0 = c[DEGREE-i,n] - c1
            c1 = tmp + c1*x2
        rws[n] = c0 + c1*x
    return rws.reshape(2,nroots)

def rys_roots_weights(nroots, t):
    if t < 1e-8:
        leg_r, leg_w = scipy.special.roots_legendre(nroots*2)
        roots = leg_r[nroots:]**2
        weights = leg_w[nroots:] * (1-t*roots)
    elif t > 50:
        hermit_r, hermit_w = scipy.special.roots_hermite(nroots*2)
        roots = hermit_r[nroots:]**2 / t
        weights = hermit_w[nroots:] / t**.5
    else:
        roots, weights = polynomial_approx(nroots, t)
    return roots, weights
