from os import path
import tempfile
import pickle
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.special import roots_legendre, roots_hermite
import rys_roots

MAX_RYS_ROOTS = 12
NODES = 2000

def tabulate_bspline():
    xs = np.linspace(0, 50, NODES)
    bs = []
    for nrys in range(1, MAX_RYS_ROOTS+1):
        rws = np.empty((len(xs),2,nrys))
        for i, x in enumerate(xs):
            r, w = rys_roots.rys_roots_weights(nrys, x)
            rws[i,0] = r
            rws[i,1] = w
        bs.append(make_interp_spline(xs, rws.reshape(len(xs),2*nrys)))
    return bs

db_file = path.join(tempfile.gettempdir(), 'bspline_tab.pkl')
bspline_tab = None
def polynomial_approx(nroots, t):
    global bspline_tab
    if bspline_tab is None:
        if path.exists(db_file):
            with open(db_file, 'rb') as f:
                bspline_tab = pickle.load(f)
        else:
            bspline_tab = tabulate_bspline()
            with open(db_file, 'wb') as f:
                pickle.dump(bspline_tab, f)
    bs = bspline_tab[nroots-1]
    rws = bs(t)
    return rws.reshape(2, nroots)

def rys_roots_weights(nroots, t):
    if t < 1e-8:
        leg_r, leg_w = roots_legendre(nroots*2)
        roots = leg_r[nroots:]**2
        weights = leg_w[nroots:] * (1-t*roots)
    elif t > 50:
        hermit_r, hermit_w = roots_hermite(nroots*2)
        roots = hermit_r[nroots:]**2 / t
        weights = hermit_w[nroots:] / t**.5
    else:
        roots, weights = polynomial_approx(nroots, t)
    return roots, weights

if __name__ == '__main__':
    polynomial_approx(1, 1.5)
