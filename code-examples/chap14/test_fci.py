import math
import numpy as np
from fci_v1 import merge_h1_eri

def pyscf_reference(h1, eri, fciwfn, norb, nelec_a, nelec_b):
    from pyscf import fci
    v = merge_h1_eri(h1, eri, nelec_a+nelec_b)
    return fci.direct_nosym.contract_2e(v, fciwfn, norb, (nelec_a,nelec_b))

def run_hc(f):
    np.random.seed(9)
    norb = 5
    nelec_a, nelec_b = (2, 2)
    h1 = np.random.rand(norb, norb)
    eri = np.random.random([norb]*4)
    eri += eri.transpose(2,3,0,1)
    na = math.comb(norb, nelec_a)
    nb = math.comb(norb, nelec_b)
    fciwfn = np.random.rand(na, nb)
    sigma = f(h1, eri, fciwfn, norb, nelec_a, nelec_b)
    ref = pyscf_reference(h1, eri, fciwfn, norb, nelec_a, nelec_b)
    assert abs(ref - sigma).max() < 1e-12

def test_fci_v1():
    from fci_v1 import compute_hc
    run_hc(compute_hc)

def test_fci_v2():
    from fci_v2 import compute_hc
    run_hc(compute_hc)

def test_fci_v3():
    from fci_v3 import compute_hc
    run_hc(compute_hc)

def test_fci_pipeline():
    from fci_pipeline import compute_hc
    run_hc(compute_hc)

def test_h_diag():
    from pyscf import fci
    from fci_v1 import make_hdiag
    np.random.seed(9)
    norb = 5
    nelec_a, nelec_b = (2, 2)
    h1 = np.random.rand(norb, norb)
    eri = np.random.random([norb]*4)
    eri += eri.transpose(2,3,0,1)
    ref = fci.direct_spin1.make_hdiag(h1, eri, norb, (nelec_a, nelec_b))
    hdiag = make_hdiag(h1, eri, norb, nelec_a, nelec_b)
    assert abs(ref - hdiag).max() < 1e-12

def test_fci_solve():
    from pyscf import fci
    from fci_v1 import make_hdiag
    np.random.seed(9)
    norb = 5
    nelec_a, nelec_b = (2, 2)
    h1 = np.random.rand(norb, norb)
    h1 = h1 + h1.T
    eri = np.random.random([norb]*4)
    eri += eri.transpose(1,0,2,3)
    eri += eri.transpose(0,1,3,2)
    eri += eri.transpose(2,3,0,1)
    ref = fci.direct_spin1.kernel(h1, eri, norb, (nelec_a, nelec_b))[0]
    e = FCI_solve(h1, eri, norb, nelec_a, nelec_b)[0]
    assert abs(e - ref) < 1e-10
