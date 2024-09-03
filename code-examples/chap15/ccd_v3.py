import ast
import inspect
import functools
import tempfile
import textwrap
import threading
from typing import Dict, Union
from contextlib import contextmanager
import numpy as np
import h5py
import pyscf
from py_qc_book.chap13.diis import DIIS

def readahead_jit(f):
    # modify and cache the function until the first run
    new_f = None

    @functools.wraps(f)
    def f_with_readahead(*args, **kwargs):
        for i, a in enumerate(args):
            if isinstance(a, h5py.Group):
                break
        else:
            return f(*args, **kwargs)

        nonlocal new_f
        if new_f is None:
            arg_names = list(inspect.signature(f).parameters)
            new_f = _inject_readahead(f, arg_names[i])
        return new_f(*args, **kwargs)
    return f_with_readahead

def _search_tasks(f, keyword):
    tasks = []
    class IdentifyTask(ast.NodeVisitor):
        def visit_Subscript(self, node):
            if (isinstance(node.value, ast.Name) and node.value.id == keyword
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
                tasks.append(node.slice.value)
            return self.generic_visit(node)
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    IdentifyTask().visit(tree)
    print('# Readahead_jit finds tasks', tasks)
    return tasks

def _inject_readahead(f, keyword):
    tasks = _search_tasks(f, keyword)
    loader = f'_{keyword}_loader'
    assert loader not in f.__code__.co_varnames
    tree = ast.parse(inspect.getsource(f))
    fn_node = tree.body[0]
    fn_node.decorator_list = [] # remove the "@readahead" decorator
    fn_node.body = [
        ast.With([
            ast.withitem(
                ast.parse(f'readahead({keyword}, {tasks})', mode='eval').body,
                ast.Name(loader))], # as loader
            fn_node.body)
    ]

    class RewriteGetitem(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if (isinstance(node.value, ast.Name) and node.value.id == keyword
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
                node.value.id = loader
            return self.generic_visit(node)

    new_tree = ast.fix_missing_locations(RewriteGetitem().visit(tree))
    print(f'# Modified {f.__name__} function')
    print(ast.unparse(new_tree))
    exec(ast.unparse(new_tree))
    return locals()[f.__name__]

@contextmanager
def readahead(h5obj, tasks, maxsize=3):
    assert maxsize > 0
    cache = {}
    mutex = threading.RLock()
    not_full = threading.Condition(mutex)
    new_slot = threading.Condition(mutex)
    terminate = False

    class Data:
        def __init__(self, data):
            self.data = data
            self.refcount = 1

    def prefetch(tasks):
        for task in tasks:
            if terminate:
                break

            with mutex:
                if task in cache:
                    cache[task].refcount += 1
                    continue

            data = Data(np.asarray(h5obj[task]))
            with not_full:
                cache[task] = data
                new_slot.notify()
                if len(cache) >= maxsize:
                    not_full.wait()

    daemon = threading.Thread(target=prefetch, args=(tasks,))
    daemon.start()

    class Loader:
        def __getitem__(self, key):
            with new_slot:
                while key not in cache:
                    if len(cache) >= maxsize:
                        raise RuntimeError('Cache size insufficient')
                    new_slot.wait()
                data = cache[key].data
                cache[key].refcount -= 1
                if cache[key].refcount <= 0:
                    cache.pop(key)
                    not_full.notify()
            return data

    yield Loader()

    terminate = True
    with not_full:
        # release any locks in prefetch, then the terminate condition in
        # prefetch function will be triggered
        not_full.notify()
    daemon.join()

@readahead_jit
def update_CCD_amplitudes(H: Union[Dict, h5py.Group], t2: np.ndarray):
    nvir, nocc = t2.shape[1:3]
    fock = H['fock']
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]
    e_o = foo.diagonal()
    e_v = fvv.diagonal()

    Fvv = fvv - .5 * einsum('klcd,bdkl->bc', H['oovv'], t2)
    Foo = foo + .5 * einsum('klcd,cdjl->kj', H['oovv'], t2)
    Fvv[np.diag_indices(nvir)] -= e_v
    Foo[np.diag_indices(nocc)] -= e_o

    t2out = .25 * H['vvoo']
    t2out -= einsum('bkcj,acik->abij', H['vovo'], t2)
    t2out += .5 * einsum('bc,acij->abij', Fvv, t2)
    t2out -= .5 * einsum('kj,abik->abij', Foo, t2)
    t2out += .5*einsum('klcd,acik,bdjl->abij', H['oovv'], t2, t2)
    t2out = t2out - t2out.transpose(0,1,3,2)
    t2out = t2out - t2out.transpose(1,0,2,3)
    oooo = .5 * einsum('klcd,cdij->ijkl', H['oovv'], t2) + np.asarray(H['oooo'])
    t2out += .5 * einsum('ijkl,abkl->abij', oooo, t2)
    t2out += .5 * einsum('abcd,cdij->abij', H['vvvv'], t2)

    t2out /= e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None]
    return t2out

def get_CCD_corr_energy(H, t2):
    return .25 * einsum('ijab,abij->', H['oovv'], t2)

def mo_integrals(mol: pyscf.gto.Mole, orbitals, Hfile=None):
    '''MO integrals in physists notation <pq||rs>'''
    no = mol.nelectron
    nmo = orbitals.shape[1]
    eri = np.zeros([nmo*2]*4)
    eri[ ::2, ::2, ::2, ::2] = eri[ ::2, ::2,1::2,1::2] = \
    eri[1::2,1::2, ::2, ::2] = eri[1::2,1::2,1::2,1::2] = \
        pyscf.ao2mo.kernel(mol, orbitals, compact=False).reshape([nmo]*4)
    eri = eri.transpose(0,2,1,3) - eri.transpose(2,0,1,3)

    if Hfile is None:
        Hfile = tempfile.mktemp()
    with h5py.File(Hfile, 'w') as H:
        H['vvoo'] = vvoo = eri[no:,no:,:no,:no]
        H['oovv'] = vvoo.conj().transpose(2,3,0,1)
        H['vovo'] = eri[no:,:no,no:,:no]
        H['oooo'] = eri[:no,:no,:no,:no]
        H['vvvv'] = eri[no:,no:,no:,no:]

        hcore = pyscf.scf.hf.get_hcore(mol)
        hcore = einsum('pq,pi,qj->ij', hcore, orbitals, orbitals)
        hcore_mo = np.zeros([nmo*2]*2)
        hcore_mo[::2,::2] = hcore_mo[1::2,1::2] = hcore
        H['fock'] = hcore_mo + einsum('ipiq->pq', eri[:no,:,:no,:])
    return Hfile

def einsum(*args):
    return np.einsum(*args, optimize=True)

def mp2(H):
    nocc = H['oooo'].shape[0]
    fock = np.asarray(H['fock'])
    e_o = fock.diagonal()[:nocc]
    e_v = fock.diagonal()[nocc:]
    eijab = e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None]
    t2 = np.asarray(H['vvoo']) / eijab
    e = get_CCD_corr_energy(H, t2)
    return e, t2

def CCD_solve(mf: pyscf.scf.hf.RHF, conv_tol=1e-5, max_cycle=100):
    '''A fixed-point iteration solver for spin-orbital CCD'''
    mol = mf.mol
    orbitals = mf.mo_coeff
    e_hf = mf.e_tot

    with tempfile.TemporaryDirectory() as tmpdir:
        Hfile = mo_integrals(mol, orbitals, f'{tmpdir}/H')
        diis = DIIS(f'{tmpdir}/diis')
        e_ccd = e_hf
        with h5py.File(Hfile, 'r') as H:
            e_corr, t2 = mp2(H) # initial guess
            e_ccd = e_hf + e_corr
            print(f'E(MP2)={e_ccd}')

            for cycle in range(max_cycle):
                t2, t2_prev = update_CCD_amplitudes(H, t2), t2
                e_ccd, e_prev = get_CCD_corr_energy(H, t2) + e_hf, e_ccd
                print(f'{cycle=}, E(CCD)={e_ccd}, dE={e_ccd-e_prev}')
                if abs(t2 - t2_prev).max() < conv_tol:
                    break
                t2 = diis.update(t2 - t2_prev, t2)
    return e_ccd

if __name__ == '__main__':
    mol = pyscf.M(atom='N 0. 0 0; N 1.5 0 0', basis='cc-pvdz')
    mf = mol.RHF().run()
    e_ccd = CCD_solve(mf)
    assert abs(e_ccd - -109.0822455) < 1e-6
