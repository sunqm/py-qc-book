import math
from collections import deque
from itertools import product
from functools import lru_cache
from typing import List
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import numba

einsum = np.einsum

class String:
    def __init__(self, occupied_orbitals: List):
        self.occupied_orbitals = set(occupied_orbitals)

    def __repr__(self):
        return f'String{self.occupied_orbitals}'

    @classmethod
    def vacuum(cls):
        return cls(set())

    @classmethod
    def fully_occupied(cls, n):
        return cls(set(range(n)))

    def add_occupancy(self, orbital_id):
        assert orbital_id not in self.occupied_orbitals
        return String(self.occupied_orbitals.union([orbital_id]))

    def annihilate(self, orbital_id):
        '''Apply an annihilation operator. Returns the sign and a new determinant.'''
        if orbital_id not in self.occupied_orbitals:
            return 0, String.vacuum()
        sign = (-1) ** sum(i > orbital_id for i in self.occupied_orbitals)
        return sign, String(self.occupied_orbitals.difference([orbital_id]))

    def create(self, orbital_id):
        '''Apply a creation operator. Returns the sign and a new determinant.'''
        if orbital_id in self.occupied_orbitals:
            return 0., String.vacuum()
        sign = (-1) ** sum(i > orbital_id for i in self.occupied_orbitals)
        return sign, String(self.occupied_orbitals.union([orbital_id]))

    def as_bin(self):
        binstr = 0
        for i in self.occupied_orbitals:
            binstr |= (1 << i)
        return binstr

@lru_cache(200)
def make_strings(norb: int, noccupied: int):
    assert norb >= noccupied
    if norb == 0:
        return [String.vacuum()]
    elif noccupied == 0:
        return [String.vacuum()]
    elif norb == noccupied:
        return [String.fully_occupied(norb)]
    return (make_strings(norb-1, noccupied) +
            [s.add_occupancy(norb-1) for s in make_strings(norb-1, noccupied-1)])

@lru_cache(200)
def make_Elt(norb, nelec):
    '''The lookup table for non-zero elements of E tensor'''
    strings = make_strings(norb, nelec)
    strings_address = {s.as_bin(): i for i, s in enumerate(strings)}

    Elt = []
    for k, binI in enumerate(strings_address):
        table_k = []
        occs = []
        uoccs = []
        sign_cache = []
        sign = 1
        for i in reversed(range(norb)):
            sign_cache.append(sign)
            if (1 << i) & binI:
                occs.append(i)
                sign = -sign
            else:
                uoccs.append(i)
        sign_cache = sign_cache[::-1]
        occs = occs[::-1]
        uoccs = uoccs[::-1]

        for p in occs:
            table_k.append([p, p, k, 1])
        for q, p in product(occs, uoccs):
            binJ = binI ^ (1 << q) ^ (1 << p)
            if p > q:
                sign = sign_cache[p] * sign_cache[q]
            else:
                sign = -sign_cache[p] * sign_cache[q]
            table_k.append([p, q, strings_address[binJ], sign])
        Elt.append(table_k)
    return np.array(Elt, dtype=int)

def merge_h1_eri(h, eri, nelec):
    v = eri * .5
    if nelec > 0:
        f = (h - einsum('prrq->pq', eri) * .5) / (2 * nelec)
        for k in range(eri.shape[0]):
            v[k,k,:,:] += f
            v[:,:,k,k] += f
    return v

def _ensure(task):
    if isinstance(task, Future):
        return task.result()
    else:
        return task

def df_scheduler(queue, fn, tasks, args):
    '''Depth-first scheduler'''
    def fn_with_future(task):
        return fn(_ensure(task), *args)
    return (queue.submit(fn_with_future, task) for task in tasks)

def bf_scheduler(queue, fn, tasks, args):
    '''Breadth-first scheduler'''
    tasks = deque(df_scheduler(queue, fn, tasks, args))
    while tasks:
        yield tasks.popleft()

def df_warmup_scheduler(queue, fn, tasks, args):
    '''Launches several tasks at beginning then fills task queue like df_scheduler'''
    def fn_with_future(task):
        return fn(_ensure(task), *args)

    max_workers = queue._max_workers * 2
    future_buf = deque()
    for task in tasks:
        future_buf.append(queue.submit(fn_with_future, task))
        if len(future_buf) > max_workers:
            yield future_buf.popleft()
    while future_buf:
        yield future_buf.popleft()

class LaunchInFuture(Future):
    def __init__(self, queue, fn):
        self.queue = queue
        self.fn = fn
        super().__init__()

    def submit(self, task):
        future = self.queue.submit(self.fn, task)
        future.add_done_callback(self.pass_result)

    def pass_result(self, value):
        self.set_result(value.result())

def dynamic_scheduler(queue, fn, tasks, args):
    '''Dynamic scheduler submits tasks only if they are ready to proceed'''
    def fn_with_future(task):
        return fn(_ensure(task), *args)

    max_workers = queue._max_workers * 2
    future_buf = deque()
    for task in tasks:
        # Block the submission to prevent the task queue being filled with
        # too many operations of the same type
        if len(future_buf) >= max_workers:
            future_buf.popleft().result()
        if not isinstance(task, Future):
            future = queue.submit(fn_with_future, task)
        else:
            future = LaunchInFuture(queue, fn_with_future)
            task.add_done_callback(future.submit)
        future_buf.append(future)
        yield future

def pipeline(ops, tasks, scheduler=dynamic_scheduler):
    '''
    Streams operations and tasks
    '''
    queue, fn, args = ops[0]
    if queue is None:
        with ThreadPoolExecutor() as queue:
            return pipeline([(queue, fn, args), *ops[1:]], tasks, scheduler)

    if len(ops) == 1:
        # Call list() to traverse through the entire schduler generator which
        # will launch all tasks and their upstream tasks.
        return [_ensure(f) for f in list(scheduler(queue, fn, tasks, args))]

    return pipeline(ops[1:], scheduler(queue, fn, tasks, args), scheduler)

@numba.njit(nogil=True)
def build_d(task, fciwfn, norb, Elt):
    Ka0, Ka1, Kb0, Kb1 = task
    ma = Ka1 - Ka0
    mb = Kb1 - Kb0
    Elt_a, Elt_b = Elt
    d = np.zeros((norb, norb, ma, mb))
    for I, tab in enumerate(Elt_a[Ka0:Ka1]):
        for a, i, J, sign in tab:
            for K in range(mb):
                d[i,a,I,K] += sign * fciwfn[J,Kb0+K]
    for I, tab in enumerate(Elt_b[Kb0:Kb1]):
        for a, i, J, sign in tab:
            for K in range(ma):
                d[i,a,K,I] += sign * fciwfn[Ka0+K,J]
    return d, task

@numba.njit(nogil=True)
def dot_v(d_task, v):
    d, task = d_task
    norb, ma, mb = d.shape[1:]
    g = v.reshape(norb**2,-1).dot(d.reshape(norb**2,ma*mb))
    return g.reshape(norb,norb,ma,mb), task

def assemble_g(g_task, norb, Elt, sigma_pool):
    thread_id = get_ident()
    if thread_id in sigma_pool:
        sigma = sigma_pool[thread_id]
    else:
        Elt_a, Elt_b = Elt
        na = len(Elt_a)
        nb = len(Elt_b)
        sigma = np.zeros((na, nb))
        sigma_pool[thread_id] = sigma
    _assemble_g(g_task, norb, Elt, sigma)

@numba.njit(nogil=True)
def _assemble_g(g_task, norb, Elt, sigma):
    g, task = g_task
    Ka0, Ka1, Kb0, Kb1 = task
    ma = Ka1 - Ka0
    mb = Kb1 - Kb0
    Elt_a, Elt_b = Elt
    for I, tab in enumerate(Elt_a[Ka0:Ka1]):
        for a, i, J, sign in tab:
            for K in range(mb):
                sigma[J,Kb0+K] += sign * g[a,i,I,K]
    for I, tab in enumerate(Elt_b[Kb0:Kb1]):
        for a, i, J, sign in tab:
            for K in range(ma):
                sigma[Ka0+K,J] += sign * g[a,i,K,I]

def create_tasks(na, nb, blocksize):
    for Ka0 in range(0, na, blocksize):
        Ka1 = min(na, Ka0 + blocksize)
        for Kb0 in range(0, nb, blocksize):
            Kb1 = min(nb, Kb0 + blocksize)
            yield Ka0, Ka1, Kb0, Kb1

def compute_hc(h1, eri, fciwfn, norb, nelec_a, nelec_b, blocksize=40, threads=2):
    from threadpoolctl import ThreadpoolController
    Elt_a = make_Elt(norb, nelec_a)
    Elt_b = make_Elt(norb, nelec_b)
    Elt = (Elt_a, Elt_b)
    v = merge_h1_eri(h1, eri, nelec_a + nelec_b).reshape(norb**2,-1)
    na = len(Elt_a)
    nb = len(Elt_b)
    tasks = create_tasks(na, nb, blocksize)

    sigma_pool = {}
    if threads == 1:
        with ThreadPoolExecutor(max_workers=threads) as q1:
            pipeline([
                (q1, build_d, (fciwfn, norb, Elt)),
                (q1, dot_v, (v,)),
                (q1, assemble_g, (norb, Elt, sigma_pool)),
            ], tasks, df_scheduler)
        return sigma_pool.popitem()[1]

    blas_threads = max((threads+1)//2, 1)
    rest_threads = threads - blas_threads
    with ThreadpoolController().limit(limits=blas_threads, user_api='blas'):
        with ThreadPoolExecutor(max_workers=rest_threads) as q1:
            with ThreadPoolExecutor(max_workers=1) as q2:
                pipeline([
                    (q1, build_d, (fciwfn, norb, Elt)),
                    (q2, dot_v, (v,)),
                    (q1, assemble_g, (norb, Elt, sigma_pool)),
                ], tasks, dynamic_scheduler)
    return sum(sigma_pool.values())
