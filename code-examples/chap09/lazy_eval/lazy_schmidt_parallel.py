import time
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import networkx as nx

concurrent_executor = ThreadPoolExecutor(max_workers=4)

class Promise:
    _edges = []

    def __init__(self, func, *args):
        self._func = func
        self._args = args
        self._future = None
        Promise._edges.extend(
            # A directed edge, means self depends on x
            [(x, self) for x in args if isinstance(x, Promise)])

    def __repr__(self):
        return f'Deferred {self._func}'

    def compute(self):
        return self._func(*[force_eval(x) for x in self._args])

    def __call__(self):
        return self.compute()

    # Many features can be done in lazy evaluation mode, such as automatic
    # differentiation, dry-run for sanity check, visualization of the call graph

    def submit(self):
        def task():
            time.sleep(0.1)
            # dependent tasks will be automatically blocked by the .result()
            # method in force_eval() function.
            args = [force_eval(x) for x in self._args]
            # Release the reference to subsequent promises, allowing them to be recycled
            # once evaluated
            self._args = None
            return self._func(*args)
        self._future = concurrent_executor.submit(task)
        return self._future

    def result(self):
        if self._future is None:
            return self.compute()
        else:
            return self._future.result()

    def parallel_compute(self):
        # Build a dependency tree based on directed edges.
        tree = nx.DiGraph(Promise._edges)
        Promise._edges.clear()
        # topological_sort generates an iterator to traverse the dependency
        # tree. Leaves are evaluated first.
        futures = [task.submit() for task in nx.topological_sort(tree)]
        print(f'{len(futures)} tasks to execute')
        # self is is the last task (the root) in the dependency tree.
        return futures[-1].result()

def force_eval(x):
    if isinstance(x, Promise):
        return x.result()
    return x

def dot(a, b):
    return Promise(np.dot, a, b)

def subtract(a, b):
    return Promise(np.subtract, a, b)

def multiply(a, b):
    return Promise(np.multiply, a, b)

def divide(a, b):
    return Promise(np.divide, a, b)

def power(a, b):
    return Promise(np.power, a, b)

def build_array(*a):
    # Nested list should be considered for a complete implementation.
    # For simplicity, we only consider a plain list.
    return np.array([force_eval(x) for x in a])

def array(a):
    return Promise(build_array, *a)

def schmidt_orth_lazy(s):
    n = s.shape[0]
    cs = []
    guess = np.eye(n)
    for j in range(n):
        fac = s[j,j]
        vec = guess[j]
        for k in range(j):
            dot_kj = dot(cs[k], s[j])
            vec = subtract(vec, multiply(dot_kj, cs[k]))
            fac = subtract(fac, multiply(dot_kj, dot_kj))
        cs = cs + [divide(vec, power(fac, .5))]
    return Promise(np.transpose, array(cs))

if __name__ == '__main__':
    s = np.eye(3)
    cs = schmidt_orth_lazy(s)
    t0 = time.perf_counter()
    print(cs.parallel_compute())
    t1 = time.perf_counter()
    print(t1 - t0)
