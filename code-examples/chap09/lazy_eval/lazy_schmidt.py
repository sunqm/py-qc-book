import numpy as np

class Promise:
    count = 0

    def __init__(self, func, *args):
        self._func = func
        self._args = args

    def __repr__(self):
        return f'Deferred {self._func}'

    def __call__(self):
        return self.compute()

    def compute(self):
        Promise.count += 1
        return self._func(*[force_eval(x) for x in self._args])

def force_eval(x):
    if isinstance(x, Promise):
        return x.compute()
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

def build_array(a):
    if isinstance(a, Promise):
        return np.array(a.compute())
    if not isinstance(a, (tuple, list)):
        return np.array(a)
    # Nested list should be considered for a complete implementation.
    # For simplicity, we only consider a plain list.
    return np.array([force_eval(x) for x in a])

def array(a):
    return Promise(build_array, a)

def schmidt_orth_lazy(s, idx=slice(None)):
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
    return Promise(np.transpose, array(cs[idx]))

if __name__ == '__main__':
    s = np.eye(3)

    cs = schmidt_orth_lazy(s)
    cs.compute()
    print(Promise.count)

    Promise.count = 0
    cs = schmidt_orth_lazy(s, -1)
    cs.compute()
    print(Promise.count)
