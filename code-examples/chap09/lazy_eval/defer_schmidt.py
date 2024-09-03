import numpy as np

def force_eval(x):
    if callable(x): # deferred object
        return x()
    return x

def defer(func, *args):
    promise = lambda: func(*[force_eval(x) for x in args])
    return promise

def dot(a, b):
    return defer(np.dot, a, b)

def subtract(a, b):
    return defer(np.subtract, a, b)

def multiply(a, b):
    return defer(np.multiply, a, b)

def divide(a, b):
    return defer(np.divide, a, b)

def power(a, b):
    return defer(np.power, a, b)

def array(a):
    return defer(lambda: np.array([x() for x in a]))

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
    return defer(np.transpose, array(cs))

if __name__ == '__main__':
    s = np.eye(3)
    cs = schmidt_orth_lazy(s)
    print(cs())
