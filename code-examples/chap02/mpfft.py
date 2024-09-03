import numpy as np
import mpmath as mp
mp.mp.dps = 30

def mpfftfreq(n):
    if n % 2 == 0:
        p1 = mp.arange(0, n//2)
        p2 = mp.arange(-n//2, 0)
    else:
        p1 = mp.arange(0, (n+1)//2)
        p2 = mp.arange(-(n-1)//2, 0)
    return np.append(p1, p2) / n

mpexp = np.vectorize(mp.exp)

def mpfft(a):
    n = a.shape[-1]
    kx = 2j*mp.pi * mpfftfreq(n)
    fac = mpexp(-np.arange(n)[:,None] * kx)
    return a.dot(fac)

def mpifft(a):
    n = a.shape[-1]
    kx = 2j*mp.pi * mpfftfreq(n)
    fac = mpexp(np.arange(n)[:,None] * kx)
    return a.dot(fac) / n

if __name__ == '__main__':
    a = np.random.rand(7)
    print(abs(np.fft.fft(a) - mpfft(a)).max())
    print(abs(a - mpifft(mpfft(a))).max())

    a = np.random.rand(3,6)
    print(abs(np.fft.fft(a) - mpfft(a)).max())
    print(abs(a - mpifft(mpfft(a))).max())
