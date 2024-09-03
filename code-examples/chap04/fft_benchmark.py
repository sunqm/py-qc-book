import os
import time
import threading
import numpy as np
import scipy
import pyfftw
import mkl_fft

pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
pyfftw.interfaces.cache.enable()

nproc = int(os.getenv('OMP_NUM_THREADS', 1))

def timing(fn):
    def wrapped(*args, **kwargs):
        t0 = time.time()
        fn(*args, **kwargs)
        t1 = time.time()
        return t1 - t0
    return wrapped

def fft(x):
    n = x.shape[-1]
    m = np.exp(-2j*np.pi*np.arange(n)[:,None] * np.fft.fftfreq(n))
    return x.dot(m)

def rfft(x):
    assert x.dtype.kind != 'c'
    n = x.shape[-1]
    m = np.exp(-2j*np.pi*np.arange(n)[:,None] * np.fft.rfftfreq(n))
    return x.dot(m)

def fft2(x):
    n1, n2 = x.shape[-2:]
    m1 = np.exp(-2j*np.pi*np.arange(n1)[:,None] * np.fft.fftfreq(n1))
    m2 = np.exp(-2j*np.pi*np.arange(n2)[:,None] * np.fft.fftfreq(n2))
    x1 = m1.T.dot(x.reshape(-1, n2).T)
    y = m2.T.dot(x1.reshape(-1, n1).T)
    return y.reshape(n1, n2, -1).transpose(2, 0, 1)

def _npfft_priv(fn, x, y, i, block):
    i0, i1 = i * block, (i+1) * block
    y[i0:i1] = fn(x[i0:i1])

def npfft(fname, x):
    if nproc == 1:
        return getattr(np.fft, fname)(x)

    if fname.startswith('fft'):
        y = np.empty(x.shape, dtype=np.complex128)
    elif fname.startswith('rfft'):
        n = x.shape[-1]
        y = np.empty(x.shape[:-1] + ((n+2)//2,), dtype=np.complex128)
    else:
        raise

    fn = getattr(np.fft, fname)
    block = (x.shape[0] + nproc-1)//nproc

    threads = []
    for i in range(nproc):
        t = threading.Thread(target=_npfft_priv, args=(fn, x, y, i, block))
        t.start()
        threads.append(t)
    [t.join() for t in threads]
    return y

def run_once(fn, x):
    n = x.shape[-1]
    x1 = pyfftw.byte_align(x)
    t1 = timing(npfft)(fn, x)
    t2 = timing(globals()[fn])(x)
    t3 = timing(getattr(scipy.fft, fn))(x, workers=nproc)
    t4 = timing(getattr(pyfftw.interfaces.scipy_fft, fn))(
        x1, workers=nproc, planner_effort='FFTW_ESTIMATE')
    f = getattr(pyfftw.builders, fn)(x1, threads=nproc)
    f() # warm up
    t5 = timing(f)()
    t6 = timing(getattr(mkl_fft.interfaces.scipy_fft, fn))(x, workers=nproc)
    print('%-3d %.4f  %.4f  %.4f  %.4f  %.4f  %.4f' % (n, t1, t2, t3, t4, t5, t6))

start, stop = 3, 101
nv = 1000000
print('complex-to-complex FFT')
for n in range(start, stop):
    x = np.random.rand(nv, n)
    x = x + np.random.random(x.shape) * 1j
    run_once('fft', x)

print('real-to-complex FFT')
for n in range(start, stop):
    x = np.random.rand(nv, n)
    run_once('fft', x)

print('RFFT')
for n in range(start, stop):
    x = np.random.rand(nv, n)
    run_once('rfft', x)

print('complex-to-complex 2D-FFT')
for n in range(start, stop):
    nv = 100000 // n
    x = np.random.rand(nv, n, n)
    x = x + np.random.random(x.shape) * 1j
    run_once('fft2', x)
