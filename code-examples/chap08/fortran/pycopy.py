import ctypes
import numpy as np

libcopy = ctypes.CDLL('libfcopy.so')
libcopy.dcopy_.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.c_void_p,
]
libcopy.strcopy_.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
]

def dcopy(a: np.ndarray) -> np.ndarray:
    assert a.dtype == np.float64
    assert a.ndim == 1
    assert a.flags.contiguous
    b = np.empty_like(a)
    size = ctypes.c_int(a.size)
    libcopy.dcopy_(ctypes.byref(size), a.ctypes, b.ctypes)
    return b

def strcopy(a: bytes) -> bytes:
    assert isinstance(a, bytes)
    b = bytes(len(a))
    libcopy.strcopy_(a, b, len(a), len(b))
    return b

if __name__ == '__main__':
    print(dcopy(np.random.rand(3)))
    print(strcopy(b'XYZxyz'))
