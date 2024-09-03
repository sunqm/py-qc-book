import numpy as np
from _blas_lite import ffi, lib

def blas_ddot(x: np.ndarray, y: np.ndarray) -> float:
    assert x.size == y.size
    assert x.dtype == y.dtype == np.float64
    assert x.ndim == y.ndim == 1

    # Get the address of array, similar to x.ctypes.data_as(c_double)
    px = ffi.cast('double *', x.ctypes.data)
    py = ffi.cast('double *', y.ctypes.data)
    # Allocate one int, similar to ctypes.byref(ctypes.c_int(x.size))
    size = ffi.new('int *', x.size)
    incx = ffi.new('int *', x.strides[0] // x.itemsize)
    incy = ffi.new('int *', y.strides[0] // y.itemsize)
    out = lib.ddot_(size, px, incx, py, incy)
    return out
