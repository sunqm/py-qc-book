from cffi import FFI

sig = '''
double ddot_(int *n, double *dx, int *incx, double *dy, int *incy);
'''
ffibuilder = FFI()
ffibuilder.set_source('_blas_lite', sig, libraries=[':libblas.so.3'])
ffibuilder.cdef(sig)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
