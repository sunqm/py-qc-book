from setuptools import setup

setup(
    name='blas_lite',
    version='0.1',
    cffi_modules=['blas_lite_builder.py:ffibuilder'],
)
