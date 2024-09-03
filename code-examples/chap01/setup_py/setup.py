from setuptools import setup, Extension

ext1 = Extension(
    'example.simple_cpp',
    sources=['example_A.cpp', 'example_B.cpp'],
    language='c++',
    libraries=[':libblas.so.3.9.0'],
    #/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.9.0
    library_dirs=['/usr/lib/x86_64-linux-gnu/blas/'],
)
setup(ext_modules=[ext1])

