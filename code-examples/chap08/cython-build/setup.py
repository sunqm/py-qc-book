from setuptools import setup, Extension

ext1 = Extension(
        'example.simple_cpp',
        sources=['example_A.cpp', 'example_B.cpp'],
        language='c++',
)
setup(ext_modules=[ext1])
