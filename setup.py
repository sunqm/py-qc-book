import os
from setuptools import setup, find_packages

setup(
    name='py_qc_book',
    version='1.0',
    author='Qiming Sun',
    packages=find_packages(include=['py_qc_book*']),
    python_requires='>=3.10',
)
