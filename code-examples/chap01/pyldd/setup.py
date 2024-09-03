from setuptools import setup, find_packages

setup(
    name='pyldd',
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            'pyldd = pyldd.cli:main'
        ]
    }
)
