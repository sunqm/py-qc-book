#!/usr/bin/env python

from os.path import dirname
from typing import List
import importlib
import click
from pipreqs import pipreqs

def dep_modules(path: str) -> List[str]:
    imports = pipreqs.get_all_imports(path)
    modules = {}
    for name in imports:
        try:
            mod = importlib.import_module(name)
            modules[name] = mod
        except Exception as e:
            print(repr(e))
    return modules

@click.command(name='pyldd')
@click.option('--depth', default=1, type=int, help="Recursively solve dependencies")
@click.option('--with-version', is_flag=True, help="Also print version of each library")
@click.argument('path', type=click.Path(exists=True, dir_okay=True))
def main(path, depth=1, with_version=False):
    '''
    Find and print all dependent libraries reuired by a package.
    '''
    imports = pipreqs.get_all_imports(path)
    paths = [path]
    sub_paths = []
    modules = {}
    for _ in range(depth):
        deps = {}
        for path in paths:
            deps.update(dep_modules(path))
        paths = [dirname(v.__file__) for v in deps.values()]
        modules.update(deps)

    for name, mod in modules.items():
        if with_version:
            try:
                print(f'{mod} version={mod.__version__}')
            except AttributeError:
                print(mod)
        else:
            print(mod)

if __name__ == '__main__':
    main()
