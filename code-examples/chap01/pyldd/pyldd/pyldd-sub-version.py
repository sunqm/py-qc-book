#!/usr/bin/env python

import importlib
import click
from pipreqs import pipreqs

@click.command(name='pyldd')
@click.option('--with-version', is_flag=True, help="Also print version of each library")
@click.argument('path', type=click.Path(exists=True, dir_okay=True))
def main(path, with_version=False):
    '''
    Find and print all dependent libraries reuired by a package.
    '''
    imports = pipreqs.get_all_imports(path)
    for name in imports:
        try:
            mod = importlib.import_module(name)
        except Exception:
            print(f'Failed to inspect module {name}')
        if with_version:
            try:
                print(f'{mod} version={mod.__version__}')
            except AttributeError:
                print(mod)
        else:
            print(mod)


if __name__ == '__main__':
    main()
