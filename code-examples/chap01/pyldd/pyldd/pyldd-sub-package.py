#!/usr/bin/env python

'''
Find and print all dependent libraries required by a package.
'''

import os
import importlib
import ast
import click

def get_all_imports(path):
    raw_imports = set()
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if os.path.splitext(file_name)[1] != ".py":
                continue
            file_name = os.path.join(root, file_name)
            with open(file_name, 'r') as f:
                contents = f.read()
            try:
                tree = ast.parse(contents)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for subnode in node.names:
                            raw_imports.add(subnode.name)
                    elif isinstance(node, ast.ImportFrom):
                        raw_imports.add(node.module)
            except Exception as exc:
                print(f'Failed to parse file: {file_name}')
    return list(raw_imports)

@click.command(name='pyldd', help=__doc__)
@click.option('--depth', type=int, default=0, help='The maximum depth of sub-packages to inspect')
@click.argument('path', type=click.Path(exists=True, dir_okay=True))
def main(path, depth):
    imports = get_all_imports(path)
    imports = [imp for imp in imports if imp.count('.') <= depth]
    for name in set(imports):
        try:
            mod = importlib.import_module(name)
            print(mod)
        except Exception:
            print(f'Module {name} not found')


if __name__ == '__main__':
    main()
