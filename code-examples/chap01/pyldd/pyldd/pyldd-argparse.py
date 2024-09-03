#!/usr/bin/env python

'''
Find and print all dependent libraries required by a package.
'''

import os
import importlib
import ast
import argparse
import pathlib

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

def main():
    parser = argparse.ArgumentParser(prog='pyldd', description=__doc__)
    parser.add_argument('path', type=pathlib.Path)
    parser.add_argument('--depth', type=int, default=0, help='The maximum depth of sub-packages to inspect')
    args = parser.parse_args()

    imports = get_all_imports(args.path)
    imports = [imp for imp in imports if imp.count('.') <= args.depth]
    for name in set(imports):
        try:
            mod = importlib.import_module(name)
            print(mod)
        except Exception:
            print(f'Module {name} not found')


if __name__ == '__main__':
    main()
