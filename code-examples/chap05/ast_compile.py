import itertools
import inspect
import ast
import textwrap

class DepdenciesVisitor(ast.NodeVisitor):
    def __init__(self):
        self.deps = {}
        self.returns = []

    def visit_Assign(self, node):
        args = [x.id for x in ast.walk(node.value) if isinstance(x, ast.Name)]
        for t in node.targets:
            if isinstance(t, ast.Name):
                if t.id in self.deps:
                    self.deps[t.id].extend(args)
                else:
                    self.deps[t.id] = args

    def visit_AugAssign(self, node):
        args = [x.id for x in ast.walk(node.value) if isinstance(x, ast.Name)]
        target = node.target
        if isinstance(target, ast.Name):
            if target.id in self.deps:
                self.deps[target.id].extend(args)
            else:
                self.deps[target.id] = args

    def visit_Return(self, node):
        self.returns.extend(
            [x.id for x in ast.walk(node.value) if isinstance(x, ast.Name)])

class RemoveUnreachable(ast.NodeTransformer):
    def __init__(self, to_remove):
        self.to_remove = set(to_remove)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id in self.to_remove
               for t in node.targets):
            return None  # Remove the node
        return node

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in self.to_remove:
            return None  # Remove the node
        return node

    def visit_Expr(self, node):
        return None

def flatten_deps(keys, deps):
    if not keys or not deps:
        return []
    # Remove keys from deps to avoid potential circular dependency
    sub_deps = {k: v for k, v in deps.items() if k not in keys}
    flattened = [flatten_deps(deps[k], sub_deps) for k in keys if k in deps]
    return set(itertools.chain(keys, *flattened))

def remove_unreachable_statements(fn):
    orig_code = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    visitor = DepdenciesVisitor()
    visitor.visit(orig_code)
    # Search for variables that are accessible from returns
    referenced = flatten_deps(visitor.returns, visitor.deps)
    unreachable = set(visitor.deps).difference(referenced)

    filtered = RemoveUnreachable(unreachable).visit(orig_code)
    return ast.unparse(filtered)

def optimize(fn):
    '''Decorator for a Python function, which optimize the function with
    remove_unreachable_statements'''
    new_fn = remove_unreachable_statements(fn)
    new_code = ast.parse(new_fn)
    # Remove the line containing the decorator from the AST
    new_code.body[0].decorator_list = []

    local_ns = {}
    exec(ast.unparse(new_code), None, local_ns)
    return local_ns[fn.__name__]

if __name__ == '__main__':
    def fun1(x, y):
        a = x + 2
        b = a * x
        c = y * 9 - b
        d = f(b)
        b += d
        return b * a

    def fun2(x, y):
        a = x + 2
        b = a * x
        a, b = a + b, a - b
        c = f(y)
        d = c
        c += d
        d + x
        return b * a

    print(remove_unreachable_statements(fun1))
    print(remove_unreachable_statements(fun2))

    @optimize
    def fun3(x, y):
        a = x + 2
        b = a * x
        a, b = a + b, a - b
        c = y
        d = c
        c += d
        d + x
        return b * a

    print(fun3(2, 4))
