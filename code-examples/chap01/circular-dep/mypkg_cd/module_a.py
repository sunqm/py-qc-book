from . import module_b

class A:
    pass

def factory():
    return module_b.B()
