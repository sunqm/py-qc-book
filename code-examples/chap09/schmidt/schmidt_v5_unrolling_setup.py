import jinja2
import numpy as np
from numba.pycc import CC

unroll_tpl = jinja2.Template('''
def schmidt_orth_n{{ n }}(s):
{%- for line in code %}
    {{ line }}
{%- endfor %}
''')

def unroll_schmidt_orth(n):
    code = []
    code.append(f'cs = np.zeros(({n}, {n}))')
    for j in range(n):
        code.append(f'fac = s[{j},{j}]')
        for k in range(j):
            code.append('dot_kj = 0.')
            for i in range(n):
                code.append(f'dot_kj += cs[{k},{i}] * s[{j},{i}]')
            for i in range(n):
                code.append(f'cs[{j},{i}] -= dot_kj * cs[{k},{i}]')
            code.append('fac -= dot_kj * dot_kj')

        code.append('fac = fac**-.5')
        code.append(f'cs[{j},{j}] = fac')
        for i in range(j):
            code.append(f'cs[{j},{i}] *= fac')
    code.append('return cs.T')

    exec(unroll_tpl.render(n=n, code=code))
    return vars()[f'schmidt_orth_n{n}']

if __name__ == "__main__":
    cc = CC('schmidt_unrolled')

    for i in range(8):
        fn = unroll_schmidt_orth(i)
        print(fn.__name__)
        cc.export(fn.__name__, 'f8[:,:](f8[:,:])')(fn)

    cc.compile()
