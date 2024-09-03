import jinja2
from basis_set_exchange import get_basis
from basis_set_exchange.lut import element_Z_from_sym
from basis_set_exchange.manip import uncontract_general

molden_tpl = jinja2.Template('''\
[Molden Format]
Comment lines
[Atoms] ({{ unit or 'Ang' }})
{%- for element, atom_number, x, y, z in atoms %}
{{ element }} {{ loop.index }} {{ atom_number }} {{ x }} {{ y }} {{ z }}
{%- endfor %}
[GTO]
{%- for gtos_on_atom in gtos %}
{{ loop.index }} 0
{{ gtos_on_atom }}
{% endfor %}
[5d]
[7f]
[9g]
[MO]
{%- for mo in mos %}
{{ mo }}
{%- endfor %}
''')

gto_shell_tpl = jinja2.Template('''\
{{ spdf_shell }} {{ len(exponents) }} 1.00
{%- for e_cs in zip(exponents, *coefficients) %}
{{ ' '.join(e_cs) }}
{%- endfor %}''')

mo_tpl = jinja2.Template('''\
Sym= {{ symmetry or 'A' }}
Ene= {{ energy or 0.0 }}
Spin= {{ spin or 'Alpha' }}
Occup= {{ occupancy or 0.0 }}
{%- for c in coefficients %}
{{ loop.index }} {{ c }}
{%- endfor %}''')

def bse_to_gto_shell(shell):
    spdfg = 'spdfg'
    exponents = shell['exponents']
    coefficients = shell['coefficients']
    spdf_shell = ''.join(spdfg[l] for l in shell['angular_momentum'])
    return gto_shell_tpl.render(
        spdf_shell=spdf_shell, exponents=exponents, coefficients=coefficients,
        len=len, zip=zip)

def gto_session(basis_name, element):
    bse_data = get_basis(basis_name, element)
    bse_data = uncontract_general(bse_data)
    _, basis = bse_data['elements'].popitem()
    return '\n'.join([bse_to_gto_shell(shell) for shell in basis['electron_shells']])

def mo_session(orbitals):
    return [mo_tpl.render(coefficients=c) for c in orbitals]

def render_molden(elements, coordinates, basis_name, orbitals):
    x, y, z = coordinates.T
    atom_numbers = [element_Z_from_sym(ele) for ele in elements]
    atoms = list(zip(elements, atom_numbers, x, y, z))
    return molden_tpl.render(
        atoms=atoms,
        gtos=[gto_session(basis_name, atom[0]) for atom in atoms],
        mos=mo_session(orbitals))

if __name__ == '__main__':
    import pandas as pd
    import pyscf
    mole = pd.DataFrame({
        'symbol': ['O', 'H', 'H'],
        'x': [0.5009, 1.4049, -0.0934],
        'y': [2.7238, 2.4390, 1.9786],
        'z': [0.8464, 0.6938, 0.7321],
    })
    basis = 'sto-3g'
    mf = pyscf.M(atom=list(mole.to_numpy()), basis=basis, verbose=0).RHF().run()

    elements = mole.symbol
    coordinates = mole.loc[:,'x':'z'].to_numpy()
    with open('demo.molden', 'w') as f:
        f.write(render_molden(elements, coordinates, basis, mf.mo_coeff.T))
