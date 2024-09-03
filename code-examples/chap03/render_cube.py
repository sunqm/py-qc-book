import numpy as np
import jinja2
from basis_set_exchange.lut import element_Z_from_sym

cube_tpl = jinja2.Template('''\
Comment line 1
Comment line 2
{{ '%5d' % len(atoms) }} {{ '%12.6f' % origin[0] }} {{ '%12.6f' % origin[1] }} {{ '%12.6f' % origin[2] }}
{%- set vx = voxel[0] %}
{%- set vy = voxel[1] %}
{%- set vz = voxel[2] %}
{{ '%5d' % n_voxels[0] }} {{ '%12.6f' % vx[0] }} {{ '%12.6f' % vx[1] }} {{ '%12.6f' % vx[2] }}
{{ '%5d' % n_voxels[1] }} {{ '%12.6f' % vy[0] }} {{ '%12.6f' % vy[1] }} {{ '%12.6f' % vy[2] }}
{{ '%5d' % n_voxels[2] }} {{ '%12.6f' % vz[0] }} {{ '%12.6f' % vz[1] }} {{ '%12.6f' % vz[2] }}
{%- set charge = 0 %}
{% for atom_number, x, y, z in atoms -%}
{{ '%5d' % atom_number }} {{ '%12.6f' % charge }} {{ '%12.6f' % x }} {{ '%12.6f' % y }} {{ '%12.6f' % z }}
{% endfor -%}
{% for data in volumetric_data -%}
{{ data }}
{% endfor -%}
''')

def render_cube(elements, coordinates, voxel, origin, data):
    atom_numbers = [element_Z_from_sym(ele) for ele in elements]
    x, y, z = coordinates.T
    atoms = list(zip(atom_numbers, x, y, z))
    n_voxels = data.shape
    formatted_data = []
    for ix in range(n_voxels[0]):
        for iy in range(n_voxels[1]):
            for iz in range(0, n_voxels[2], 6):
                # In each line, writes up to 6 floats in the 13.5E format
                formatted_data.append(
                    ''.join(f'{v:13.5E}' for v in data[ix,iy,iz:iz+6]))
    return cube_tpl.render(
        atoms=atoms, len=len,
        n_voxels=n_voxels, voxel=voxel, origin=origin,
        volumetric_data=formatted_data)

if __name__ == '__main__':
    import pandas as pd
    import pyscf
    mole = pd.DataFrame({
        'symbol': ['O', 'H', 'H'],
        'x': [0.5009, 1.4049, -0.0934],
        'y': [2.7238, 2.4390, 1.9786],
        'z': [0.8464, 0.6938, 0.7321],
    })

    # NOTE: all units are in Bohr
    boundary = [[-9., 9.],
                [-9., 9.],
                [-9., 9.]]
    mesh = [60, 60, 60]
    mgrids = np.mgrid[[slice(r[0], r[1], m*1j) for r, m in zip(boundary, mesh)]]
    grids = mgrids.reshape(3, -1).T
    origin = mgrids[:,0,0,0]
    voxel = np.array([mgrids[:,1,0,0] - origin,
                      mgrids[:,0,1,0] - origin,
                      mgrids[:,0,0,1] - origin])

    # AO values on given grids
    basis = 'sto-3g'
    mol = pyscf.M(atom=list(mole.to_numpy()), basis=basis, verbose=0)
    ao = mol.eval_gto('GTOval', grids)
    ao_2pz = ao[:,4].reshape(mesh)

    elements = mole.symbol
    # NOTE: unit in Bohr
    coordinates = mole.loc[:,'x':'z'].to_numpy() / 0.529177249
    with open('demo.cub', 'w') as f:
        f.write(render_cube(elements, coordinates, voxel, origin, ao_2pz))
