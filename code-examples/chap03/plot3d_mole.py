import itertools
import numpy as np
import pandas as pd
from mayavi import mlab

R = (1, 0, 0)
G = (0, 1, 0)
B = (0, 0, 1)
W = (1, 1, 1)
K = (0, 0, 0)

covalent_radius = pd.Series({
    'H': 0.31,
    'C': 0.73,
    'N': 0.71,
    'O': 0.66,
})

color_map = pd.Series({
    'H': (.8, .8, .8),
    'C': (.6, .6, .6),
    'N': B,
    'O': R,
})

def view_surface(mole, grids, v, contours=5):
    mlab.figure(1, bgcolor=W, size=(500, 500))
    geom = mole.loc[:,'x':'z'].to_numpy()
    radii = covalent_radius[mole.symbol].to_numpy()
    cmap = color_map[mole.symbol].to_numpy()
    for xyz, r, c in zip(geom, radii, cmap):
        mlab.points3d(*xyz, scale_factor=r, resolution=30, color=c)

    # Bonds between atoms
    dist = np.linalg.norm(geom[:,None] - geom, axis=2)
    conn = dist < radii[:,None] + radii
    for i, j in np.argwhere(conn):
        if i < j:
            xyz = geom[[i,j]]
            mlab.plot3d(xyz[:,0], xyz[:,1], xyz[:,2], tube_radius=0.06)

    mlab.contour3d(grids[0], grids[1], grids[2], v,
                   contours=contours, transparent=True)

    mlab.view(75, 0, 7)

    mlab.show()

if __name__ == '__main__':
    from scipy.special import erf
    mole = pd.DataFrame({
        'symbol': ['O', 'H', 'H'],
        'x': [0.5009, 1.4049, -0.0934],
        'y': [2.7238, 2.4390, 1.9786],
        'z': [0.8464, 0.6938, 0.7321],
        'Q': [-0.64, 0.32, 0.32],
    })
    print(mole)

    xyz = mole.loc[:,'x':'z'].to_numpy()
    xyz_center = xyz.mean(axis=0)

    nx, ny, nz = (80, 80, 80)
    delta = 0.1
    xr = np.arange(nx) * delta + xyz_center[0] - nx*delta/2
    yr = np.arange(ny) * delta + xyz_center[1] - ny*delta/2
    zr = np.arange(nz) * delta + xyz_center[2] - nz*delta/2
    grids = np.array(np.meshgrid(xr, yr, zr, indexing='ij'))
    r = np.linalg.norm(grids - xyz[:,:,None,None,None], axis=1)
    radii = covalent_radius[mole.symbol].to_numpy()
    zeta = 1./radii
    v = np.einsum('q,qxyz->xyz', mole.Q, erf(zeta[:,None,None,None]*r)/r)
    view_surface(mole, grids, v)
    view_surface(mole, grids, v, [-.1, .1])
