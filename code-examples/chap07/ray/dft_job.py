import pyscf
import ray

@ray.remote
def dft_energy(molecule):
    mol = pyscf.M(atom=molecule, basis='def2-tzvp', verbose=4)
    mf = mol.RKS(xc='wb97x').density_fit().run()
    return mf.e_tot

molecules = ['O 0 0 0; H 0.757 0.5 0; H -0.757 0.5 0',
             'O 0 0 0; H 0.757 0.6 0; H -0.757 0.6 0']
futures = [dft_energy.remote(x) for x in molecules]
print(ray.get(futures))
