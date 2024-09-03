import pyscf
from dask.distributed import Client

client = Client('3.28.54.133:8786')

def dft_energy(molecule):
    mol = pyscf.M(atom=molecule, basis='def2-tzvp', verbose=4)
    mf = mol.RKS(xc='wb97x').density_fit().run()
    return mf.e_tot

molecules = ['O 0 0 0; H 0.757 0.5 0; H -0.757 0.5 0',
             'O 0 0 0; H 0.757 0.6 0; H -0.757 0.6 0']
results = [client.submit(dft_energy, x) for x in molecules]
print(client.gather(results))

