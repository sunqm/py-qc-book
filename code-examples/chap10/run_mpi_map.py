'''
Execute this script with the command:

    mpirun -n 1 python run_mpi_map.py : -n 7 python mpi_map.py
'''

import numpy as np
import get_j_mpi_map

n = 25
dm = np.identity(n)
out = get_j_mpi_map.get_j(dm)
print(out.shape)
