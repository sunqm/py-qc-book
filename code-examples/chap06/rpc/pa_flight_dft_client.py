import pickle
import json
import numpy as np
from pyarrow import flight

def run_dft(geometry, basis_set, xc_func):
    request = json.dumps({
        "geometry": geometry,
        "basis_set": basis_set,
        "xc_func": xc_func
    })

    action = flight.Action('run_dft', request.encode())
    with flight.FlightClient('grpc://localhost:50011') as client:
        result = client.do_action(action)
        energy = pickle.loads(next(result).body)
        shape, dtype = pickle.loads(next(result).body)
        orbital_energies = np.frombuffer(next(result).body, dtype).reshape(shape)
        return energy, orbital_energies

if __name__ == '__main__':
    e, mo_energy = run_dft('O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0', 'sto-3g', 'b3lyp')
    print("DFT energy:", e)
    print("orbital energies:", mo_energy)
