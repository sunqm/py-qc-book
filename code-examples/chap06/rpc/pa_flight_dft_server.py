import pickle
import json
import numpy as np
import pyarrow.flight as flight

def run_dft(geometry, basis_set, xc_func):
    import pyscf
    mf = pyscf.M(atom=geometry, basis=basis_set).RKS(xc=xc_func).run()
    return mf.e_tot, mf.mo_energy

class DFTFlightServer(flight.FlightServerBase):
    def do_action(self, context, action):
        if action.type == 'run_dft':
            # Decode inputs from the action body
            params = json.loads(action.body.to_pybytes())

            energy, orbital_energies = run_dft(
                params['geometry'], params['basis_set'], params['xc_func'])

            yield flight.Result(pickle.dumps(energy))
            yield flight.Result(pickle.dumps((orbital_energies.shape,
                                              orbital_energies.dtype)))
            yield flight.Result(orbital_energies.data)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    server = DFTFlightServer('grpc://localhost:50011')
    server.serve()
