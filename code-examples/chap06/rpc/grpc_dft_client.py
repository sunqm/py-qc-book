import grpc
import dft_rpc_pb2
import dft_rpc_pb2_grpc

def run_dft(geometry, basis_set, xc_func):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = dft_rpc_pb2_grpc.DFTCalculationStub(channel)
        response = stub.run_dft(dft_rpc_pb2.DFTJobRequest(
            geometry=geometry,
            basis_set=basis_set,
            xc_func=xc_func
        ))
        return response.energy

if __name__ == '__main__':
    e = run_dft('O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0', 'sto-3g', 'b3lyp')
    print("DFT job results:", e)
