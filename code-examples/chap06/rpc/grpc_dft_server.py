from concurrent import futures
import grpc
import dft_rpc_pb2
import dft_rpc_pb2_grpc

class DFTCalculationServicer(object):
    def run_dft(self, request, context):
        import pyscf
        mol = pyscf.M(atom=request.geometry, basis=request.basis_set)
        mf = mol.RKS(xc=request.xc_func).run()
        return dft_rpc_pb2.DFTJobResponse(energy=mf.e_tot)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dft_rpc_pb2_grpc.add_DFTCalculationServicer_to_server(DFTCalculationServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
