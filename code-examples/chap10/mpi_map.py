import itertools
from mpi4py.MPI import COMM_WORLD as comm

rank = comm.Get_rank()
size = comm.Get_size()
workers = range(1, size)
TAG = 98215

if rank == 0:
    import atexit
    from time import sleep
    def wait_all(requests, timeout=None):
        if timeout is None:
            return [req.wait() for req in requests]

        if not all(req.test() for req in requests):
            elapsed, interval = 0, 0.1
            while elapsed < timeout:
                sleep(interval)
                if all(req.test() for req in requests):
                    break
                elapsed += interval
            else:
                print('Response from receivers timeout')
                comm.Abort()
                raise TimeoutError
        return [req.wait() for req in requests]

    def shutdown():
        reqs = [comm.isend(('Terminate', None), i, tag=TAG) for i in workers]
        wait_all(reqs, timeout=0.5)
    atexit.register(shutdown)

def _build_batches(iterable):
    iterable = iter(iterable)
    while True:
        batch = list(itertools.islice(iterable, size))
        if not batch:
            return
        yield batch

def map(func, *iterables):
    assert rank == 0
    for i in workers:
        comm.send(('Func', func), dest=i, tag=TAG)
    results = []
    for args_list in _build_batches(zip(*iterables)):
        reqs = [comm.isend(('Args', args), i, tag=TAG)
                for args, i in zip(args_list[1:], workers)]
        reqs = wait_all(reqs, timeout=0.5)
        results.append(func(*args_list[0]))
        for i in workers[:len(reqs)]:
            results.append(comm.recv(source=i, tag=TAG))
    return results

def wait():
    assert rank != 0
    func = None
    while True:
        label, args = comm.recv(tag=TAG)
        match label:
            case 'Terminate': return
            case 'Func': func = args
            case 'Args':
                result = func(*args)
                comm.send(result, dest=0, tag=TAG)

if __name__ == '__main__':
    wait()
