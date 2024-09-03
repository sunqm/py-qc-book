import os
import sys
import time
import io
import socket
import tempfile
import pickle
from contextlib import contextmanager
from multiprocessing import Process, Event
from multiprocessing.shared_memory import SharedMemory
import numpy as np

def timing(f):
    def timing_f(*args):
        t0 = time.perf_counter()
        f(*args)
        t1 = time.perf_counter()
        return t1 - t0
    return timing_f

def communicate(server, client, channel, args):
    if isinstance(channel, tuple):
        r, w = channel
    else:
        r = w = channel
    proc = Process(target=client, args=(r, *args))
    proc.start()
    server(w, *args)
    proc.join()

class Guard:
    def __init__(self):
        self.ready_for_write = Event()
        self.ready_for_write.set()
        self.ready_for_read = Event()
        self.ready_for_read.clear()

    @contextmanager
    def for_send(self):
        self.ready_for_write.wait()
        yield
        self.ready_for_write.clear()
        self.ready_for_read.set()

    @contextmanager
    def for_recv(self):
        self.ready_for_read.wait()
        yield
        self.ready_for_read.clear()
        self.ready_for_write.set()

#
# Transfer np array using anonymous pipes
#
def pipe_recv(fd, size, count):
    for _ in range(count):
        buf = io.BytesIO()
        remaining = size
        while remaining > 0:
            dat = os.read(fd, remaining)
            buf.write(dat)
            remaining -= len(dat)
        buf = np.ndarray(shape=(size,), dtype='c', buffer=buf.getvalue())

def pipe_send(fd, size, count):
    a = np.ones(size, dtype='c')
    buf = a.data
    for _ in range(count):
        start = 0
        while start < size:
            m = os.write(fd, buf[start:])
            start += m

@timing
def pipe_transfer(size, count):
    r, w = os.pipe()
    communicate(pipe_send, pipe_recv, (r, w), (size, count))
    os.close(r)
    os.close(w)

#
# Transfer np array using named pipes
#
def namedpipe_recv(pipe_file, size, count):
    fd = os.open(pipe_file, os.O_RDONLY)
    pipe_recv(fd, size, count)

def namedpipe_send(pipe_file, size, count):
    fd = os.open(pipe_file, os.O_WRONLY)
    pipe_send(fd, size, count)

@timing
def namedpipe_transfer(size, count):
    pipe_file = tempfile.mktemp()
    os.mkfifo(pipe_file)
    communicate(namedpipe_send, namedpipe_recv, pipe_file, (size, count))
    os.remove(pipe_file)

#
# Serialize np array and transfer data using anonymous pipes.
# Mimic the operation of the queue in ProcessPoolExecutor.
#
def pipe_pickle_recv(fd, size, count):
    for _ in range(count):
        buf = io.BytesIO()
        remaining = int.from_bytes(os.read(fd, 8), sys.byteorder)
        while remaining > 0:
            dat = os.read(fd, remaining)
            buf.write(dat)
            remaining -= len(dat)
        pickle.loads(buf.getvalue())

def pipe_pickle_send(fd, size, count):
    a = np.ones(size, dtype='c')
    for _ in range(count):
        buf = pickle.dumps(a)
        os.write(fd, len(buf).to_bytes(8, sys.byteorder))
        start = 0
        while start < size:
            m = os.write(fd, buf[start:])
            start += m

@timing
def pipe_pickle_transfer(size, count):
    r, w = os.pipe()
    communicate(pipe_pickle_send, pipe_pickle_recv, (r, w), (size, count))
    os.close(r)
    os.close(w)

#
# Transfer np array using Unix domain sockets
#
def socket_recv(sock, size, count):
    for _ in range(count):
        buf = io.BytesIO()
        remaining = size
        while remaining > 0:
            dat = sock.recv(remaining)
            buf.write(dat)
            remaining -= len(dat)
        buf = np.ndarray(shape=(size,), dtype='c', buffer=buf.getvalue())

def socket_send(sock, size, count):
    a = np.ones(size, dtype='c')
    buf = a.data
    for _ in range(count):
        start = 0
        while start < size:
            m = sock.send(buf[start:])
            start += m

@timing
def socket_transfer(size, count):
    r, w = socket.socketpair()
    communicate(socket_send, socket_recv, (r, w), (size, count))
    r.close()
    w.close()

#
# Serialize np array and transfer data using Unix domain sockets
#
def socket_pickle_recv(sock, size, count):
    for _ in range(count):
        buf = io.BytesIO()
        remaining = int.from_bytes(sock.recv(8), sys.byteorder)
        while remaining > 0:
            dat = sock.recv(remaining)
            buf.write(dat)
            remaining -= len(dat)
        pickle.loads(buf.getvalue())

def socket_pickle_send(sock, size, count):
    a = np.ones(size, dtype='c')
    for _ in range(count):
        buf = pickle.dumps(a)
        sock.send(len(buf).to_bytes(8, sys.byteorder))
        start = 0
        while start < size:
            m = sock.send(buf[start:])
            start += m

@timing
def socket_pickle_transfer(size, count):
    r, w = socket.socketpair()
    communicate(socket_pickle_send, socket_pickle_recv, (r, w), (size, count))
    r.close()
    w.close()

#
# Transfer np array using shared memory.
# This method is only available with multiprocessing.Process
#
def shm_recv(shm, size, count, guard):
    for _ in range(count):
        with guard.for_recv():
            buf = np.ndarray(shape=(size,), dtype='c', buffer=shm.buf)

def shm_send(shm, size, count, guard):
    data = np.ones(shape=(size,), dtype='c')
    for _ in range(count):
        with guard.for_send():
            buf = np.ndarray(shape=(size,), dtype='c', buffer=shm.buf)
            buf[:] = data

@timing
def shm_transfer(size, count):
    shm = SharedMemory(create=True, size=size)
    communicate(shm_send, shm_recv, shm, (size, count, Guard()))
    shm.unlink()

#
# Transfer np array using the shared memory file.
# This method can be used with ProcessPoolExecutor.
#
def shm_file_recv(shm_file, size, count, guard):
    for _ in range(count):
        with guard.for_recv():
            shm = SharedMemory(name=shm_file)
            buf = np.ndarray(shape=(size,), dtype='c', buffer=shm.buf)

def shm_file_send(shm_file, size, count, guard):
    data = np.ones(shape=(size,), dtype='c')
    for _ in range(count):
        with guard.for_send():
            shm = SharedMemory(name=shm_file)
            buf = np.ndarray(shape=(size,), dtype='c', buffer=shm.buf)
            buf[:] = data

@timing
def shm_file_transfer(size, count):
    shm = SharedMemory(create=True, size=size)
    communicate(shm_file_send, shm_file_recv, shm.name, (size, count, Guard()))
    shm.unlink()

#
# Transfer np array using np.memmap (on disk).
# This method is only available with multiprocessing.Process
#
def np_memmap_recv(buf, size, count, guard):
    for _ in range(count):
        with guard.for_recv():
            pass

def np_memmap_send(buf, size, count, guard):
    data = np.ones(shape=(size,), dtype='c')
    for _ in range(count):
        with guard.for_send():
            buf[:] = data

@timing
def np_memmap_transfer(size, count):
    mmap_file = tempfile.mktemp()
    buf = np.memmap(mmap_file, shape=(size,), dtype='c', mode='w+')
    communicate(np_memmap_send, np_memmap_recv, buf, (size, count, Guard()))
    os.remove(mmap_file)

#
# Transfer np array using memory mapped files (on disk).
# This method can be used with ProcessPoolExecutor.
#
def mmap_recv(mmap_file, size, count, guard):
    for _ in range(count):
        with guard.for_recv():
            buf = np.memmap(mmap_file, shape=(size,), dtype='c', mode='r+')

def mmap_send(mmap_file, size, count, guard):
    data = np.ones(shape=(size,), dtype='c')
    for _ in range(count):
        with guard.for_send():
            buf = np.memmap(mmap_file, shape=(size,), dtype='c', mode='w+')
            buf[:] = data

@timing
def mmap_transfer(size, count):
    mmap_file = tempfile.mktemp()
    communicate(mmap_send, mmap_recv, mmap_file, (size, count, Guard()))
    os.remove(mmap_file)

#
# Transfer np array using memory mapped files (on shared memory).
# This method can be used with ProcessPoolExecutor.
#
def mmap_shm_recv(mmap_file, size, count, guard):
    for _ in range(count):
        with guard.for_recv():
            buf = np.memmap(mmap_file, shape=(size,), dtype='c', mode='r+')

def mmap_shm_send(mmap_file, size, count, guard):
    data = np.ones(shape=(size,), dtype='c')
    for _ in range(count):
        with guard.for_send():
            buf = np.memmap(mmap_file, shape=(size,), dtype='c', mode='w+')
            buf[:] = data

@timing
def mmap_shm_transfer(size, count):
    mmap_file = tempfile.mktemp(dir='/dev/shm')
    communicate(mmap_shm_send, mmap_shm_recv, mmap_file, (size, count, Guard()))
    os.remove(mmap_file)

if __name__ == '__main__':
    gvars = globals()
    keys = ('namedpipe', 'pipe', 'pipe_pickle', 'socket', 'socket_pickle',
            'shm', 'shm_file', 'np_memmap', 'mmap', 'mmap_shm')
    fns = {key: gvars[f'{key}_transfer'] for key in keys}
    print('  '.join(keys))
    for size, count in [(int(1e5), 10000),
                        (int(1e6), 1000),
                        (int(1e7), 100),
                        (int(1e8), 10),]:
        print(f'{size:.1e} bytes x {count}')
        ts = [fn(size, count) for fn in fns.values()]
        print('  '.join((f'{t:.5f}' for t in ts)))
