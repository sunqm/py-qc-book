from collections import deque
from typing import Union
from functools import lru_cache
import json
import numpy as np
import h5py
import scipy.linalg

class SimpleDIIS:
    def __init__(self, max_space=8):
        self.errvecs = deque(maxlen=max_space)
        self.tvecs = deque(maxlen=max_space)

    def update(self, errvec, tvec):
        self.errvecs.append(_HashableVector(errvec.ravel()))
        self.tvecs.append(_HashableVector(tvec))
        return extrapolate(self.errvecs, self.tvecs)

class DIIS:
    def __init__(self, filename, max_space=8):
        self.filename = filename
        # keys records the active dataset names in HDF5 storage
        self.keys = deque(maxlen=max_space)
        # head points to the next available dataset Id
        self.head = 0

    @property
    def max_space(self):
        return self.keys.maxlen

    def update(self, errvec, tvec):
        with h5py.File(self.filename, mode='a') as f:
            head, self.head = self.head, (self.head + 1) % self.max_space
            self.keys.append(head)

            if f'e{head}' in f:
                # Reuse existing datasets
                f[f'e{head}'][:] = errvec.ravel()
                f[f't{head}'][:] = tvec
            else:
                f[f'e{head}'] = errvec.ravel()
                f[f't{head}'] = tvec
            if 'metadata' in f:
                del f['metadata']
            f['metadata'] = self.dumps()
            f.flush()

            errvecs = [_HashableVector(f[f'e{key}']) for key in self.keys]
            tvecs = [_HashableVector(f[f't{key}']) for key in self.keys]
            return extrapolate(errvecs, tvecs)

    def dumps(self):
        '''Only JSON-serializes a few necessary attributes'''
        return json.dumps({
            'max_space': self.max_space,
            'keys': list(self.keys),
            'head': self.head,
        })

    def save(self):
        with h5py.File(self.filename, mode='a') as f:
            f['metadata'] = self.dumps()

    @classmethod
    def restore(cls, filename):
        with h5py.File(filename, mode='r') as f:
            attrs = json.loads(f['metadata'][()])
        obj = cls(filename)
        obj.keys = deque(attrs['keys'], maxlen=attrs['max_space'])
        obj.head = attrs['head']
        return obj

def extrapolate(errvecs, tvecs):
    space = len(tvecs)
    B = np.zeros((space+1, space+1))
    B[-1,:-1] = B[:-1,-1] = 1.
    g = np.zeros(space+1)
    g[-1] = 1
    for i, e1 in enumerate(errvecs):
        for j, e2 in enumerate(errvecs):
            if j < i:
                continue
            B[i,j] = B[j,i] = e1.dot(e2)

    c = scipy.linalg.solve(B, g, assume_a='sym')[:-1]
    sol = tvecs[0] * c[0]
    for v, x in zip(tvecs[1:], c[1:]):
        sol += v * x
    return sol

class _HashableVector:
    def __init__(self, vec: Union[np.ndarray, h5py.Dataset]):
        self.data = vec

    @lru_cache
    def dot(self, other):
        assert self.data.ndim == other.data.ndim == 1
        return np.asarray(self.data).dot(np.asarray(other.data))

    def __mul__(self, val):
        return np.asarray(self.data) * val

    __rmul__ = __mul__
