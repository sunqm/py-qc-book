import ctypes
from functools import lru_cache
from typing import List, Dict
from dataclasses import dataclass
import yaml
import numpy as np
import math
import numba
import basis_set_exchange as bse

BOHR = 0.5291772 # Angstrom

@numba.njit(cache=True)
def gaussian_int(n, alpha):
    r'''int_0^inf x^n exp(-alpha x^2) dx'''
    assert n >= 0
    n1 = (n + 1) * .5
    return math.gamma(n1) / (2. * alpha**n1)

@numba.njit('double[::1](int64, double[::1])', cache=True)
def gto_norm(l, expnt):
    '''Radial part normalization'''
    assert l >= 0
    # Radial part normalization
    norm = (gaussian_int(l*2+2, 2*expnt)) ** -.5
    # Racah normalization, assuming angular part is normalized to unity
    norm *= ((2*l+1)/(4*np.pi))**.5
    return norm

@numba.njit('int64(int64)', inline='always', cache=True)
def n_cart(l: int):
    return (l + 1) * (l + 2) // 2

def gto_offsets(gtos):
    '''Offsets are the position of the first GTO function for each GTO shell
    inside all GTO basis functions. The last element of the offsets is the
    total number of basis functions.
    '''
    dims = [n_cart(b.angular_momentum) for b in gtos]
    return np.append(0, np.cumsum(dims))

@dataclass
class CGTO:
    angular_momentum: int
    exponents: np.ndarray
    coefficients: np.ndarray
    coordinates: np.ndarray

    _norm_coef = None

    @property
    def norm_coefficients(self):
        '''contraction coefficients with normalization factors'''
        if self._norm_coef is None:
            norm = gto_norm(self.angular_momentum, self.exponents)
            self._norm_coef = norm * self.coefficients
        return self._norm_coef

    @classmethod
    def from_dict(cls, dic: dict):
        angular_momentum = int(dic['angular_momentum'])
        exponents = np.asarray(dic['exponents'], dtype=float)
        coefficients = np.asarray(dic['coefficients'], dtype=float)
        coordinates = np.asarray(dic['coordinates'], dtype=float)
        assert angular_momentum >= 0
        assert all(exponents > 0)
        assert exponents.ndim == 1
        assert coefficients.ndim == 1
        assert coordinates.shape == (3,)
        return cls(angular_momentum, exponents, coefficients, coordinates)

    @classmethod
    def from_yaml(cls, conf: str):
        '''Creates a list of CGTOs from yaml configuration'''
        conf = yaml.load(conf, Loader=yaml.CSafeLoader)
        return [cls.from_dict(x) for x in conf]

    @classmethod
    def from_bse(cls, name: str, element: str, coordinates: np.ndarray):
        '''Creates a list of CGTOs from BSE database'''
        return [cls.from_dict({'coordinates': coordinates, **basis})
                for basis in bse_basis(name, element)]

    def dumps(self) -> bytearray:
        '''Serializes the CGTO object'''
        norm_coef = self.norm_coefficients
        bas = _CGTO(angular_momentum=self.angular_momentum,
                    n_primitive=self.exponents.size,
                    coordinates=tuple(self.coordinates),
                    exponents=self.exponents.ctypes.data,
                    coefficients=norm_coef.ctypes.data)
        return bytearray(bas)

    def to_jitCGTO(self):
        return jitCGTO(angular_momentum=self.angular_momentum,
                       exponents=self.exponents,
                       coefficients=self.coefficients,
                       coordinates=self.coordinates)

@numba.experimental.jitclass([
    ('angular_momentum', numba.int64),
    ('exponents', numba.float64[::1]),
    ('coefficients', numba.float64[::1]),
    ('coordinates', numba.float64[::1]),
    ('norm_coefficients', numba.float64[::1]),
])
class jitCGTO:
    def __init__(self,
                 angular_momentum: int,
                 exponents: np.ndarray,
                 coefficients: np.ndarray,
                 coordinates: np.ndarray):
        self.angular_momentum = angular_momentum
        self.exponents = exponents
        self.coefficients = coefficients
        self.coordinates = coordinates
        self.norm_coefficients = gto_norm(angular_momentum, exponents) * coefficients

@lru_cache(1000)
def bse_basis(name: str, element: str) -> List[dict]:
    '''Reads and converts BSE GTO basis'''
    data = bse.get_basis(name, elements=[element])
    basis = []
    for elem_basis in data['elements'].values():
        for raw_basis in elem_basis['electron_shells']:
            exps = np.array(raw_basis['exponents']).astype(float)
            coefs = np.array(raw_basis['coefficients']).astype(float)
            ls = raw_basis['angular_momentum']
            if len(ls) == len(coefs): # SP basis
                for l, c in zip(ls, coefs):
                    basis.append({
                        'angular_momentum': int(l),
                        'exponents': exps,
                        'coefficients': c,
                    })
            else:
                for c in coefs:
                    basis.append({
                        'angular_momentum': int(ls[0]),
                        'exponents': exps,
                        'coefficients': c,
                    })
    return basis

@dataclass
class Molecule:
    elements: List[str]
    coordinates: np.ndarray

    @classmethod
    def from_xyz(cls, xyz: str):
        elements = []
        coordinates = []
        for line in xyz.splitlines():
            line = line.strip()
            if not line or line[0] == '#':
                continue
            elements.append(line.split()[0])
            coordinates.append(line.split()[1:4])
        coordinates = np.array(coordinates).astype(float) / BOHR
        assert coordinates.shape[1] == 3
        return cls(elements=elements, coordinates=coordinates)

    def assign_basis(self, basis_set: Dict[str, str]) -> List[CGTO]:
        '''Creates a list of CGTOs for the molecule in xyz geomoetry (Angstrom unit)
        '''
        gtos = []
        for elem, r in zip(self.elements, self.coordinates):
            basis_name = basis_set[elem]
            assert isinstance(basis_name, str)
            gtos.extend(CGTO.from_bse(basis_name, elem, r))
        return gtos
