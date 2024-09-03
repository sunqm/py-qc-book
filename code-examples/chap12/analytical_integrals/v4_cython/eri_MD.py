import numpy as np
from basis import CGTO, n_cart
from _tensors import primitive_ERI_MD, _contracted_ERI


def contracted_ERI(bas_i, bas_j, bas_k, bas_l) -> np.ndarray:
    return _contracted_ERI(bas_i, bas_j, bas_k, bas_l, primitive_ERI_MD)
