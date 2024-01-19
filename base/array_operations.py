import numpy as np
from scipy.special import softmax


def safe_softmax(x: np.ndarray, axis):
    # m = np.max(x, axis=axis, keepdims=True)
    # exp = np.exp(x - m)
    # return exp / exp.sum(axis=axis)
    return softmax(x, axis=axis)


def normalize(x: np.ndarray, axis, norm: int):
    raise NotImplementedError
    norm = np.linalg.norm(x, ord=norm, axis=axis, keepdims=True)
    return x / norm


def make_batch_indices(n_items: int, batch_size: int, shuffle: bool = False):

    n_batches = int(np.ceil(n_items / batch_size))
    batch_indices = []
    indices = np.arange(n_items)
    if shuffle:
        raise NotImplementedError
    for i in range(n_batches):
        batch_indices.append(
            indices[i * batch_size:min((i + 1) * batch_size, n_items)]
        )
    return batch_indices


def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p)  # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s
