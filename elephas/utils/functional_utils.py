from typing import List

import numpy as np


def add_params(
    param_list_left: List[np.array], param_list_right: List[np.array]
) -> List[np.array]:
    """Add two lists of parameters one by one

    :param param_list_left: list of numpy arrays
    :param param_list_right: list of numpy arrays
    :return: list of numpy arrays
    """
    return [x + y for x, y in zip(param_list_left, param_list_right)]


def subtract_params(
    param_list_left: List[np.array], param_list_right: List[np.array]
) -> List[np.array]:
    """Subtract two lists of parameters

    :param param_list_left: list of numpy arrays
    :param param_list_right: list of numpy arrays
    :return: list of numpy arrays
    """
    return [x - y for x, y in zip(param_list_left, param_list_right)]


def get_neutral(array_list: List[np.array]) -> List[np.array]:
    """Get list of zero-valued numpy arrays for
    specified list of numpy arrays

    :param array_list: list of numpy arrays
    :return: list of zeros of same shape as input
    """
    return [np.zeros_like(x) for x in array_list]


def divide_by(array_list: List[np.array], num_workers: int) -> List[np.array]:
    """Divide a list of parameters by an integer num_workers.

    :param array_list:
    :param num_workers:
    :return:
    """
    return [x / num_workers for x in array_list]


def average_and_subtract(
    base: List[np.array], deltas: List[List[np.array]]
) -> List[np.array]:
    """Return ``[base[i] - mean(d[i] for d in deltas)]`` elementwise.

    Equivalent to repeatedly applying ``subtract_params(base, divide_by(d, N))``
    once per delta, but with a single accumulator and one final scaled subtract.
    Saves ``2N`` model-sized allocations on the driver, which dominates the
    per-epoch aggregation cost for large models / many workers.
    """
    n = len(deltas)
    if n == 0:
        return [np.array(b, copy=True) for b in base]
    deltas_iter = iter(deltas)
    total = [d.copy() for d in next(deltas_iter)]
    for delta in deltas_iter:
        for t, d in zip(total, delta):
            t += d
    inv_n = 1.0 / n
    return [b - t * inv_n for b, t in zip(base, total)]
