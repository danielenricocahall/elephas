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


def accumulate_model_gradients_and_history(x, y):
    state_dict, history = x
    other_state_dict, other_history = y
    updated_state = add_params(state_dict, other_state_dict)
    combined_history = {k: v + other_history[k] for k, v in history.items()}
    return updated_state, combined_history
