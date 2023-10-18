"""
Author: Paul-Edouard Sarlin (skydes)
"""

import collections.abc as collections

import numpy as np
import torch

string_classes = (str, bytes)


def map_tensor(input_, func):
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif input_ is None:
        return None
    else:
        return func(input_)


def batch_to_numpy(batch):
    return map_tensor(batch, lambda tensor: tensor.cpu().numpy())


def batch_to_device(batch, device, non_blocking=True):
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking)

    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def index_batch(tensor_dict):
    batch_size = len(next(iter(tensor_dict.values())))
    for i in range(batch_size):
        yield map_tensor(tensor_dict, lambda t: t[i])
