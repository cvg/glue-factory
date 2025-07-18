from collections.abc import MutableMapping
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import torch

Key: TypeAlias = str | tuple[str, ...]
Value: TypeAlias = Any
Tree: TypeAlias = dict[Key, Value]


def to_view(data, i):
    return {k + i: v for k, v in data.items()}


def get_view(data, i):
    data_g = {k: v for k, v in data.items() if not k[-1].isnumeric()}
    data_i = {k[:-1]: v for k, v in data.items() if k[-1] == i}
    return {**data_g, **data_i}


def get_twoview(data, idx):
    li = idx[0]
    ri = idx[-1]
    assert idx == f"{li}to{ri}"
    data_lr = {k[:-4] + "0to1": v for k, v in data.items() if k[-4:] == f"{li}to{ri}"}
    data_rl = {k[:-4] + "1to0": v for k, v in data.items() if k[-4:] == f"{ri}ito{li}"}
    data_l = {
        k[:-1] + "0": v for k, v in data.items() if k[-1:] == li and k[-3:-1] != "to"
    }
    data_r = {
        k[:-1] + "1": v for k, v in data.items() if k[-1:] == ri and k[-3:-1] != "to"
    }
    return {**data_lr, **data_rl, **data_l, **data_r}


def stack_twoviews(data, indices=["0to1", "0to2", "1to2"]):
    idx0 = indices[0]
    m_data = data[idx0] if idx0 in data else get_twoview(data, idx0)
    # stack on dim=0
    for idx in indices[1:]:
        data_i = data[idx] if idx in data else get_twoview(data, idx)
        for k, v in data_i.items():
            m_data[k] = torch.cat([m_data[k], v], dim=0)
    return m_data


def unstack_twoviews(data, B, indices=["0to1", "0to2", "1to2"]):
    out = {}
    for i, idx in enumerate(indices):
        out[idx] = {k: v[i * B : (i + 1) * B] for k, v in data.items()}
    return out


def concat_tree(
    trees: list[Tree],
    check: bool = False,
) -> Tree:
    """Concatenate a list of trees into a single batch"""
    if not trees:
        return {}
    keys = set(trees[0].keys())
    if check:
        for batch in trees[1:]:
            if keys != set(batch.keys()):
                raise ValueError("All trees must have the same keys.")

    def combine_recursive(val_list: Sequence[Any]) -> Any:
        if isinstance(val_list[0], torch.Tensor):
            return torch.cat(val_list)
        elif isinstance(val_list[0], Mapping):
            return concat_tree(val_list, check)
        elif isinstance(val_list[0], Sequence):
            return sum(val_list, start=[])

    return {k: combine_recursive([batch[k] for batch in trees]) for k in keys}


def flatten_dict(
    dictionary: Mapping[str, Any],
    parent_keys: tuple[str, ...] = (),
    sep: str | None = ".",
) -> dict[str | tuple[str, ...], Any]:
    items = []
    for key, value in dictionary.items():
        new_key = parent_keys + (key,)
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    flat_dict = dict(items)
    if len(parent_keys) == 0 and sep is not None:
        # Top-level
        return {sep.join(k): v for k, v in flat_dict.items()}
    else:
        return flat_dict


def unflatten_dict(
    flat_dict: Mapping[str | tuple[str, ...], Any],
    sep: str | None = ".",
) -> dict[str, Any]:
    unflattened = {}
    for key, value in flat_dict.items():
        if isinstance(key, tuple):
            parts = key
        elif sep is not None:
            parts = key.split(sep)
        else:
            parts = (key,)
        current = unflattened
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return unflattened


def flat_map(
    input_: Tree,
    func: Callable[[Key, Value], Value],
    sep: str | None = ".",
    unflatten: bool = False,
) -> Tree:
    """Apply a function to each item in a flattened dictionary."""
    flat_dict = flatten_dict(input_, sep=sep)
    out = {k: func(k, v) for k, v in flat_dict.items()}
    if unflatten:
        out = unflatten_dict(out, sep=sep)
    return out
