import math
from collections.abc import MutableMapping
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from . import types


def map_tensor(input_, func):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, Sequence):
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
    trees: list[types.Tree],
    check: bool = False,
) -> types.Tree:
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
    input_: types.Tree,
    func: Callable[[types.Key, types.Value], types.Value],
    sep: str | None = ".",
    unflatten: bool = False,
) -> types.Tree:
    """Apply a function to each item in a flattened dictionary."""
    flat_dict = flatten_dict(input_, sep=sep)
    out = {k: func(k, v) for k, v in flat_dict.items()}
    if unflatten:
        out = unflatten_dict(out, sep=sep)
    return out


def to_sequence(map):
    return map.flatten(-2).transpose(-1, -2)


def to_map(sequence):
    n = sequence.shape[-2]
    e = math.isqrt(n)
    assert e * e == n
    assert e * e == n
    sequence.transpose(-1, -2).unflatten(-1, [e, e])


def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: str = "zeros",  # zeros, ones, random, random_c
    bounds: tuple[int] = (None, None),
):
    shape = list(x.shape)
    d = x.shape[pad_dim]
    assert d <= length
    if d == length:
        return x
    shape[pad_dim] = length - d

    low, high = bounds

    if mode == "zeros":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode == "ones":
        xn = torch.ones(*shape, device=x.device, dtype=x.dtype)
    elif mode == "random":
        low = low if low is not None else x.min()
        high = high if high is not None else x.max()
        xn = torch.empty(*shape, device=x.device).uniform_(low, high)
    elif mode == "random_c":
        low, high = bounds  # we use the bounds as fallback for empty seq.
        xn = torch.cat(
            [
                torch.empty(*shape[:-1], 1, device=x.device).uniform_(
                    x[..., i].min() if d > 0 else low,
                    x[..., i].max() if d > 0 else high,
                )
                for i in range(shape[-1])
            ],
            dim=-1,
        )
    else:
        raise ValueError(mode)
    return torch.cat([x, xn], dim=pad_dim)


def pad_and_stack(
    sequences: Sequence[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if length is None:
        length = max([x.shape[pad_dim] for x in sequences])

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y


def extract_patches(
    tensor: torch.Tensor,
    required_corners: torch.Tensor,
    ps: int,
) -> torch.Tensor:
    c, h, w = tensor.shape
    corner = required_corners.long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 0, 1), corner.float()


def batch_extract_patches(tensor: torch.Tensor, kpts: torch.Tensor, ps: int):
    b, c, h, w = tensor.shape
    b, n, _ = kpts.shape
    out = torch.zeros((b, n, c, ps, ps), dtype=tensor.dtype, device=tensor.device)
    corners = torch.zeros((b, n, 2), dtype=tensor.dtype, device=tensor.device)
    for i in range(b):
        out[i], corners[i] = extract_patches(tensor[i], kpts[i] - ps / 2 - 1, ps)
    return out, corners


def draw_image_patches(img, patches, corners):
    b, c, h, w = img.shape
    b, n, c, p, p = patches.shape
    b, n, _ = corners.shape
    for i in range(b):
        for k in range(n):
            y, x = corners[i, k]
            img[i, :, x : x + p, y : y + p] = patches[i, k]


def build_heatmap(img, patches, corners):
    hmap = torch.zeros_like(img)
    draw_image_patches(hmap, patches, corners.long())
    hmap = hmap.squeeze(1)
    return hmap, (hmap > 0.0).float()  # bxhxw


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points, eps=0.0):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
        eps: Epsilon value to prevent zero division.
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)


def batched_eye_like(x: torch.Tensor, n: int):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)


def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3)."""
    z = torch.zeros_like(v[..., 0])
    M = torch.stack(
        [
            z,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            z,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            z,
        ],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))
    return M


def transform_points(T, points):
    return from_homogeneous(to_homogeneous(points) @ T.transpose(-1, -2))


def is_inside(pts, shape):
    return (pts > 0).all(-1) & (pts < shape[:, None]).all(-1)


def so3exp_map(w, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


@torch.jit.script
def distort_points(pts, dist):
    """Distort normalized 2D coordinates
    and check for validity of the distortion model.
    """
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2**2
        undist = undist + pts * radial

        # The distortion model is supposedly only valid within the image
        # boundaries. Because of the negative radial distortion, points that
        # are far outside of the boundaries might actually be mapped back
        # within the image. To account for this, we discard points that are
        # beyond the inflection point of the distortion model,
        # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
        limited = ((k2 > 0) & ((9 * k1**2 - 20 * k2) > 0)) | ((k2 <= 0) & (k1 > 0))
        limit = torch.abs(
            torch.where(
                k2 > 0,
                (torch.sqrt(9 * k1**2 - 20 * k2) - 3 * k1) / (10 * k2),
                1 / (3 * k1),
            )
        )
        valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2 * p12 * uv + p21 * (r2 + 2 * pts**2)
            # TODO: handle tangential boundaries

    return undist, valid


@torch.jit.script
def J_distort_points(pts, dist):
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]

    J_diag = torch.ones_like(pts)
    J_cross = torch.zeros_like(pts)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        uv = torch.prod(pts, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2**2
        d_radial = 2 * k1 + 4 * k2 * r2
        J_diag += radial + (pts**2) * d_radial
        J_cross += uv * d_radial

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            J_diag += 2 * p12 * pts.flip(-1) + 6 * p21 * pts
            J_cross += 2 * p12 * pts + 2 * p21 * pts.flip(-1)

    J = torch.diag_embed(J_diag) + torch.diag_embed(J_cross).flip(-1)
    return J


def get_image_coords(img):
    h, w = img.shape[-2:]
    return (
        torch.stack(
            torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=img.device),
                torch.arange(w, dtype=torch.float32, device=img.device),
                indexing="ij",
            )[::-1],
            dim=0,
        ).permute(1, 2, 0)
    )[None] + 0.5
