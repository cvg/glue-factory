import math
from collections.abc import MutableMapping
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.nn.functional as F

from . import types


def map_tensor(input_, func):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, np.ndarray):
        return func(torch.from_numpy(input_))
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


def pmap(
    func: Callable, iterable: Iterable[Any], num_processes: int | None = None
) -> Sequence[Any]:
    multi_pool = tmp.Pool(processes=num_processes)
    results = multi_pool.imap(func, iterable)
    multi_pool.close()
    multi_pool.join()
    multi_pool.terminate()
    return results


def grad_norm(params):
    return torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])


def param_norm(params):
    return torch.nn.utils.get_total_norm([p for p in params if p.requires_grad])


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def add_prefix(d: dict, prefix: str) -> dict:
    return {prefix + k: v for k, v in d.items()}


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


def iterelements(data: dict, pattern="view") -> Iterable[Any]:
    i = 0
    while True:
        view = data.get(f"{pattern}{i}", None)
        if view is None:
            break
        yield view
        i += 1


def pack_elements(data, pattern="view"):
    return pack_tree(iterelements(data, pattern=pattern))


def pack_tree(
    trees: Iterable[types.Tree],
    check: bool = False,
    fn: Callable[[Sequence[Any]], Any] = lambda x: x,
    sep: str | None = ".",
) -> types.Tree:
    """Concatenate a list of trees into a list per entry"""
    if not trees:
        return {}

    trees = list(trees)
    flat_trees = [flatten_dict(batch, sep=sep) for batch in trees]
    keys = set(flat_trees[0].keys())
    if check:
        for batch in trees[1:]:
            if keys != set(batch.keys()):
                raise ValueError("All trees must have the same keys.")
    joined_tree = {k: fn([batch[k] for batch in flat_trees]) for k in keys}
    return unflatten_dict(joined_tree, sep=sep)


def concat_tree(trees: Iterable[types.Tree], check: bool = False) -> types.Tree:
    """Concatenate a list of trees into a single batch"""

    def combine(val_list: Sequence[Any]) -> Any:
        if isinstance(val_list[0], torch.Tensor):
            return torch.cat(val_list)
        elif isinstance(val_list[0], Sequence):
            return sum(val_list, start=[])
        else:
            raise TypeError(f"Cannot combine values of type {type(val_list[0])}")

    return pack_tree(trees, check=check, fn=combine)


def compare_tree(
    tree_i: types.Tree,
    tree_j: types.Tree,
    compare_fn: Callable[[Any, Any], bool | None] | None = None,
) -> types.Tree:
    if compare_fn is None:

        def compare_fn(el1, el2):
            if isinstance(el1, torch.Tensor):
                if el1.dtype in [torch.float16, torch.float32, torch.float64]:
                    return torch.all(torch.abs(el1 - el2) < 1e-2).item()
                return torch.all(el1 == el2).item()
            if isinstance(el1, np.ndarray):
                if np.issubdtype(el1.dtype, np.floating):
                    return np.all(np.abs(el1 - el2) < 1e-2)
                return np.array_equal(el1, el2)
            elif isinstance(el1, (int, float, str, bool)):
                return el1 == el2
            elif isinstance(el1, Iterable):
                return all(compare_fn(e1, e2) for e1, e2 in zip(el1, el2))
            else:
                return None

    is_equal = pack_tree([tree_i, tree_j], fn=lambda x: compare_fn(x[0], x[1]))
    flat_is_equal = flatten_dict(is_equal)
    flat_is_equal = {k: v for k, v in flat_is_equal.items() if v is not None}
    return flat_is_equal


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


def set_slice(
    src: torch.Tensor, val: float | torch.Tensor, dim: int = 0, start: int = 0
) -> torch.Tensor:
    """Set a slice to a constant value. Useful for border removal in padded images."""
    # Avoids graph breaks, equivalent to: src[..., start:] = val
    n = src.shape[dim]
    indices_in_dim = torch.arange(n, device=src.device)
    # Fix broadcast dimensions
    indices_in_dim = indices_in_dim.view(n, *[1] * (src.dim() - 1)).transpose(0, dim)
    return torch.where(indices_in_dim < start, src, val)


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


def grid_sample(
    image: torch.Tensor,
    coords: torch.Tensor,
    interpolation: str = "bilinear",
    align_corners: bool = False,
):
    assert image.dim() == coords.dim()
    is_batched = image.dim() == 4

    if is_batched:
        assert coords.dim() == 4
        return F.grid_sample(
            image.to(coords.device), coords, interpolation, align_corners=False
        )
    else:
        return F.grid_sample(
            image[None].to(coords.device),
            coords[None],
            mode=interpolation,
            align_corners=align_corners,
        )[0]


def get_pixel_grid(
    *,
    fmap: torch.Tensor | None = None,  # B x H X W X D
    camera: Any | None = None,
    size: Optional[tuple[int, int]] = None,
    device: torch.device | None = None,
    dtype=torch.float32,
    normalized: bool = False,
) -> torch.Tensor:
    if fmap is None:
        if camera is None:
            if size is None:
                raise ValueError("Specify fmap, size, or camera")
            w, h = size
        else:
            w, h = camera.size.int()
            device = camera.device
            dtype = camera.dtype
    else:
        *_, h, w, _ = fmap.shape
        device = fmap.device
        dtype = fmap.dtype
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(w, dtype=dtype, device=device),
            torch.arange(h, dtype=dtype, device=device),
            indexing="xy",
        ),
        dim=-1,
    )
    if fmap.ndim == 4:
        b = fmap.shape[0]
        grid = grid[None].expand(b, -1, -1, -1)
    grid += 0.5
    if normalized:
        grid *= 2 / grid.new_tensor([w, h])
        grid -= 1
    elif fmap is not None and camera is not None:
        # In case fmaps are at a different image resolution
        grid *= camera.size / grid.new_tensor([w, h])
    return grid


def chw_from_hwc(coords):
    # ...HWC -> ...CHW
    return coords.transpose(-2, -1).transpose(-3, -2)


def hwc_from_chw(image):
    # ...CHW -> ...HWC
    return image.transpose(-3, -2).transpose(-2, -1)


def denormalize_coords(coords, hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Denormalize coordinates from [-1, 1] to [0, H] or [0, W] (COLMAP)"""
    coords = coords.clone()
    if hw is None:
        hw = coords.shape[-3:-1]
    coords[..., 0] = (coords[..., 0] + 1) / 2 * (hw[1] - 1)
    coords[..., 1] = (coords[..., 1] + 1) / 2 * (hw[0] - 1)
    return coords


def normalize_coords(coords, hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Normalize coordinates from [0, H] or [0, W] (COLMAP) to [-1, 1]"""
    coords = coords.clone()
    if hw is None:
        hw = coords.shape[-3:-1]
    coords[..., 0] = coords[..., 0] / (hw[1] - 1) * 2 - 1
    coords[..., 1] = coords[..., 1] / (hw[0] - 1) * 2 - 1
    return coords


def cycle_dist(
    q_to_ref: torch.Tensor, ref_to_q: torch.Tensor, normalized: bool = False
) -> torch.Tensor:
    """Compute cycle consistency error between two coordinate fields."""
    q_to_ref_to_q = hwc_from_chw(grid_sample(chw_from_hwc(ref_to_q), q_to_ref))

    return torch.linalg.norm(
        get_pixel_grid(fmap=q_to_ref, normalized=normalized)
        - (q_to_ref_to_q if normalized else denormalize_coords(q_to_ref_to_q)),
        dim=-1,
    )


def interpolate_matches(
    kpts_q: torch.Tensor,  # B x N X 2
    kpts_t: torch.Tensor,  # B x M X 2
    warp: torch.Tensor,  # B x H x W x 2
    cert: torch.Tensor,  # B x H x W x 1
    q_hw: tuple[int, int] | None = None,  # To normalize in range ([-1, 1])
    t_hw: tuple[int, int] | None = None,
    mutual_check: bool = True,
    max_kp_error: float = 3.0,  # pixels
    filter_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Normalize to [-1, 1] for grid sampling
    kpts_q = normalize_coords(kpts_q, q_hw)
    kpts_q_to_t = grid_sample(warp.permute(0, 3, 1, 2), kpts_q[:, None])[
        :, :, 0
    ].permute(0, 2, 1)
    scores = grid_sample(cert[:, None], kpts_q[:, None])[:, 0, 0]
    # Corresponding coordinates in the other image (target), COLMAP coords.
    kpts_q_to_t = denormalize_coords(kpts_q_to_t, t_hw)
    # Output points are again in COLMAP coordinates
    dist = torch.cdist(kpts_q_to_t, kpts_t)  # in pixels
    matches = torch.min(dist, dim=-1)
    matches, match_dist = matches.indices, matches.values
    valid = torch.isfinite(match_dist) & (match_dist < max_kp_error)
    if mutual_check:
        indicesq = torch.arange(matches.shape[-1], device=kpts_q.device)[None]
        mutual = indicesq == torch.min(dist, dim=-2).indices.gather(1, matches)
        valid = valid & mutual
    valid = valid & (scores > filter_threshold)
    return torch.where(valid, matches, -1), torch.where(valid, scores, 0)


def match_keypoints_dense(
    pred: dict,  # Containts warp and certainty tensors
    data: dict,  # Contains keypoints and images
    max_kp_error: float,
    filter_threshold: float,
    mutual_check: bool = True,
) -> dict:
    """Match keypoints using dense correspondences."""
    kpts0 = data["keypoints0"]  # COLMAP coordinates
    kpts1 = data["keypoints1"]  # COLMAP coordinates

    img0 = data["view0"]["image"]
    img1 = data["view1"]["image"]

    mpred = {}
    mpred["matches0"], mpred["matching_scores0"] = interpolate_matches(
        kpts0,
        kpts1,
        pred["warp0"],
        pred["certainty0"],
        img0.shape[-2:],
        img1.shape[-2:],
        max_kp_error=max_kp_error,
        mutual_check=mutual_check,
        filter_threshold=filter_threshold,
    )
    mpred["matches1"], mpred["matching_scores1"] = interpolate_matches(
        kpts1,
        kpts0,
        pred["warp1"],
        pred["certainty1"],
        img1.shape[-2:],
        img0.shape[-2:],
        max_kp_error=max_kp_error,
        mutual_check=mutual_check,
        filter_threshold=filter_threshold,
    )

    # Pipe the keypoints again
    mpred["keypoints0"] = data["keypoints0"]
    mpred["keypoints1"] = data["keypoints1"]
    return mpred
