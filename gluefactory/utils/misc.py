import functools
import math
import pprint
from ast import arg
from collections.abc import MutableMapping
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.nn.functional as F
import torchvision.transforms.functional as tvf

from . import tensor, types

# ----------------------------------------------------------------------------
# Wrappers
# ----------------------------------------------------------------------------


# Hacky workaround for torch.amp.custom_fwd to support older versions of PyTorch.
AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


def filter_batch_for_jit(
    fn: Callable[[Any], Any], exclude_cls: Sequence[Any] = (list, tuple, str, bytes)
) -> Callable[[Any], Any]:
    # Remove non-tensor entries from a batch for JIT compatibility
    def is_valid(key: Any, arg: Any) -> bool:
        return not isinstance(arg, exclude_cls)

    @functools.wraps(fn)
    def wrapper(*args):
        return fn(
            *(
                filter_tree(arg, is_valid) if isinstance(arg, (dict, Mapping)) else arg
                for arg in args
            )
        )

    return wrapper


def map_tensor(input_, func):
    scalar_classes = (str, bytes, bool, float, int)
    if isinstance(input_, scalar_classes):
        return input_
    elif isinstance(input_, Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, Sequence):
        return input_.__class__([map_tensor(sample, func) for sample in input_])
    elif isinstance(input_, np.ndarray):
        return func(torch.from_numpy(input_))
    elif input_ is None:
        return None
    else:
        return func(input_)


def batch_to_numpy(batch):
    return map_tensor(
        batch,
        lambda tensor: (
            tensor.cpu().numpy()
            if tensor.dtype != torch.bfloat16
            else tensor.cpu().to(torch.float32).numpy()
        ),
    )


def batch_to_device(batch, device, non_blocking=False):
    if device == "numpy":
        return batch_to_numpy(batch)

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


def all_gather(
    item: torch.Tensor | Any, num_devices: int | None = None, dim: int | None = 0
) -> torch.Tensor | Sequence[torch.Tensor]:
    if num_devices is None:
        num_devices = torch.distributed.get_world_size()
    if isinstance(item, torch.Tensor):
        item_list = [torch.zeros_like(item) for _ in range(num_devices)]
        torch.distributed.all_gather(item_list, item)
    else:
        item_list = [None for _ in range(num_devices)]
        torch.distributed.all_gather_object(item_list, item)
    if dim is None:
        return item_list
    else:
        return torch.cat(item_list, dim=dim)


def grad_norm(params):
    return torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])


def param_norm(params):
    return torch.nn.utils.get_total_norm([p for p in params if p.requires_grad])


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return tree_map(data, lambda t: t[0])


def unsqueeze_n(tensor: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    for _ in range(n):
        tensor = tensor.unsqueeze(dim)
    return tensor


def bunsqueeze_like(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[0] == ref.shape[0], "Batch size must match"
    for _ in range(ref.dim() - tensor.dim()):
        tensor = tensor.unsqueeze(-1)
    return tensor


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
    data_rl = {k[:-4] + "1to0": v for k, v in data.items() if k[-4:] == f"{ri}to{li}"}
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


def concat_elements(data, pattern="view"):
    return concat_tree(iterelements(data, pattern=pattern))


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
        if isinstance(val_list[0], (torch.Tensor, tensor.TensorWrapper)):
            return torch.cat(val_list)
        elif isinstance(val_list[0], tuple):
            return tuple(
                [combine([v[i] for v in val_list]) for i in range(len(val_list[0]))]
            )
        elif isinstance(val_list[0], Sequence):
            return sum(val_list, start=[])
        elif isinstance(val_list[0], (int, float)):
            return val_list
        else:
            raise TypeError(f"Cannot combine values of type {type(val_list[0])}")

    return pack_tree(trees, check=check, fn=combine)


def split_tree(tree, num_splits: int) -> list[types.Tree]:
    """Split a tree into a list of trees along the first dimension of tensors."""
    flat_tree = flatten_dict(tree)

    split_trees = [{}] * num_splits
    for k, v in flat_tree.items():
        if isinstance(v, (torch.Tensor, tensor.TensorWrapper)):
            splits = torch.split(v, num_splits, dim=0)
            for i in range(num_splits):
                split_trees[i][k] = splits[i]
        else:
            raise NotImplementedError(f"Cannot split values of type {type(v)}")
    return [unflatten_dict(t) for t in split_trees]


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
    cast_to_str: bool = False,
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
        return {
            sep.join(map(str, k) if cast_to_str else k): v for k, v in flat_dict.items()
        }
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
    out = {}
    for k in sorted(flat_dict.keys()):
        out[k] = func(k, flat_dict[k])
    if unflatten:
        out = unflatten_dict(out, sep=sep)
    return out


def filter_tree(
    input_: types.Tree | Any,
    valid_fn: Callable[[types.Key, types.Value], bool],
    sep: str | None = None,
) -> types.Tree:
    """Filter a tree structure based on a predicate function."""
    flat_dict = flatten_dict(input_, sep=sep)
    filtered = {k: v for k, v in flat_dict.items() if valid_fn(k, v)}
    return unflatten_dict(filtered, sep=sep)


def tree_map(
    input_: types.Tree,
    func: Callable[[types.Value], types.Value],
    sep: str | None = None,
    unflatten: bool = True,
) -> types.Tree:
    """Apply a function to each item in a flattened dictionary."""
    return flat_map(input_, func=lambda k, v: func(v), sep=sep, unflatten=unflatten)


def tree_tensormap(
    input_: types.Tree,
    func: Callable[[torch.Tensor], torch.Tensor],
    sep: str | None = None,
    unflatten: bool = True,
) -> types.Tree:
    """Apply a function to each tensor item in a flattened dictionary."""
    return flat_map(
        input_,
        func=lambda k, v: func(v) if isinstance(v, torch.Tensor) else v,
        sep=sep,
        unflatten=unflatten,
    )


def tree_all_gather(tree: types.Tree) -> types.Tree:
    """Gather all tensors from all devices."""
    trees = all_gather(tree, dim=None)
    return concat_tree(trees)


def tree_summary(tree: types.Tree, flatten: bool = False) -> str:
    """Summarize a tree structure."""

    def _summarize(t):
        if isinstance(t, torch.Tensor):
            return f"{type(t).__name__}{tuple(t.shape)} {t.dtype}"
        elif isinstance(t, tensor.TensorWrapper):
            return f"{type(t).__name__}{tuple(t.shape)} {t.dtype}"
        elif isinstance(t, np.ndarray):
            return f"ndarray{t.shape} {t.dtype}"
        elif isinstance(t, (list, tuple)):
            return f"{type(t).__name__}[{len(t)}]"
        elif isinstance(t, (int, str, bytes, float, bool)):
            return str(t)
        elif t is None:
            return "None"
        else:
            return type(t).__name__

    return pprint.pformat(
        tree_map(
            tree, _summarize, unflatten=not flatten, sep=("." if flatten else None)
        ),
        indent=2,
    )


def assert_tree_finite(tree: types.Tree) -> types.Tree:
    """Check which tensors in a tree contain NaNs."""

    def _is_nan(k, t):
        if isinstance(t, torch.Tensor):
            assert (
                not torch.isnan(t).any().item()
            ), f"NaN detected in tensor at key: {k}"
            return not torch.isnan(t).any().item()
        else:
            return True

    return flat_map(tree, _is_nan)  # type: ignore


def print_summary(tree: types.Tree, flatten: bool = False):
    print(tree_summary(tree, flatten=flatten))


def to_sequence(map):
    return map.flatten(-2).transpose(-1, -2)


def to_map(sequence):
    n = sequence.shape[-2]
    e = math.isqrt(n)
    assert e * e == n
    assert e * e == n
    sequence.transpose(-1, -2).unflatten(-1, [e, e])


def resize_image(
    image,
    hw_in: tuple[int, int],
    hw_out: tuple[int, int],
    interpolation: str = "bilinear",
    antialias: bool = False,
):
    if hw_in[0] >= 0 and hw_in[1] >= 0:
        # Find the axes that match hw_in
        hw_dims = [image.shape.index(dim) for dim in (hw_in)]
    else:
        # You can also specify the index of the axis from behind
        hw_dims = hw_in
        hw_in = (image.shape[hw_dims[0]], image.shape[hw_dims[1]])

    if hw_in == hw_out:
        # Nothing to do
        return image
    image_in = image.moveaxis(hw_dims, (-2, -1))
    interpolation = {
        "nearest": tvf.InterpolationMode.NEAREST,
        "nn": tvf.InterpolationMode.NEAREST,
        "linear": tvf.InterpolationMode.BILINEAR,
        "bilinear": tvf.InterpolationMode.BILINEAR,
        "cubic": tvf.InterpolationMode.BICUBIC,
        "bicubic": tvf.InterpolationMode.BICUBIC,
    }[interpolation]
    resize_op = tvf.resize(
        image_in, size=hw_out, interpolation=interpolation, antialias=antialias
    )
    image_out = resize_op.moveaxis((-2, -1), hw_dims)
    return image_out


def l2_normalize(
    tensor: torch.Tensor, dim: int = -1, eps: float = 1e-10
) -> torch.Tensor:
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True).clamp_min(eps)
    return tensor / norm


def is_image_of_shape(image: torch.Tensor, hw: tuple[int, int]) -> bool:
    h, w = hw
    return h in image.shape and w in image.shape


def resize_image_like(
    tree: types.Tree,
    hw_in: tuple[int, int],
    hw_out: tuple[int, int],
    interpolation: str = "bilinear",
    antialias: bool = False,
):
    return tree_map(
        tree,
        lambda x: (
            resize_image(
                x,
                hw_in=hw_in,
                hw_out=hw_out,
                interpolation=interpolation,
                antialias=antialias,
            )
            if isinstance(x, torch.Tensor) and is_image_of_shape(x, hw_in)
            else x
        ),
    )


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


def get_image_coords(img, expand: bool = False):
    h, w = img.shape[-2:]
    coords = (
        torch.stack(
            torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=img.device),
                torch.arange(w, dtype=torch.float32, device=img.device),
                indexing="ij",
            )[::-1],
            dim=0,
        ).permute(1, 2, 0)
    ) + 0.5
    coords = coords[None]
    if expand:
        coords = coords.expand(img.shape[0], -1, -1, -1)
    return coords


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    assert tensor.ndim == mask.ndim, (tensor.shape, mask.shape)
    sum_tensor = torch.where(mask, tensor, 0.0).sum(dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
    return sum_tensor / count


def wmean(
    tensor: torch.Tensor,
    weights: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    assert tensor.ndim == weights.ndim, (tensor.shape, weights.shape)
    sum_tensor = (tensor * weights).sum(dim=dim, keepdim=keepdim)
    sum_weights = weights.sum(dim=dim, keepdim=keepdim).clamp_min(1e-6)
    return sum_tensor / sum_weights


def grid_sample(
    image: torch.Tensor,
    coords: torch.Tensor,
    interpolation: str = "bilinear",
    align_corners: bool = False,
    padding_mode: str = "zeros",
):
    assert image.dim() == coords.dim()
    is_batched = image.dim() == 4

    if is_batched:
        assert coords.dim() == 4
        return F.grid_sample(
            image.to(coords.device),
            coords,
            interpolation,
            align_corners=align_corners,
            padding_mode=padding_mode,
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
    if fmap is not None and fmap.ndim == 4:
        b = fmap.shape[0]
        grid = grid[None].expand(b, -1, -1, -1)
    grid = grid + 0.5
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


# @AMP_CUSTOM_FWD_F32
def denormalize_coords(coords, hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Denormalize coordinates from [-1, 1] to [0, H] or [0, W] (COLMAP)"""
    coords = coords.clone()
    if hw is None:
        hw = coords.shape[-3:-1]
    return torch.stack(
        [(coords[..., 0] + 1) / 2 * hw[1], (coords[..., 1] + 1) / 2 * hw[0]], dim=-1
    )


# @AMP_CUSTOM_FWD_F32
def normalize_coords(coords, hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Normalize coordinates from [0, H] or [0, W] (COLMAP) to [-1, 1]"""
    coords = coords.clone()
    if hw is None:
        hw = coords.shape[-3:-1]
    return torch.stack(
        [coords[..., 0] / hw[1] * 2 - 1, coords[..., 1] / hw[0] * 2 - 1], dim=-1
    )


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


def interpolate_points(
    pts: torch.Tensor,  # ... x N X 2
    features: torch.Tensor,  # ... x H x W x D
    mode: str = "bilinear",
    normalize: bool = False,
    is_chw: bool = False,
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> torch.Tensor:  # ... x N x D
    # Normalize to [-1, 1] for grid sampling
    if not is_chw:
        features = chw_from_hwc(features)
    if normalize:
        pts = normalize_coords(pts, features.shape[-2:])
    sampled_features = grid_sample(
        features,
        pts[..., None, :, :],
        interpolation=mode,
        align_corners=align_corners,
        padding_mode=padding_mode,
    )
    sampled_features = hwc_from_chw(sampled_features)[..., 0, :, :]
    return sampled_features


def interpolate_patches(
    pts: torch.Tensor,  # B x N x 2
    features: torch.Tensor,
    ps: int,
    mode: str = "nearest",
    normalize: bool = False,
    is_chw: bool = False,
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # B x N x D x ps x ps, B x N x 2
    if not is_chw:
        features = chw_from_hwc(features)
    if normalize:
        pts_i = pts
    else:
        pts_i = denormalize_coords(pts, features.shape[-2:])
    dummy_patch = torch.zeros(
        (1, 1, ps, ps), device=features.device, dtype=features.dtype
    )
    p_xy = get_image_coords(dummy_patch)
    cxy_i = torch.round(pts_i - ps / 2 - 0.5)
    p_xy_i = cxy_i[:, :, None, None, :] + p_xy[:, None]
    p_xy_n = normalize_coords(p_xy_i, features.shape[-2:])
    patches = torch.vmap(grid_sample, in_dims=(None, 1), out_dims=1)(
        features,
        p_xy_n,
        interpolation=mode,
        align_corners=align_corners,
        padding_mode=padding_mode,
    )
    if not is_chw:
        patches = hwc_from_chw(patches)
    return patches, p_xy_n, cxy_i


def patch_interpolate_points(
    pts: torch.Tensor,  # B x N X 2
    patches: torch.Tensor,  # B x N x D x ps x ps OR B x N x ps x ps x D
    **kwargs,
):
    return torch.vmap(interpolate_points, in_dims=0, out_dims=0)(
        pts[:, :, None],
        patches,
        **kwargs,
    )[..., 0, :]


def log_softmax(scores: torch.Tensor, dim: int | tuple = -1) -> torch.Tensor:
    """Numerically stable log softmax."""
    if isinstance(dim, int):
        return torch.log_softmax(scores, dim=dim)
    else:
        last = tuple(range(-len(dim), 0))
        scores = scores.moveaxis(dim, last)
        log_probs = torch.log_softmax(scores.flatten(-len(dim)), dim=-1).reshape(
            *scores.shape
        )
        return log_probs.moveaxis(last, dim)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return (
        kpts_q_to_t,
        scores,
        torch.where(valid, matches, -1),
        torch.where(valid, scores, 0),
    )


def match_keypoints_dense(
    pred: dict,  # Containts warp and certainty tensors
    data: dict,  # Contains keypoints and images
    max_kp_error: float,
    filter_threshold: float,
    mutual_check: bool = True,
    sparse_to_dense: bool = False,
) -> dict:
    """Match keypoints using dense correspondences."""
    kpts0 = data["keypoints0"]  # COLMAP coordinates
    kpts1 = data["keypoints1"]  # COLMAP coordinates

    img0 = data["view0"]["image"]
    img1 = data["view1"]["image"]

    mpred = {}
    pts0_i1, kp_scores0, mpred["matches0"], mpred["matching_scores0"] = (
        interpolate_matches(
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
    )
    pts1_i0, kp_scores1, mpred["matches1"], mpred["matching_scores1"] = (
        interpolate_matches(
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
    )

    if sparse_to_dense:
        keypoints0 = torch.cat([kpts0, pts1_i0], dim=-2)
        keypoints1 = torch.cat([pts0_i1, kpts1], dim=-2)
        scores = torch.cat([kp_scores0, kp_scores1], dim=-1)

        matches = torch.arange(keypoints0.shape[-2], device=keypoints0.device)[
            None
        ].repeat(keypoints0.shape[-3], 1)

        matches = torch.where(scores > filter_threshold, matches, -1)

        mpred["keypoints0"] = keypoints0
        mpred["keypoints1"] = keypoints1
        mpred["matches0"] = matches
        mpred["matches1"] = matches.clone()
        mpred["matching_scores0"] = scores
        mpred["matching_scores1"] = scores.clone()
    else:
        # Pipe the keypoints again
        mpred["keypoints0"] = data["keypoints0"]
        mpred["keypoints1"] = data["keypoints1"]
    return mpred
