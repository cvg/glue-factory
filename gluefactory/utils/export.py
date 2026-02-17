"""
Export the predictions of a model for a given dataloader (e.g. ImageFolder).
Use a standalone script with `python3 -m dsfm.scipts.export_predictions dir`
or call from another script.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from ..geometry import transforms as gtr
from . import misc


@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
    mode: str = "w",
    store_directional: bool = False,
    mixed_precision: bool = False,
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for data_ in tqdm(loader):
        with torch.autocast(
            device_type=device,
            enabled=mixed_precision,
            dtype=torch.float16,
        ):
            data = misc.batch_to_device(data_, device, non_blocking=True)
            pred = model(data)
            if callback_fn is not None:
                pred = {**callback_fn(pred, data), **pred}
            all_keys = set(pred.keys())
            if keys != "*":
                matched_keys = []
                for pattern in keys:
                    found = False
                    for key in all_keys - set(matched_keys):
                        if pattern in key:
                            matched_keys.append(key)
                            found = True
                    assert found, f"Pattern {pattern} not found in prediction keys."
            else:
                matched_keys = list(all_keys)
            for pattern in optional_keys:
                for key in all_keys - set(matched_keys):
                    if pattern in key:
                        matched_keys.append(key)

            pred = {k: v for k, v in pred.items() if k in matched_keys}
            assert len(pred) > 0

            # renormalization: transform from preprocessed to original image space
            for k in pred.keys():
                for prefix in ("keypoints", "orig_lines", "lines", "p2d0_i", "p2d1_i"):
                    if k.startswith(prefix):
                        idx = k.replace(prefix, "")
                        transform = (
                            data["transform"]
                            if len(idx) == 0
                            else data[f"view{idx}"]["transform"]
                        )
                        inv_transform = torch.linalg.inv(transform.to(pred[k].dtype))
                        pred[k] = gtr.transform_points(inv_transform, pred[k])
                        break

            pred = {k: v[0].cpu() for k, v in pred.items()}

            if as_half:
                pred = {
                    k: v.half() if torch.is_floating_point(v) else v
                    for k, v in pred.items()
                }
            try:
                name = data["name"][0]
                if store_directional:
                    view_names = [
                        x["name"][0] for x in misc.iterelements(data, "view{i}")
                    ]
                    assert (
                        len(view_names) == 2
                    ), "Can only store directional data for 2-view inputs."
                    pairs = [(0, 1), (1, 0)]
                    for k, (i, j) in enumerate(pairs):
                        grpi = hfile.require_group(view_names[i])
                        grpi_j = grpi.create_group(view_names[j])
                        write_tree_h5(grpi_j, misc.get_view(pred, str(k)))
                else:
                    grp = hfile.create_group(name)
                    write_tree_h5(grp, pred)
            except RuntimeError:
                print(f"Skipping {name} (already in file?)")
                continue

            del pred
    hfile.close()
    return output_file


def write_tree_h5(
    grp_or_path,
    data: dict[str, any],
    mode: str = "w",
    compression: str = "gzip",
    compression_opts: int = 2,
):
    """Write a dict tree to an h5 group or file with type annotations.

    Supports automatic serialization of TensorWrapper subclasses (Pose, Camera)
    via their to_h5/from_h5 methods.

    Args:
        grp_or_path: h5py Group or file path to write to
        data: Dict to write
        mode: File mode if grp_or_path is a path ('w', 'a', etc.)
        compression: Compression algorithm (None, 'gzip', 'lzf')
        compression_opts: Compression level (1-9 for gzip)
    """
    from ..utils.tensor import TensorWrapper

    def get_compression_kwargs(arr):
        if arr.ndim == 0:
            return {}
        return {"compression": compression, "compression_opts": compression_opts}

    def write_value(grp, key, value):
        # Handle TensorWrapper subclasses (Pose, Camera, etc.)
        if isinstance(value, TensorWrapper):
            value.to_h5(grp, key, **get_compression_kwargs(value.data_.cpu().numpy()))
            return

        if isinstance(value, torch.Tensor):
            arr = value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, dict):
            subgrp = grp.create_group(key)
            for k, v in value.items():
                write_value(subgrp, k, v)
            return
        else:
            grp.attrs[key] = value
            return

        grp.create_dataset(key, data=arr, **get_compression_kwargs(arr))

    def write_to_grp(grp):
        for k, v in data.items():
            write_value(grp, k, v)

    if isinstance(grp_or_path, (str, Path)):
        Path(grp_or_path).parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(str(grp_or_path), mode) as hfile:
            write_to_grp(hfile)
        return grp_or_path
    else:
        write_to_grp(grp_or_path)
