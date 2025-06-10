import string

import h5py
import torch

from ..datasets.base_dataset import collate
from ..settings import DATA_PATH
from ..utils.tensor import batch_to_device
from .base_model import BaseModel
from .utils.misc import pad_to_length


def pad_local_features(pred: dict, seq_l: int):
    pred["keypoints"] = pad_to_length(
        pred["keypoints"],
        seq_l,
        -2,
        mode="random_c",
    )
    if "keypoint_scores" in pred.keys():
        pred["keypoint_scores"] = pad_to_length(
            pred["keypoint_scores"], seq_l, -1, mode="zeros"
        )
    if "descriptors" in pred.keys():
        pred["descriptors"] = pad_to_length(
            pred["descriptors"], seq_l, -2, mode="random"
        )
    if "scales" in pred.keys():
        pred["scales"] = pad_to_length(pred["scales"], seq_l, -1, mode="zeros")
    if "oris" in pred.keys():
        pred["oris"] = pad_to_length(pred["oris"], seq_l, -1, mode="zeros")

    if "depth_keypoints" in pred.keys():
        pred["depth_keypoints"] = pad_to_length(
            pred["depth_keypoints"], seq_l, -1, mode="zeros"
        )
    if "valid_depth_keypoints" in pred.keys():
        pred["valid_depth_keypoints"] = pad_to_length(
            pred["valid_depth_keypoints"], seq_l, -1, mode="zeros"
        )
    return pred


def pad_line_features(pred, seq_l: int = None):
    raise NotImplementedError


def recursive_load(grp, pkeys):
    return {
        k: (
            torch.from_numpy(grp[k].__array__())
            if isinstance(grp[k], h5py.Dataset)
            else recursive_load(grp[k], list(grp.keys()))
        )
        for k in pkeys
    }


class CacheLoader(BaseModel):
    default_conf = {
        "path": "???",  # can be a format string like exports/{scene}/
        "data_keys": None,  # load all keys
        "device": None,  # load to same device as data
        "trainable": False,
        "add_data_path": True,
        "collate": True,
        "scale": ["keypoints", "lines", "orig_lines"],
        "padding_fn": None,
        "padding_length": None,  # required for batching!
        "numeric_type": "float32",  # [None, "float16", "float32", "float64"]
    }

    required_data_keys = ["name"]  # we need an identifier

    def _init(self, conf):
        self.hfiles = {}
        self.padding_fn = conf.padding_fn
        if self.padding_fn is not None:
            self.padding_fn = eval(self.padding_fn)
        self.numeric_dtype = {
            None: None,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[conf.numeric_type]

    def _forward(self, data):
        preds = []
        device = self.conf.device
        if not device:
            devices = set(
                [v.device for v in data.values() if isinstance(v, torch.Tensor)]
            )
            if len(devices) == 0:
                device = "cpu"
            else:
                assert len(devices) == 1
                device = devices.pop()

        var_names = [x[1] for x in string.Formatter().parse(self.conf.path) if x[1]]
        for i, name in enumerate(data["name"]):
            fpath = self.conf.path.format(**{k: data[k][i] for k in var_names})
            if self.conf.add_data_path:
                fpath = DATA_PATH / fpath
            hfile = h5py.File(str(fpath), "r")
            grp = hfile[name]
            pkeys = (
                self.conf.data_keys if self.conf.data_keys is not None else grp.keys()
            )
            pred = recursive_load(grp, pkeys)
            if self.numeric_dtype is not None:
                pred = {
                    k: (
                        v
                        if not isinstance(v, torch.Tensor)
                        or not torch.is_floating_point(v)
                        else v.to(dtype=self.numeric_dtype)
                    )
                    for k, v in pred.items()
                }
            pred = batch_to_device(pred, device)
            for k, v in pred.items():
                for pattern in self.conf.scale:
                    if k.startswith(pattern):
                        view_idx = k.replace(pattern, "")
                        scales = (
                            data["scales"]
                            if len(view_idx) == 0
                            else data[f"view{view_idx}"]["scales"]
                        )
                        pred[k] = pred[k] * scales[i]
            # use this function to fix number of keypoints etc.
            if self.padding_fn is not None:
                pred = self.padding_fn(pred, self.conf.padding_length)
            preds.append(pred)
            hfile.close()
        if self.conf.collate:
            return batch_to_device(collate(preds), device)
        else:
            assert len(preds) == 1
            return batch_to_device(preds[0], device)

    def loss(self, pred, data):
        raise NotImplementedError
