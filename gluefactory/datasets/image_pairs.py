"""
Simply load images from a folder or nested folders (does not have any split).
"""

from pathlib import Path

import numpy as np
import torch

from .. import settings
from ..geometry import reconstruction
from ..models import cache_loader
from ..utils import preprocess
from .base_dataset import BaseDataset


def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def parse_homography(homography_elems) -> reconstruction.Camera:
    return (
        np.array([float(x) for x in homography_elems[:9]])
        .reshape(3, 3)
        .astype(np.float32)
    )


def parse_camera(calib_elems) -> reconstruction.Camera:
    # assert len(calib_list) == 9
    K = np.array([float(x) for x in calib_elems[:9]]).reshape(3, 3).astype(np.float32)
    return reconstruction.Camera.from_calibration_matrix(K)


def parse_relative_pose(pose_elems) -> reconstruction.Pose:
    if len(pose_elems) == 12:
        R, t = pose_elems[:9], pose_elems[9:12]
        R = np.array([float(x) for x in R]).reshape(3, 3).astype(np.float32)
        t = np.array([float(x) for x in t]).astype(np.float32)
        return reconstruction.Pose.from_Rt(R, t)
    elif len(pose_elems) == 16:
        T = np.array([float(x) for x in pose_elems]).reshape(4, 4).astype(np.float32)
        return reconstruction.Pose.from_4x4mat(T)
    else:
        raise ValueError(f"Can not interpret pose {pose_elems}.")


class ImagePairs(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "pairs": "???",  # ToDo: add image folder interface
        "root": "???",
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
        "extra_data": None,  # relative_pose, homography
        "load_features": {
            "do": False,
            **cache_loader.CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        pair_f = (
            Path(conf.pairs)
            if Path(conf.pairs).exists()
            else settings.DATA_PATH / conf.pairs
        )
        with open(str(pair_f), "r") as f:
            self.items = [line.rstrip() for line in f]
        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)

        if conf.load_features.do:
            self.feature_loader = cache_loader.CacheLoader(conf.load_features)

    def get_dataset(self, split):
        return self

    def _read_view(self, name):
        path = settings.DATA_PATH / self.conf.root / name
        img = preprocess.load_image(path)
        data = self.preprocessor(img)
        data["name"] = name
        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            data = {"cache": features, **data}
        return data

    def __getitem__(self, idx):
        line = self.items[idx]
        pair_data = line.split(" ")
        name0, name1 = pair_data[:2]
        data0 = self._read_view(name0)
        data1 = self._read_view(name1)

        data = {
            "view0": data0,
            "view1": data1,
        }
        if self.conf.extra_data == "relative_pose":
            data["view0"]["camera"] = parse_camera(pair_data[2:11]).scale(
                data0["scales"]
            )
            data["view1"]["camera"] = parse_camera(pair_data[11:20]).scale(
                data1["scales"]
            )
            data["T_0to1"] = parse_relative_pose(pair_data[20:])
            data["T_1to0"] = data["T_0to1"].inv()
        elif self.conf.extra_data == "homography":
            data["H_0to1"] = (
                data1["transform"]
                @ parse_homography(pair_data[2:11])
                @ np.linalg.inv(data0["transform"])
            )
        else:
            assert (
                self.conf.extra_data is None
            ), f"Unknown extra data format {self.conf.extra_data}"

        data["name"] = names_to_pair(name0, name1)
        return data

    def __len__(self):
        return len(self.items)
