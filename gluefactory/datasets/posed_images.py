"""
Simply load images from a folder or nested folders (does not have any split).
"""

import ast
import logging

import cv2
import h5py
import numpy as np
import torch
from _collections_abc import Iterable
from tqdm import tqdm

from .. import settings
from ..geometry import reconstruction
from ..utils import preprocess
from . import base_dataset

logger = logging.getLogger(__name__)

# Structure:

# <root>
#   scene
#       <image_dir>/
#       <depth_dir>/
#       views.txt
#       pairs.txt
#       view_groups.txt
#       # extra_data.txt (optional)


def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def parse_pose_camera(line):
    pose = reconstruction.Pose.from_Rt(
        torch.from_numpy(np.array(line[:9]).astype(np.float32).reshape(3, 3)),
        torch.from_numpy(np.array(line[9:12]).astype(np.float32)),
    )
    camera_dict = {
        "model": line[12],
        "width": int(line[13]),
        "height": int(line[14]),
        "params": np.array(line[15:]).astype(np.float32),
    }
    return pose, reconstruction.Camera.from_colmap(camera_dict)


def load_depth(depth_path, dformat):
    if dformat == "png":
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        depth_img = depth_img.astype(np.float32) / 256
        return torch.tensor(depth_img)  # .transpose(-1, -2)
    elif dformat == "h5":
        with h5py.File(str(depth_path), "r") as f:
            depth = f["/depth"].__array__().astype(np.float32, copy=False)
        return torch.Tensor(depth)
    else:
        raise ValueError(dformat)


class PosedImageDataset(base_dataset.BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "root": "???",
        "image_dir": "???",
        "depth_dir": None,  # optional
        "views": "???",
        "extra_data": None,  # text file with extra data
        "extra_keys": [],
        "view_groups": None,
        "depth_format": "h5",
        "scene_list": None,
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
        "batch_size": 1,
    }

    def get_image_path(self, scene, img_name):
        return self.root / self.conf.image_dir.format(scene=scene) / img_name

    def get_depth_path(self, scene, img_name):
        depth_name = f"{img_name.split('.')[0]}.{self.conf.depth_format}"
        return self.root / self.conf.depth_dir.format(scene=scene) / depth_name

    def _init(self, conf):
        self.root = settings.DATA_PATH / conf.root
        assert self.root.exists()
        # we first read the scenes
        if isinstance(conf.scene_list, Iterable):
            self.scenes = list(conf.scene_list)
        elif isinstance(conf.scene_list, str):
            scenes_path = self.root / conf.scene_list
            self.scenes = scenes_path.read_text().rstrip("\n").split("\n")
        else:
            self.scenes = [s.name for s in self.root.glob("*")]
        logger.info(f"Found scenes {self.scenes}.")
        # read posed views, check if images exist
        self.views = {}
        self.extra_data = {}

        self.items = []
        for scene in self.scenes:
            scene_view_path = self.root / conf.views.format(scene=scene)
            with open(str(scene_view_path), "r") as f:
                self.views[scene] = {
                    line.rstrip().split(" ")[0]: parse_pose_camera(
                        line.rstrip().split(" ")[1:]
                    )
                    for line in f
                }

            # Check if images exist
            for imname in self.views[scene].keys():
                impath = self.get_image_path(scene, imname)
                assert impath.exists(), impath
                # check if depth exists (optional)
                if conf.depth_dir:
                    depthpath = self.get_depth_path(scene, imname)
                    assert depthpath.exists(), depthpath
            if conf.extra_data:
                with open(
                    str(self.root / conf.extra_data.format(scene=scene)), "r"
                ) as f:
                    self.extra_data[scene] = {
                        line.rstrip().split(" ")[0]: [
                            ast.literal_eval(x) for x in line.rstrip().split(" ")[1:]
                        ]
                        for line in f
                        if not line.startswith("#")
                    }
                for k in self.extra_data[scene]:
                    assert k in self.views[scene]

            if conf.view_groups is None:
                self.items += [[scene, imname] for imname in self.views[scene].keys()]
            else:
                view_group_path = self.root / conf.view_groups.format(scene=scene)
                view_groups = view_group_path.read_text().rstrip("\n").split("\n")
                self.items += [[scene] + p.split(" ") for p in view_groups]

        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)

    def get_dataset(self, split: str, epoch: int = 0):
        return self

    def _read_view(self, scene, name):
        pose, camera = self.views[scene][name]
        img = preprocess.load_image(self.get_image_path(scene, name))
        data = self.preprocessor(img)
        data["T_w2cam"] = pose
        data["camera"] = camera.compose_image_transform(data["transform"])
        data["name"] = name

        if self.conf.depth_dir:
            depth = load_depth(
                self.get_depth_path(scene, name), dformat=self.conf.depth_format
            )
            data["depth"] = self.preprocessor.interpolate(
                depth[None],
                data["transform"],
                data["image"].shape[-2:],
                mode="nearest",
            )[0]
            data["valid_depth"] = (data["depth"] > 0).float()

            assert data["depth"].shape[-2:] == data["image"].shape[-2:]

        if self.conf.extra_data:
            data = {
                **data,
                **dict(zip(self.conf.extra_keys, self.extra_data[scene][name])),
            }
        return data

    def __getitem__(self, idx):
        scene, *image_names = self.items[idx]

        data = {}
        for i, image_name in enumerate(image_names):
            data[f"view{i}"] = self._read_view(scene, image_name)

        data["name"] = "/".join([iname.replace("/", "-") for iname in image_names])
        data["query_name"] = image_names[0]
        data["references"] = image_names[1:]
        data["scene"] = scene
        data["nviews"] = len(image_names)

        for i in range(1, data["nviews"]):
            data[f"T_0to{i}"] = (
                data[f"view{i}"]["T_w2cam"] @ data["view0"]["T_w2cam"].inv()
            )

        def recursive_tolist(d):
            return {
                k: [v] if not isinstance(v, dict) else recursive_tolist(v)
                for k, v in d.items()
            }

        return data

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from ..visualization.viz2d import plot_heatmaps, plot_image_grid

    conf = {
        "root": "",
        "image_dir": "{scene}/images",
        "depth_dir": "{scene}/depths",
        "views": "{scene}/views.txt",
        "view_groups": "{scene}/pairs.txt",
        "depth_format": "h5",
        "scene_list": ["megadepth1500"],
        "preprocessing": {
            "resize": 1600,
            "side": "long",
            "interpolation": "area",
            "antialias": False,
        },
        "num_workers": 1,
    }

    dataset = PosedImageDataset(conf)

    loader = dataset.get_data_loader("test")

    images, depths = [], []
    for i, data in tqdm(enumerate(loader)):
        images.append(
            [
                data[f"view{i}"]["image"][0].permute(1, 2, 0)
                for i in range(data["nviews"][0])
            ]
        )
        depths.append([data[f"view{i}"]["depth"][0] for i in range(data["nviews"][0])])
        if i > 3:
            break

    axes = plot_image_grid(images, dpi=200)
    for i in range(len(images)):
        plot_heatmaps(depths[i], axes=axes[i])
    plt.savefig("posed_images.png")
    plt.show()
