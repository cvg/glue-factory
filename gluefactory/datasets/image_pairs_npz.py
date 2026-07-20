"""
Load image pairs from NPZ file (compatible with datasets from RDD paper):

https://github.com/xtcpete/rdd
"""

from pathlib import Path

import numpy as np
import torch

from .. import settings
from ..geometry import reconstruction
from ..utils import preprocess
from .base_dataset import BaseDataset


class ImagePairsNPZ(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "pairs": "???",  # ToDo: add image folder interface
        "root": "???",
        "images": "",  # at root
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
    }

    def _init(self, conf):
        pair_f = (
            Path(conf.pairs)
            if Path(conf.pairs).exists()
            else settings.DATA_PATH / conf.root / conf.pairs
        )
        self.items = list(np.load(pair_f, allow_pickle=True)["pair_info"])
        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)

    def get_dataset(self, split: str, epoch: int = 0):
        return self

    def _read_view(self, name):
        path = settings.DATA_PATH / self.conf.root / self.conf.images / name
        img = preprocess.load_image(path)
        data = self.preprocessor(img)
        data["name"] = name
        return data

    def __getitem__(self, idx):
        pair_info = self.items[idx]

        data = {
            "name": "/".join(pair_info["pair_names"]),
            "scene": self.conf.root,
            "nviews": 2,
        }

        for i, (imname, pose4x4, intrinsics) in enumerate(
            zip(pair_info["pair_names"], pair_info["pose"], pair_info["intrinsic"])
        ):
            data[f"view{i}"] = self._read_view(imname)
            data[f"view{i}"]["T_w2cam"] = reconstruction.Pose.from_4x4mat(
                torch.from_numpy(pose4x4)
            ).float()
            data[f"view{i}"]["camera"] = (
                reconstruction.Camera.from_calibration_matrix(
                    torch.from_numpy(intrinsics)
                )
                .float()
                .compose_image_transform(data[f"view{i}"]["transform"])
            )

        data["T_0to1"] = data["view1"]["T_w2cam"] @ data["view0"]["T_w2cam"].inv()
        data["T_1to0"] = data["T_0to1"].inv()

        return data

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from ..visualization.viz2d import plot_image_grid

    conf = {
        "root": "air_ground",
        "images": "images",
        "pairs": "indices.npz",
        "preprocessing": {
            "resize": 1600,
            "side": "long",
            "interpolation": "area",
            "antialias": False,
        },
        "num_workers": 1,
    }

    dataset = ImagePairsNPZ(conf)

    loader = dataset.get_data_loader("test")

    images = []
    for i, data in tqdm(enumerate(loader)):
        images.append(
            [
                data[f"view{i}"]["image"][0].permute(1, 2, 0)
                for i in range(data["nviews"][0])
            ]
        )
        if i > 3:
            break

    axes = plot_image_grid(images, dpi=200)
    plt.savefig("image_pairs_npz.png")
    plt.show()
