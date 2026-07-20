"""
Simply load images from a folder or nested folders (does not have any split).
"""

import ast
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from .. import settings
from ..utils import preprocess
from . import base_dataset

logger = logging.getLogger(__name__)


def names_to_pair(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


class InLocPairsDataset(base_dataset.BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "root": "inloc",
        "pairs": "inloc/pairs/pairs-query-netvlad40-temporal.txt",
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
        "batch_size": 1,
        "load_xyz": False,
    }

    def _init(self, conf):
        self.root = settings.DATA_PATH / conf.root
        assert self.root.exists()
        # we first read the scenes
        pair_f = (
            Path(conf.pairs)
            if Path(conf.pairs).exists()
            else settings.DATA_PATH / conf.pairs
        )
        with open(str(pair_f), "r") as f:
            self.items = [line.rstrip() for line in f]

        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)

    def get_dataset(self, split: str, epoch: int = 0):
        return self

    def _read_view(self, name):
        if (Path(self.conf.root) / name).exists():
            path = Path(self.conf.root) / name
        else:
            path = settings.DATA_PATH / self.conf.root / name
        img = preprocess.load_image(path)
        data = self.preprocessor(img)
        data["name"] = name

        return data

    def __getitem__(self, idx):
        line = self.items[idx]
        pair_data = line.split(" ")
        name0, name1 = pair_data[:2]
        data0 = self._read_view(name0)
        data1 = self._read_view(name1)

        data = {}
        data = {
            "view0": data0,
            "view1": data1,
        }

        data["name"] = names_to_pair(name0, name1)
        data["query_name"] = name0
        data["nviews"] = 2

        return data

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from ..visualization.viz2d import plot_image_grid

    conf = {
        "preprocessing": {
            "resize": 1600,
            "side": "long",
            "interpolation": "area",
            "antialias": False,
        },
        "num_workers": 1,
    }

    dataset = InLocPairsDataset(conf)

    loader = dataset.get_data_loader("test")

    images, depths = [], []
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
    plt.savefig("inloc_pairs.png")
    plt.show()
