"""
Simply load images from a folder or nested folders (does not have any split).
"""

import argparse
import logging
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from PIL import Image, ImageDraw
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def get_lines_gt( lines: np.ndarray, size, width=3):

    img = Image.new(mode="RGB", size=(size[0], size[1]))
    draw = ImageDraw.Draw(img)
    for line in lines.astype(np.int32):
        draw.line(line.reshape(-1).tolist(), fill=(256,256,256), width=width)

    # img.save("test.png")
    # exit()

    return np.array(img)

def get_lines(lines: np.ndarray, points: np.ndarray):

    lines_xy = []
    for line in lines:
        lines_xy.append([points[line[0]], points[line[1]]])

    return np.array(lines_xy)

    


class Wireframe(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "preprocessing": ImagePreprocessor.default_conf,
        "data_dir": "wireframe-pointline",
        "subset": None,
        "ignore_large_images": True,
        "grayscale": False,
    }

    def _init(self, conf):
        assert conf.batch_size == 1
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.root = DATA_PATH / conf.data_dir
        if not self.root.exists():
            logger.info("Wireframe Data Missing")

            raise FileNotFoundError("Please Download The pointline.zip and extract into the data folder")

            # URL: https://github.com/huangkuns/wireframe


        test_size = -1 # TODO: ADD TO CONFIG
        self.items = sorted([x.name for x in self.root.iterdir()])[:test_size]


    def get_dataset(self, split):
        assert split in ["val", "test"]
        return self

    # def _read_image(self, seq: str, idx: int) -> dict:
    #     img = load_image(self.root / seq / f"{idx}.ppm", self.conf.grayscale)
    #     return self.preprocessor(img)

    def __getitem__(self, idx):
        filename = self.items[idx]

        file = {}
        with open(self.root / filename, 'rb') as f:
            fileread = pickle.load(f)
            file['img'] = fileread['img']
            file['lines'] = fileread['lines']
            file['points'] = fileread['points']
            file['imgname'] = fileread['imgname']
            
            del fileread

        image = self.preprocessor(torch.from_numpy(file['img'].astype(np.float32)).permute(2,0,1))
        lines = get_lines(file["lines"], file["points"])
        # lines_gt = self.preprocessor(torch.from_numpy(get_lines_gt(lines, file['img'].shape).astype(np.float32)))

        return {
            "view0": image,
            "view1": image, ## redundancy
            "name": file["imgname"],
            "points": torch.as_tensor(file["points"]),
            "lines": torch.as_tensor(file["lines"]),
            "shape": torch.as_tensor(file['img'].shape),
            "line_ends": lines,
        }

    def __len__(self):
        return len(self.items)

def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = Wireframe(conf)
    loader = dataset.get_data_loader("test")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                data["img"][0]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


def dataset_test(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = Wireframe(conf)

    for idx in range(args.num_items):
        data = dataset[idx]
        print(data["view0"].keys())
        print(f'[{idx}] {data["name"]}: {data["view0"]["image"].shape} {data["view0"].keys()}')


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    dataset_test(args)
    # visualize(args)
