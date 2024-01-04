"""
Simply load images from a folder or nested folders (does not have any split).
"""
import argparse
import logging
import zipfile
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def read_homography(path):
    with open(path, 'r') as hf:
        lines = hf.readlines()
        H = []
        for l in lines:
            H.append([float(x) for x in l.replace('\t',' ').strip().split(' ') if len(x) > 0])
        H = np.array(H)
        H = H / H[2, 2]
    return H

class EVD(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "preprocessing": ImagePreprocessor.default_conf,
        "data_dir": "EVD",
        "subset": None,
        "grayscale": False,
    }

    url = "http://cmp.felk.cvut.cz/wbs/datasets/EVD.zip"
    md5hash = '7ce52151abdafa71a609424d09e43075'

    def _init(self, conf):
        assert conf.batch_size == 1
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.root = DATA_PATH / conf.data_dir
        if not self.root.exists():
            logger.info("Downloading the EVD dataset.")
            self.download()
        self.pairs = self.index_dataset()
        if not self.pairs:
            raise ValueError("No image found!")
        self.items = []  # (seq, q_idx, is_illu)

    def download(self):
        data_dir = self.root.parent
        data_dir.mkdir(exist_ok=True, parents=True)
        zip_path = data_dir / self.url.rsplit("/", 1)[-1]
        torch.hub.download_url_to_file(self.url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)


    def index_dataset(self):
        sets = sorted([x for x in os.listdir(os.path.join(self.root, '1'))])
        img_pairs_list = []
        for s in sets:
            if s == '.DS_Store':
                continue
            img_pairs_list.append(((os.path.join(self.root, '1', s)),
                                  (os.path.join(self.root, '2', s)),
                                  (os.path.join(self.root, 'h', s.replace('png', 'txt')))))
        return img_pairs_list

    def __getitem__(self, idx):
        imgfname1, imgfname2, h_fname = self.pairs[idx]
        H = read_homography(h_fname)
        data0 = self.preprocessor(load_image(imgfname1))
        data1 = self.preprocessor(load_image(imgfname2))
        H = data1["transform"] @ H @ np.linalg.inv(data0["transform"])
        pair_name = imgfname1.split('/')[-1].split('.')[0]
        return {
            "H_0to1": H.astype(np.float32),
            "scene": pair_name,
            "view0": data0,
            "view1": data1,
            "idx": idx,
            "name": pair_name,
        }

    def __len__(self):
        return len(self.pairs)

    def get_dataset(self, split):
        return self

def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = EVD(conf)
    loader = dataset.get_data_loader("test")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
