"""
Simply load images from a folder or nested folders (does not have any split).
"""

from pathlib import Path
import argparse

import numpy as np
import torch
import torchvision
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
import zipfile

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from .base_dataset import BaseDataset
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid


class WxBSDataset(BaseDataset, torch.utils.data.Dataset):
    """Wide multiple baselines stereo dataset."""
    url = 'http://cmp.felk.cvut.cz/wbs/datasets/WxBS_v1.1.zip'
    zip_fname = 'WxBS_v1.1.zip'
    validation_pairs = ['kyiv_dolltheater2', 'petrzin']
    default_conf = {
        "preprocessing": ImagePreprocessor.default_conf,
        "data_dir": "WxBS",
        "subset": None,
        "grayscale": False,
    }
    def _init(self, conf):
        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        self.root = DATA_PATH / conf.data_dir
        if not self.root.exists():
            logger.info("Downloading the WxBS dataset.")
            self.download()
        self.pairs = self.index_dataset()
        if not self.pairs:
            raise ValueError("No image found!")
        
    def __len__(self):
        return len(self.pairs)

    def download(self):
        data_dir = self.root
        data_dir.mkdir(exist_ok=True, parents=True)
        zip_path = data_dir / self.url.rsplit("/", 1)[-1]
        torch.hub.download_url_to_file(self.url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        os.unlink(zip_path)

    def index_dataset(self):
        sets = sorted([x for x in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, x))])

        img_pairs_list = []
        for s in sets[::-1]:
            if s == '.DS_Store':
                continue
            ss = os.path.join(self.root, s)
            pairs = os.listdir(ss)
            for p in sorted(pairs):
                if p == '.DS_Store':
                    continue
                cur_dir = os.path.join(ss, p)
                if os.path.isfile(os.path.join(cur_dir, '01.png')):
                    img_pairs_list.append((os.path.join(cur_dir, '01.png'),
                                           os.path.join(cur_dir, '02.png'),
                                           os.path.join(cur_dir, 'corrs.txt'),
                                           os.path.join(cur_dir, 'crossval_errors.txt')))
                elif os.path.isfile(os.path.join(cur_dir, '01.jpg')):
                    img_pairs_list.append((os.path.join(cur_dir, '01.jpg'),
                                           os.path.join(cur_dir, '02.jpg'),
                                           os.path.join(cur_dir, 'corrs.txt'),
                                           os.path.join(cur_dir, 'crossval_errors.txt')))
                else:
                    continue
        return img_pairs_list

    def __getitem__(self, idx):
        imgfname1, imgfname2, pts_fname, err_fname = self.pairs[idx]
        data0 = self.preprocessor(load_image(imgfname1))
        data1 = self.preprocessor(load_image(imgfname2))
        pts = np.loadtxt(pts_fname)
        crossval_errors = np.loadtxt(err_fname)
        pair_name = '/'.join(pts_fname.split('/')[-3:-1]).replace('/', '_')
        scene_name = '/'.join(pts_fname.split('/')[-3:-2])
        out = {
            "pts_0to1": pts,
            "scene": scene_name,
            "view0": data0,
            "view1": data1,
            "idx": idx,
            "name": pair_name,
            "crossval_errors": crossval_errors}
        return out

    def get_dataset(self, split):
        assert split in ['val', 'test']
        return self


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = WxBSDataset(conf)
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
