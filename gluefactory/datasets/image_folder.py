"""
Simply load images from a folder or nested folders (does not have any split).
"""

import logging
from pathlib import Path

import omegaconf
import torch

from ..utils.image import ImagePreprocessor, load_image
from .base_dataset import BaseDataset


class ImageFolder(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "glob": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "images": "???",
        "root_folder": "/",
        "preprocessing": ImagePreprocessor.default_conf,
    }

    def _init(self, conf):
        self.root = conf.root_folder
        if isinstance(conf.images, str):
            if not Path(conf.images).is_dir():
                with open(conf.images, "r") as f:
                    self.images = f.read().rstrip("\n").split("\n")
                logging.info(f"Found {len(self.images)} images in list file.")
            else:
                self.images = []
                glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
                for g in glob:
                    self.images += list(Path(conf.images).glob("**/" + g))
                if len(self.images) == 0:
                    raise ValueError(
                        f"Could not find any image in folder: {conf.images}."
                    )
                self.images = [i.relative_to(conf.images) for i in self.images]
                self.root = conf.images
                logging.info(f"Found {len(self.images)} images in folder.")
        elif isinstance(conf.images, omegaconf.listconfig.ListConfig):
            self.images = conf.images.to_container()
        else:
            raise ValueError(conf.images)

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        path = self.images[idx]
        img = load_image(path)
        data = {"name": str(path), **self.preprocessor(img)}
        return data

    def __len__(self):
        return len(self.images)
