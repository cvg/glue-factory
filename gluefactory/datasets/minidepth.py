import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import logging


from gluefactory.datasets import BaseDataset
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.settings import DATA_PATH
from gluefactory.utils.image import load_image, ImagePreprocessor

logger = logging.getLogger(__name__)


class MiniDepthDataset(BaseDataset):
    """
    Assumes minidepth dataset in folder as jpg images
    """
    default_conf = {
        "data_dir": "minidepth/images",
        "grayscale": False,
        "train_batch_size": 2,  # prefix must match split
        "test_batch_size": 1,
        "split": "train",
        "seed": 0,
        "preprocessing": {
            'resize': [800, 800]
        },
        "load_features": {
            "do": False
        },
    }

    def _init(self, conf):
        self.grayscale = bool(conf.grayscale)
        # self.conf is set in superclass

        # set img preprocessor
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        # Auto-download the dataset
        if not (DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the minidepth dataset...")
            self.download_minidepth()

        # Form pairs of images from the multiview dataset
        self.img_dir = DATA_PATH / conf.data_dir
        # load all image paths
        self.image_paths = list(Path(self.img_dir).glob("**/*.jpg"))
        # making them relative for system independent names in export files (path used as name in export)
        self.image_paths = [i.relative_to(self.img_dir) for i in self.image_paths.copy()]
        if len(self.image_paths) == 0:
            raise ValueError(
                f"Could not find any image in folder: {self.img_dir}."
            )
        logger.info("IMAGE PATHS: ", len(self.image_paths))
        # Load features
        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

    def download_minidepth(self):
        logger.info("Downloading the MiniDepth dataset...")
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "minidepth_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://filedn.com/lt6zb4ORSwapNyVniJf1Pqh"
        zip_name = "minidepth.zip"
        zip_path = tmp_dir / zip_name
        torch.hub.download_url_to_file(url_base + zip_name, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        shutil.move(tmp_dir / zip_name.split(".")[0], data_dir)

    def get_dataset(self, split):
        return self

    def _read_image(self, path, enforce_batch_dim=True):
        # Only reads Image as tensor, no additional metadata
        img = load_image(path, grayscale=self.grayscale)
        if enforce_batch_dim:
            if img.ndim < 4:
                img = img.unsqueeze(0)
        assert img.ndim >= 3
        print("Read-Img: ", img.shape)
        return img

    def _read_groundtruth(self, image_path, enforce_batch_dim=True):
        raise NotImplementedError

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = self._read_image(self.img_dir / path)
        data = {"name": str(path), **self.preprocessor(img)}  # add metadata, like transform, image_size etc...
        if self.conf.load_features.do:
            gt = self._read_groundtruth(path)
            data = {**data, **gt}
        # fix err in dkd todo check together with batching
        del data['image_size']  # torch.from_numpy(data['image_size'])
        return data

    def __len__(self):
        return len(self.image_paths)

