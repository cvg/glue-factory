"""
Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
import shutil
import tarfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..geometry.homography import (
    compute_homography,
    sample_homography_corners,
    warp_points,
)
from ..models.cache_loader import CacheLoader, pad_local_features
from ..settings import DATA_PATH_HOMO
from ..utils.image import read_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .augmentations import IdentityAugmentation, augmentations
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def sample_homography(img, conf: dict, size: list):
    data = {}
    H, _, coords, _ = sample_homography_corners(img.shape[:2][::-1], **conf)
    data["image"] = cv2.warpPerspective(img, H, tuple(size))
    data["H_"] = H.astype(np.float32)
    data["coords"] = coords.astype(np.float32)
    data["image_size"] = np.array(size, dtype=np.float32)
    return data


class HomographyDataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "revisitop1m",  # the top-level directory
        "image_dir": "jpg/",  # the subdirectory with the images
        "image_list": "revisitop1m.txt",  # optional: list or filename of list
        "check_file_exists": False,  # check if the image exists
        "glob": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        # splits
        "train_size": 100,
        "val_size": 10,
        "shuffle_seed": 0,  # or None to skip
        # image loading
        "grayscale": False,
        "triplet": False,
        "right_only": False,  # image0 is orig (rescaled), image1 is right
        "reseed": False,
        "homography": {
            "difficulty": 0.8,
            "translation": 1.0,
            "max_angle": 60,
            "n_angles": 10,
            "patch_shape": [640, 480],
            "min_convexity": 0.05,
        },
        "photometric": {
            "name": "dark",
            "p": 0.75,
            # 'difficulty': 1.0,  # currently unused
        },
        # feature loading
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
            "thresh": 0.0,
            "max_num_keypoints": -1,
            "force_num_keypoints": False,
        },
    }

    def _init(self, conf):
        data_dir = DATA_PATH_HOMO / conf.data_dir
        if not data_dir.exists():
            if conf.data_dir == "revisitop1m":
                logger.info("Downloading the revisitop1m dataset.")
                self.download_revisitop1m()
            else:
                raise FileNotFoundError(data_dir)

        image_dir = data_dir / conf.image_dir
        images = []
        if conf.image_list is None:
            glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
            for g in glob:
                images += list(image_dir.glob("**/" + g))
            if len(images) == 0:
                raise ValueError(f"Cannot find any image in folder: {image_dir}.")
            images = [i.relative_to(image_dir).as_posix() for i in images]
            images = sorted(images)  # for deterministic behavior
            logger.info("Found %d images in folder.", len(images))
        elif isinstance(conf.image_list, (str, Path)):
            image_list = data_dir / conf.image_list
            if not image_list.exists():
                raise FileNotFoundError(f"Cannot find image list {image_list}.")
            images = image_list.read_text().rstrip("\n").split("\n")
            for image in images:
                if self.conf.check_file_exists and not (image_dir / image).exists():
                    raise FileNotFoundError(image_dir / image)
            logger.info("Found %d images in list file.", len(images))
        elif isinstance(conf.image_list, omegaconf.listconfig.ListConfig):
            images = conf.image_list.to_container()
            for image in images:
                if self.conf.check_file_exists and not (image_dir / image).exists():
                    raise FileNotFoundError(image_dir / image)
        else:
            raise ValueError(conf.image_list)

        if conf.shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(images)
        train_images = images[: conf.train_size]
        val_images = images[conf.train_size : conf.train_size + conf.val_size]
        self.images = {"train": train_images, "val": val_images}

    def download_revisitop1m(self):
        data_dir = DATA_PATH_HOMO / self.conf.data_dir
        tmp_dir = data_dir.parent / "revisitop1m_tmp"
        if tmp_dir.exists():  # The previous download failed.
            shutil.rmtree(tmp_dir)
        image_dir = tmp_dir / self.conf.image_dir
        image_dir.mkdir(exist_ok=True, parents=True)
        num_files = 100
        url_base = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/"
        list_name = "revisitop1m.txt"
        torch.hub.download_url_to_file(url_base + list_name, tmp_dir / list_name)
        for n in tqdm(range(num_files), position=1):
            tar_name = "revisitop1m.{}.tar.gz".format(n + 1)
            tar_path = image_dir / tar_name
            torch.hub.download_url_to_file(url_base + "jpg/" + tar_name, tar_path)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=image_dir)
            tar_path.unlink()
        shutil.move(tmp_dir, data_dir)

    def get_dataset(self, split):
        return _Dataset(self.conf, self.images[split], split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_names, split):
        self.conf = conf
        self.split = split
        self.image_names = np.array(image_names)
        self.image_dir = DATA_PATH_HOMO / conf.data_dir / conf.image_dir

        aug_conf = conf.photometric
        aug_name = aug_conf.name
        assert (
            aug_name in augmentations.keys()
        ), f'{aug_name} not in {" ".join(augmentations.keys())}'
        self.photo_augment = augmentations[aug_name](aug_conf)
        self.left_augment = (
            IdentityAugmentation() if conf.right_only else self.photo_augment
        )
        self.img_to_tensor = IdentityAugmentation()

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

    def _transform_keypoints(self, features, data):
        """Transform keypoints by a homography, threshold them,
        and potentially keep only the best ones."""
        # Warp points
        features["keypoints"] = warp_points(
            features["keypoints"], data["H_"], inverse=False
        )
        h, w = data["image"].shape[1:3]
        valid = (
            (features["keypoints"][:, 0] >= 0)
            & (features["keypoints"][:, 0] <= w - 1)
            & (features["keypoints"][:, 1] >= 0)
            & (features["keypoints"][:, 1] <= h - 1)
        )
        features["keypoints"] = features["keypoints"][valid]

        # Threshold
        if self.conf.load_features.thresh > 0:
            valid = features["keypoint_scores"] >= self.conf.load_features.thresh
            features = {k: v[valid] for k, v in features.items()}

        # Get the top keypoints and pad
        n = self.conf.load_features.max_num_keypoints
        if n > -1:
            inds = np.argsort(-features["keypoint_scores"])
            features = {k: v[inds[:n]] for k, v in features.items()}

            if self.conf.load_features.force_num_keypoints:
                features = pad_local_features(
                    features, self.conf.load_features.max_num_keypoints
                )

        return features

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def _read_view(self, img, H_conf, ps, left=False):
        data = sample_homography(img, H_conf, ps)
        if left:
            data["image"] = self.left_augment(data["image"], return_tensor=True)
        else:
            data["image"] = self.photo_augment(data["image"], return_tensor=True)

        gs = data["image"].new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        if self.conf.grayscale:
            data["image"] = (data["image"] * gs).sum(0, keepdim=True)

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            features = self._transform_keypoints(features, data)
            data["cache"] = features

        return data

    def getitem(self, idx):
        name = self.image_names[idx]
        img = read_image(self.image_dir / name, False)
        if img is None:
            logging.warning("Image %s could not be read.", name)
            img = np.zeros((1024, 1024) + (() if self.conf.grayscale else (3,)))
        img = img.astype(np.float32) / 255.0
        size = img.shape[:2][::-1]
        ps = self.conf.homography.patch_shape

        left_conf = omegaconf.OmegaConf.to_container(self.conf.homography)
        if self.conf.right_only:
            left_conf["difficulty"] = 0.0

        data0 = self._read_view(img, left_conf, ps, left=True)
        data1 = self._read_view(img, self.conf.homography, ps, left=False)

        H = compute_homography(data0["coords"], data1["coords"], [1, 1])

        data = {
            "name": name,
            "original_image_size": np.array(size),
            "H_0to1": H.astype(np.float32),
            "idx": idx,
            "view0": data0,
            "view1": data1,
        }

        if self.conf.triplet:
            # Generate third image
            data2 = self._read_view(img, self.conf.homography, ps, left=False)
            H02 = compute_homography(data0["coords"], data2["coords"], [1, 1])
            H12 = compute_homography(data1["coords"], data2["coords"], [1, 1])

            data = {
                "H_0to2": H02.astype(np.float32),
                "H_1to2": H12.astype(np.float32),
                "view2": data2,
                **data,
            }

        return data

    def __len__(self):
        return len(self.image_names)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = HomographyDataset(conf)
    loader = dataset.get_data_loader("train")
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
