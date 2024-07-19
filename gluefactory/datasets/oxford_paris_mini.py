import logging
import shutil
import tarfile
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.datasets import BaseDataset
from gluefactory.settings import DATA_PATH, root
from gluefactory.utils.image import ImagePreprocessor, load_image

logger = logging.getLogger(__name__)


class OxfordParisMini(BaseDataset):
    """
    Subset of the Oxford Paris dataset as defined here: https://cmp.felk.cvut.cz/revisitop/
    Supports loading groundtruth and only serves images for that gt exists.
    Dataset only deals with loading one element. Batching is done by Dataloader!

    This class only used to load conf and autodownload. For usable Datasets, use get_dataset to get Dataset with split
    """

    default_conf = {
        "data_dir": "oxford_paris_mini/images",  # as subdirectory of DATA_PATH(defined in settings.py)
        "grayscale": False,
        "train_batch_size": 2,  # prefix must match split
        "test_batch_size": 1,
        "val_batch_size": 1,
        "all_batch_size": 1,
        "device": None,  # specify device to move image data to. if None is given, just read, skip move to device
        "split": "train",  # train, val, test
        "seed": 0,
        "num_workers": 0,  # number of workers used by the Dataloader
        "prefetch_factor": None,
        "preprocessing": {"resize": [800, 800]},
        "load_features": {
            "do": False,
            "check_exists": True,
            "check_nan": False,
            "device": None,  # choose device to move groundtruthdata to if None is given, just read, skip move to device
            "point_gt": {
                "path": "outputs/results/superpoint_gt",
                "data_keys": ["superpoint_heatmap"],
            },
            "line_gt": {
                "path": "outputs/results/deeplsd_gt",
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
            },
        },
        "img_list": "gluefactory/datasets/oxford_paris_images.txt",  # path to image list containing images we want to use for this dataset to be consitent (given from repo root)
        "rand_shuffle_seed": None,  # seed to randomly shuffle before split in train and val
        "val_size": 10,  # size of validation set given TODO: isn't it better to just give a percentage??
        "train_size": 100,
    }

    def _init(self, conf):
        with open(root / self.conf.img_list, "r") as f:
            self.img_list = f.readlines()
        # Auto-download the dataset if not existing
        if not (DATA_PATH / conf.data_dir).exists():
            self.download_oxford_paris_mini()
        # load image names
        images = self.img_list
        if self.conf.rand_shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(images)
        train_images = images[: conf.train_size]
        val_images = images[conf.train_size: conf.train_size + conf.val_size]
        self.images = {"train": train_images, "val": val_images, "test": images, "all": images}


    def download_oxford_paris_mini(self):
        logger.info("Downloading the OxfordParis Mini dataset...")
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "oxpa_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/"
        num_parts = 100
        # go through dataset parts, one by one and only keep wanted images in img_list
        for i in tqdm(range(num_parts), position=1):
            tar_name = f"revisitop1m.{i+1}.tar.gz"
            tar_url = url_base + "jpg/" + tar_name
            tmp_tar_path = tmp_dir / tar_name
            torch.hub.download_url_to_file(tar_url, tmp_tar_path)
            with tarfile.open(tmp_tar_path) as tar:
                tar.extractall(path=data_dir)
            tmp_tar_path.unlink()
            # Delete unwanted files
            existing_files = set([str(i.relative_to(data_dir)) for i in data_dir.glob("**/*.jpg")])
            to_del = existing_files - set(self.img_list)
            for d in to_del:
                Path(data_dir / d).unlink()

    def get_dataset(self, split):
        assert split in ["train", "val", "test", "all"]
        return _Dataset(self.conf, self.images[split], split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_paths, split):
        self.split = split
        self.conf = conf
        self.grayscale = bool(conf.grayscale)
        # self.conf is set in superclass
        # set img preprocessor
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        # select split scenes
        self.img_dir = DATA_PATH / conf.data_dir

        # Extract image paths
        self.image_paths = image_paths

        # making them relative for system independent names in export files (path used as name in export)
        self.image_paths = [
            i.relative_to(self.img_dir) for i in self.image_paths.copy()
        ]
        if len(self.image_paths) == 0:
            raise ValueError(f"Could not find any image in folder: {self.img_dir}.")
        logger.info(f"NUMBER OF IMAGES: {len(self.image_paths)}")
        # Load features
        if conf.load_features.do:
            self.point_gt_location = DATA_PATH.parent / conf.load_features.point_gt.path
            self.line_gt_location = DATA_PATH.parent / conf.load_features.line_gt.path
            # filter out where missing groundtruth
            new_img_path_list = []
            for img_path in self.image_paths:
                h5_file_name = img_path.with_suffix(".hdf5").name
                point_gt_file_path = (
                    self.point_gt_location / img_path.parent / h5_file_name
                )
                line_gt_file_path = (
                    self.line_gt_location / img_path.parent / h5_file_name
                )
                # perform sanity checks if wanted
                flag = True
                if (
                    self.conf.load_features.check_exists
                    or self.conf.load_features.check_nan
                ):
                    flag = False
                    if self.conf.load_features.check_exists:
                        if point_gt_file_path.exists() and line_gt_file_path.exists():
                            flag = True
                    if self.conf.load_features.check_nan:
                        flag = not self.contains_any_gt_nan(img_path)
                if flag:
                    new_img_path_list.append(img_path)
            self.image_paths = new_img_path_list
            logger.info(f"NUMBER OF IMAGES WITH GT: {len(self.image_paths)}")

    def _read_image(self, path, enforce_batch_dim=False):
        """
        Read image as tensor and puts it on device
        """
        img = load_image(path, grayscale=self.grayscale)
        if enforce_batch_dim:
            if img.ndim < 4:
                img = img.unsqueeze(0)
        assert img.ndim >= 3
        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img

    def _read_groundtruth(self, image_path, enforce_batch_dim=True):
        """
        Reads groundtruth for points and lines from respective h5files.
        We can assume that gt files are existing at this point->filtered in init!

        image_path: path to image as relative to base directory(self.img_path)
        """
        ground_truth = {}
        h5_file_name = image_path.with_suffix(".hdf5").name
        point_gt_file_path = self.point_gt_location / image_path.parent / h5_file_name
        line_gt_file_path = self.line_gt_location / image_path.parent / h5_file_name
        # Read data for points
        with h5py.File(point_gt_file_path, "r") as point_file:
            ground_truth = {
                **self.read_datasets_from_h5(
                    self.conf.load_features.point_gt.data_keys, point_file
                ),
                **ground_truth,
            }
        # Read data for lines
        with h5py.File(line_gt_file_path, "r") as line_file:
            ground_truth = {
                **self.read_datasets_from_h5(
                    self.conf.load_features.line_gt.data_keys, line_file
                ),
                **ground_truth,
            }
        return ground_truth

    def __getitem__(self, idx):
        """
        Dataloader is usually just returning one datapoint by design. Batching is done in Loader normally.
        """
        path = self.image_paths[idx]
        img = self._read_image(self.img_dir / path)
        data = {
            "name": str(path),
            **self.preprocessor(img),
        }  # add metadata, like transform, image_size etc...
        if self.conf.load_features.do:
            gt = self._read_groundtruth(path)
            data = {**data, **gt}

        return data

    def read_datasets_from_h5(self, keys, file):
        data = {}
        for key in keys:
            d = torch.from_numpy(
                np.nan_to_num(file[key].__array__())
            )  # nan_to_num needed because of weird sp gt format
            if self.conf.load_features.device is not None:
                data[key] = d.to(self.conf.load_features.device)
            else:
                data[key] = d
        return data

    def contains_any_gt_nan(self, img_path):
        gt = self._read_groundtruth(img_path)
        for k, v in gt.items():
            if isinstance(v, torch.Tensor) and torch.iszer(v).any():
                return True
        return False

    def __len__(self):
        return len(self.image_paths)
