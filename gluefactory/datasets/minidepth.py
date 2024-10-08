import logging
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

from gluefactory.datasets import BaseDataset
from gluefactory.settings import DATA_PATH, root
from gluefactory.utils.image import ImagePreprocessor, load_image

logger = logging.getLogger(__name__)


class MiniDepthDataset(BaseDataset):
    """
    Assumes minidepth dataset in folder as jpg images.
    Supports loading groundtruth and only serves images for that gt exists.
    Dataset only deals with loading one element. Batching is done by Dataloader!

    This class only used to load conf and autodownload. For usable Datasets, use get_dataset to get Dataset with split
    """

    default_conf = {
        "data_dir": "minidepth/images",  # as subdirectory of DATA_PATH(defined in settings.py)
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
                "path": "outputs/results/superpoint_gt/minidepth",
                "data_keys": ["superpoint_heatmap"],
            },
            "line_gt": {
                "path": "outputs/results/deeplsd_gt/minidepth",
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
            },
        },
        "train_scenes_file_path": "gluefactory/datasets/minidepth_train_scenes.txt",  # path to training scenes file where train scenes from megadepth1500 are excluded, based on repo root
        "val_scenes_file_path": "gluefactory/datasets/minidepth_val_scenes.txt",
    }

    def _init(self, conf):
        # Auto-download the dataset if not existing
        if not (DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the minidepth dataset...")
            self.download_minidepth()

    def download_minidepth(self):
        logger.info("Downloading the MiniDepth dataset...")
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "minidepth_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://filedn.com/lt6zb4ORSwapNyVniJf1Pqh/"
        zip_name = "minidepth.zip"
        zip_path = tmp_dir / zip_name
        torch.hub.download_url_to_file(url_base + zip_name, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        shutil.move(tmp_dir / zip_name.split(".")[0], data_dir.split("/")[0])

    def get_dataset(self, split):
        assert split in ["train", "val", "test", "all"]
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.conf = conf
        self.grayscale = bool(conf.grayscale)
        # self.conf is set in superclass
        # set img preprocessor
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        # select split scenes
        self.img_dir = DATA_PATH / conf.data_dir
        scene_file_path = self.img_dir.parent
        # Extract the scenes corresponding to the right split
        scenes_file = None
        if split == "train":
            scenes_file = root / conf.train_scenes_file_path
        elif split == "val":
            scenes_file = root / conf.val_scenes_file_path
        else:
            # select all images if 'all' or 'test' given
            scenes_file = None

        # Extract image paths
        self.image_paths = []
        if scenes_file is not None:
            with open(scenes_file, "r") as f:
                self.scenes = [line.strip("\n") for line in f.readlines()]
            for s in self.scenes:
                scene_folder = self.img_dir / s
                self.image_paths += list(Path(scene_folder).glob("**/*.jpg"))
        else:
            self.image_paths += list(Path(self.img_dir).glob("**/*.jpg"))

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

    def get_dataset(self, split):
        return self

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
