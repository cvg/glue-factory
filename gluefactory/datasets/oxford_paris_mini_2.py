import logging
import pickle
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import os

from gluefactory.datasets import BaseDataset
from gluefactory.settings import DATA_PATH, root
from gluefactory.utils.image import ImagePreprocessor, load_image, read_image
from gluefactory.utils.tensor import batch_to_device

logger = logging.getLogger(__name__)


class OxfordParisMini(BaseDataset):
    """
    Subset of the Oxford Paris dataset as defined here: https://cmp.felk.cvut.cz/revisitop/
    Supports loading groundtruth and only serves images for that gt exists.
    Dataset only deals with loading one element. Batching is done by Dataloader!

    Adapted to use POLD2 structure of files -> Files and gt in same folder besides each other
    Some facts:
    - Pold2 gt is generated same size as original image and can be resized
    """

    default_conf = {
        "data_dir": "revisitop1m/jpg",  # as subdirectory of DATA_PATH(defined in settings.py)
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
        "reshape": None, # ex [800, 800]
        "load_features": {
            "do": False,
            "check_exists": True,
            "point_gt": {
                "data_keys": ["superpoint_heatmap"],
            },
            "line_gt": {
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
            },
        },
        "img_list": "gluefactory/datasets/oxford_paris_images.txt",  # path to image list containing images we want to use for this dataset to be consitent (given from repo root)
        "rand_shuffle_seed": None,  # seed to randomly shuffle before split in train and val
        "val_size": 10,  # size of validation set given
        "train_size": 100000,
    }

    def _init(self, conf):
        with open(root / self.conf.img_list, "r") as f:
            self.img_list = [file_name.strip("\n") for file_name in f.readlines()]
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

        shutil.rmtree(tmp_dir)

        #remove empty directories
        for file in os.listdir(data_dir):
            cur_file: Path = data_dir / file
            if cur_file.is_file():
                continue
            if len(os.listdir(cur_file)) == 0:
                shutil.rmtree(cur_file)

    def get_dataset(self, split):
        assert split in ["train", "val", "test", "all"]
        return _Dataset(self.conf, self.images[split], split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_paths: list[str], split):
        self.split = split
        self.conf = conf
        self.grayscale = bool(conf.grayscale)
        # self.conf is set in superclass
        # set img preprocessor
        if self.conf.reshape is not None:
            reshape_preprocessing = f"resize: {self.conf.reshape}"
            self.preprocessor = ImagePreprocessor(reshape_preprocessing)

        # select split scenes
        self.img_dir = DATA_PATH / conf.data_dir

        # Extract image paths
        self.image_paths = [Path(i) for i in image_paths]

        # making them relative for system independent names in export files (path used as name in export)
        if len(self.image_paths) == 0:
            raise ValueError(f"Could not find any image in folder: {self.img_dir}.")
        logger.info(f"NUMBER OF IMAGES: {len(self.image_paths)}")
        # Load features
        if conf.load_features.do:
            # filter out where missing groundtruth
            new_img_path_list = []
            for img_path in self.image_paths:
                if not self.conf.load_features.check_exists:
                    new_img_path_list.append(img_path)
                    continue
                # perform checks TODO: Add offsets etc?
                img_folder = img_path.parent
                keypoint_file = img_folder / "keypoint_scores.npy"
                af_file = img_folder / "af_scores.npy"
                df_file = img_folder / "df_scores.npy"

                if keypoint_file.exists() and af_file.exists() and df_file.exists():
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

    def _read_groundtruth(self, image_path):
        """
        Reads groundtruth for points and lines from respective files
        We can assume that gt files are existing at this point->filtered in init!

        image_path: path to image as relative to base directory(self.img_path)
        """
        features = {}
        kp_file = image_path.parent / "keypoints.npy"
        kps_file = image_path.parent / "keypoint_scores.npy"

        # Load keypoints and scores
        kp = torch.from_numpy(np.load(kp_file)).to(dtype=torch.float32)
        if self.conf.reshape is not None:
            kp = self.preprocessor(kp)
        features["keypoints"] = kp

        kps = torch.from_numpy(np.load(kps_file)).to(dtype=torch.float32)
        if self.conf.reshape is not None:
            kps = self.preprocessor(kps)
        features["keypoint_scores"] = kps
        features = batch_to_device(features, self.conf.device)


        # Load pickle file for DF max and min values
        with open(image_path.parent / "values.pkl", "rb") as f:
            values = pickle.load(f)

        # Load DF
        df_img = read_image(image_path.parent / "df.jpg", True)
        df_img = df_img.astype(np.float32) / 255.0
        df_img *= values["max_df"]

        # Load AF
        af_img = read_image(image_path.parent / "angle.jpg", True)
        af_img = af_img.astype(np.float32) / 255.0
        af_img *= np.pi

        # Get closest point to line for each pixel
        # offset = self.df_and_angle_to_offset(df_img, af_img)
        ofx_img = read_image(image_path.parent / "offset_x.jpg", True)
        ofx_img = ofx_img.astype(np.float32) / 255.0
        ofy_img = read_image(image_path.parent / "offset_y.jpg", True)
        ofy_img = ofy_img.astype(np.float32) / 255.0
        offset = np.stack((ofx_img, ofy_img), axis=-1)
        offset = offset * values["max_offset"]
        offset = offset + values["min_offset"]

        df = torch.from_numpy(df_img).to(dtype=torch.float32)
        if self.conf.reshape is not None:
            df = self.preprocessor(df)
        features[self.conf.load_features.line_gt[0]] = df
        af = torch.from_numpy(af_img).to(dtype=torch.float32)
        if self.conf.reshape is not None:
            af = self.preprocessor(af)
        features[self.conf.load_features.line_gt[1]] = af

        offset = torch.from_numpy(offset).to(dtype=torch.float32)
        if self.conf.reshape is not None:
            offset = self.preprocessor(offset)
        features["offset"] = offset

        features = batch_to_device(features, self.conf.device)

        return features

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


    def __len__(self):
        return len(self.image_paths)
