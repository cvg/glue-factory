import logging
import shutil
import random
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

from gluefactory.datasets import BaseDataset
from gluefactory.settings import DATA_PATH, root
from gluefactory.utils.image import load_image
from gluefactory.datasets.utils import resize_img_kornia

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
        "reshape": None,  # ex 800  # if reshape is activated AND multiscale learning is activated -> reshape has prevalence
        "multiscale_learning": {
            "do": False,
            "scales_list": [1000, 800, 600, 400],
            "scale_selection": 'random' # random or round-robin
        },
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
        super().__init__()
        self.conf = conf
        self.grayscale = bool(conf.grayscale)

        # Initialize Image Preprocessors for square padding and resizing
        self.preprocessors = {} # stores preprocessor for each reshape size
        if self.conf.reshape is not None:
            self.register_image_preprocessor_for_size(self.conf.reshape)
        if self.conf.multiscale_learning.do:
            for scale in self.conf.multiscale_learning.scales_list:
                self.register_image_preprocessor_for_size(scale)

        if self.conf.multiscale_learning.do:
            if self.conf.multiscale_learning.scale_selection == 'round-robin':
                self.scale_selection_idx = 0
            # Keep track uf how many selected with current scale for batching (all img in same batch need same size)
            self.num_select_with_current_scale = 0
            self.current_scale = None
            # we need to make sure that the appropriate batch size for the dataset conf is set correctly.
            self.relevant_batch_size = self.conf[f"{split}_batch_size"]

        # select split scenes
        self.img_dir = DATA_PATH / conf.data_dir
        # Extract the scenes corresponding to the right split
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

    def _read_groundtruth(self, image_path, original_img_size: tuple, shape: int = None) -> dict:
        """
        Reads groundtruth for points and lines from respective h5files.
        We can assume that gt files are existing at this point->filtered in init!

        image_path: path to image as relative to base directory(self.img_path)
        """
        # TODO: implement reshape of gt here once gt is generated and format is clear (use preprocessors for padding as well)
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
        orig_shape = img.shape[-1], img.shape[-2]
        size_to_reshape_to = self.select_resize_shape(orig_shape)
        data = {
            "name": str(path),
            "image": img if size_to_reshape_to == orig_shape else self.preprocessors[size_to_reshape_to](img),
        }  # add metadata, like transform, image_size etc...
        if self.conf.load_features.do:
            gt = self._read_groundtruth(path, orig_shape, None if size_to_reshape_to == orig_shape else size_to_reshape_to)
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
    
    
    def do_change_size_now(self) -> bool:
        """
        Based on current state descides whether to change shape to reshape images to.
        This decision is needed as all images in a batch need same shape. So we only potentially change shape
        when a new batch is starting.

        Returns:
            bool: should shape be potentially changed?
        """
        # check if batch changes
        if self.num_select_with_current_scale % self.relevant_batch_size == 0:
            self.num_select_with_current_scale = 0  # if batch changes set counter to 0
            return True
        else:
            return False
        

    def select_resize_shape(self, original_img_size: tuple):
        """
        Depending on whether resize or multiscale learning is activated the shape to resize the
        image to is returned. If none of it is activated, the original image size will be returned.
        Reshape has prevalence over multiscale learning!
        """
        do_reshape = self.conf.reshape is not None
        do_ms_learning = self.conf.multiscale_learning.do
        if not do_reshape and not do_ms_learning:
            return original_img_size

        if do_reshape:
            return int(self.conf.reshape)

        if do_ms_learning:
            if self.do_change_size_now():
                self.num_select_with_current_scale += 1
                scales_list = self.conf.multiscale_learning.scales_list
                scale_selection = self.conf.multiscale_learning.scale_selection
                assert len(scales_list) > 1 # need more than one scale for multiscale learning to make sense

                if scale_selection == "random":
                    choice = int(random.choice(scales_list))
                    self.current_scale = choice
                    return choice
                elif scale_selection == "round-robin":
                    current_scale = scales_list[self.scale_selection_idx]
                    self.current_scale = current_scale
                    self.scale_selection_idx += 1
                    self.scale_selection_idx = self.scale_selection_idx % len(scales_list)
                    return int(current_scale)
            else:
                self.num_select_with_current_scale += 1
                return self.current_scale

        raise Exception("Shouldn't end up here!")
    
    
    def set_num_selected_with_current_scale(self, value: int) -> None:
        """
        Sets the self.num_selected_with_current_scale variable to a certain value.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets

        Args:
            value (int): new value for variable
        """
        self.num_select_with_current_scale = value


    def get_current_scale(self) -> int:
        """
        Returns the current used scale to reshape images to. Returns None if multiscale learning is deactivated.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets

        Returns:
            int: current scale used to reshape in multi-scale training. None if its deactivated
        """
        return self.current_scale
    
    
    def set_current_scale(self, value):
        """
        Sets the current scale used for multiscale training. Used to set size of reshape of this dataset during batch.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets.

        Returns:
            int: current scale used to reshape in multi-scale training. None if its deactivated
        """
        self.current_scale = value
