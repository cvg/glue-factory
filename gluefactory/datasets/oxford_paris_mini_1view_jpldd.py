import logging
import os
import pickle
import random
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gluefactory.datasets import BaseDataset
from gluefactory.settings import DATA_PATH, root
from gluefactory.utils.image import load_image, read_image, ImagePreprocessor

logger = logging.getLogger(__name__)


class OxfordParisMiniOneViewJPLDD(BaseDataset):
    """
    Subset of the Oxford Paris dataset as defined here: https://cmp.felk.cvut.cz/revisitop/
    Supports loading groundtruth and only serves images for that gt exists.
    Dataset only deals with loading one element. Batching is done by Dataloader!

    Adapted to use POLD2 structure of files -> Files and gt in same folder besides each other
    Some facts:
    - Pold2 gt is generated same size as original image and can be resized
    """

    default_conf = {
        "data_dir": "revisitop1m_POLD2/jpg",  # as subdirectory of DATA_PATH(defined in settings.py)
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
        "square_pad": True,  # square padding is needed to batch together images with current scaling technique(keeping aspect ratio). Can and should be deactivated on benchmarks
        "multiscale_learning": {
            "do": False,
            "scales_list": [1000, 800, 600, 400],  # use interger scales to have resize keep aspect ratio -> not squashing img by forcing it to square
            "scale_selection": 'random' # random or round-robin
        },
        "load_features": {
            "do": False,
            "check_exists": True,
            "point_gt": {
                "data_keys": ["superpoint_heatmap"],
                "use_score_heatmap": True,
            },
            "line_gt": {
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
                "enforce_threshold": 5.0,  # Enforce values in distance field to be no greater than this value
            },
        },
        "img_list": "gluefactory/datasets/oxford_paris_images.txt",
        # img list path from repo root -> use checked in file list, it is similar to pold2 file
        "rand_shuffle_seed": None,  # seed to randomly shuffle before split in train and val
        "val_size": 10,  # size of validation set given
        "train_size": 100000,
        "debug": False,  # if True also gives back original keypoint gt locations
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
        val_images = images[conf.train_size : conf.train_size + conf.val_size]
        self.images = {
            "train": train_images,
            "val": val_images,
            "test": images,
            "all": images,
        }
        print(f"DATASET OVERALL(NO-SPLIT) IMAGES: {len(images)}")

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
            tar_name = f"revisitop1m.{i + 1}.tar.gz"
            tar_url = url_base + "jpg/" + tar_name
            tmp_tar_path = tmp_dir / tar_name
            torch.hub.download_url_to_file(tar_url, tmp_tar_path)
            with tarfile.open(tmp_tar_path) as tar:
                tar.extractall(path=data_dir)
            tmp_tar_path.unlink()
            # Delete unwanted files
            existing_files = set(
                [str(i.relative_to(data_dir)) for i in data_dir.glob("**/*.jpg")]
            )
            to_del = existing_files - set(self.img_list)
            for d in to_del:
                Path(data_dir / d).unlink()

        shutil.rmtree(tmp_dir)

        # remove empty directories
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
    def __init__(self, conf, image_sub_paths: list[str], split):
        super().__init__()
        self.split = split
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

        self.img_dir = DATA_PATH / conf.data_dir
        # Extract image paths
        self.image_sub_paths = image_sub_paths  # [Path(i) for i in image_sub_paths]

        # making them relative for system independent names in export files (path used as name in export)
        if len(self.image_sub_paths) == 0:
            raise ValueError(f"Could not find any image in folder: {self.img_dir}.")
        logger.info(f"NUMBER OF IMAGES: {len(self.image_sub_paths)}")
        logger.info(f"KNOWN BATCHSIZE FOR MY SPLIT({self.split}) is {self.relevant_batch_size}")
        # Load features
        if conf.load_features.do:
            # filter out where missing groundtruth
            new_img_path_list = []
            for img_path in self.image_sub_paths:
                if not self.conf.load_features.check_exists:
                    new_img_path_list.append(img_path)
                    continue
                # perform checks
                full_artificial_img_path = Path(self.img_dir / img_path)
                img_folder = (
                    full_artificial_img_path.parent / full_artificial_img_path.stem
                )
                keypoint_file = img_folder / "keypoint_scores.npy"
                af_file = img_folder / "angle.jpg"
                df_file = img_folder / "df.jpg"

                if keypoint_file.exists() and af_file.exists() and df_file.exists():
                    new_img_path_list.append(img_path)

            self.image_sub_paths = new_img_path_list
            logger.info(f"NUMBER OF IMAGES WITH GT: {len(self.image_sub_paths)}")

    def register_image_preprocessor_for_size(self, size: int) -> None:
        """
        We use image preprocessor to reshape images and square pad them. We resize keeping the aspect ratio of images.
        Thus image sizes can be different even when long side scaled to same length. Thus square padding is needed so that
        all images can be stuck together in a batch.
        """
        self.preprocessors[size] = ImagePreprocessor({
                                                        "resize": size,
                                                        "edge_divisible_by": None,
                                                        "side": "long",
                                                        "interpolation": "bilinear",
                                                        "align_corners": None,
                                                        "antialias": True,
                                                        "square_pad": bool(self.conf.square_pad),
                                                        "add_padding_mask": True,}
                                                     )

    def _read_image(self, img_folder_path, enforce_batch_dim=False):
        """
        Read image as tensor and puts it on device
        """
        img_path = img_folder_path / "base_image.jpg"
        img = load_image(img_path, grayscale=self.grayscale)
        if enforce_batch_dim:
            if img.ndim < 4:
                img = img.unsqueeze(0)
        assert img.ndim >= 3
        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img

    def _read_groundtruth(self, image_folder_path, original_img_size: tuple, shape: int = None) -> dict:
        """
        Reads groundtruth for points and lines from respective files
        We can assume that gt files are existing at this point->filtered in init!

        image_folder_path: full image folder path to get the gt data from
        original_img_size: format w,h original image size to be able to create heatmaps.
        shape: shape to reshape img to if it is not None
        """
        w, h = original_img_size
        features = {}
        kp_file = image_folder_path / "keypoints.npy"
        kps_file = image_folder_path / "keypoint_scores.npy"

        # Load keypoints and scores
        orig_kp = torch.from_numpy(np.load(kp_file)).to(dtype=torch.float32)
        integer_kp = torch.round(orig_kp).to(dtype=torch.int)

        kps = torch.from_numpy(np.load(kps_file)).to(dtype=torch.float32)

        heatmap = np.zeros((h, w))
        coordinates = integer_kp
        if self.conf.load_features.point_gt.use_score_heatmap:
            heatmap[coordinates[:, 1], coordinates[:, 0]] = kps
        else:
            heatmap[coordinates[:, 1], coordinates[:, 0]] = 1.0
        heatmap = torch.from_numpy(heatmap).to(dtype=torch.float32)

        if shape is not None:
            heatmap = self.preprocessors[shape](heatmap.unsqueeze(0))['image'].squeeze(0)  # only store image here as padding map will be stored by preprocessing image

        if self.conf.debug:
            non_zero_coord = torch.nonzero(heatmap)
            features["orig_points"] = non_zero_coord[:][:, torch.tensor([1, 0])]

        features[self.conf.load_features.point_gt.data_keys[0]] = heatmap

        # Load pickle file for DF max and min values
        with open(image_folder_path / "values.pkl", "rb") as f:
            values = pickle.load(f)

        # Load DF
        df_img = read_image(image_folder_path / "df.jpg", True)
        df_img = df_img.astype(np.float32) / 255.0
        df_img *= values["max_df"]
        thres = self.conf.load_features.line_gt.enforce_threshold
        if thres is not None:
            df_img = np.where(df_img > thres, thres, df_img)

        # Load AF
        af_img = read_image(image_folder_path / "angle.jpg", True)
        af_img = af_img.astype(np.float32) / 255.0
        af_img *= np.pi

        df = torch.from_numpy(df_img).to(dtype=torch.float32)
        if shape is not None:
            df = self.preprocessors[shape](df.unsqueeze(0))['image'].squeeze(0)  # only store image here as padding map will be stored by preprocessing image
        features[self.conf.load_features.line_gt.data_keys[0]] = df
        af = torch.from_numpy(af_img).to(dtype=torch.float32)
        if shape is not None:
            af = self.preprocessors[shape](af.unsqueeze(0))['image'].squeeze(0)  # only store image here as padding map will be stored by preprocessing image
        features[self.conf.load_features.line_gt.data_keys[1]] = af

        return features

    def __getitem__(self, idx):
        """
        Dataloader is usually just returning one datapoint by design. Batching is done in Loader normally.
        """
        full_artificial_img_path = self.img_dir / self.image_sub_paths[idx]
        folder_path = full_artificial_img_path.parent / full_artificial_img_path.stem
        img = self._read_image(folder_path)
        orig_shape = img.shape[-1], img.shape[-2]
        size_to_reshape_to = self.select_resize_shape(orig_shape)
        data = {
            "name": str(folder_path / "base_image.jpg"),
        }  # keys: 'name', 'scales', 'image_size', 'transform', 'original_image_size', 'image'
        if size_to_reshape_to == orig_shape:
            data['image'] = img
        else:
            data = {**data, **self.preprocessors[size_to_reshape_to](img)}
        if self.conf.load_features.do:
            gt = self._read_groundtruth(folder_path, orig_shape, None if size_to_reshape_to == orig_shape else size_to_reshape_to)
            data = {**data, **gt}
        return data
    
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


    def __len__(self):
        return len(self.image_sub_paths)
