import logging
import random
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

from gluefactory.datasets import BaseDataset, augmentations
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
        "data_dir": "minidepth",  # as subdirectory of DATA_PATH(defined in settings.py)
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
        "square_pad": False,
        "multiscale_learning": {
            "do": False,
            "scales_list": [1000, 800, 600, 400],
            "scale_selection": "random",  # random or round-robin
        },
        "load_features": {
            "do": False,
            "check_exists": True,
            "device": None,  # choose device to move groundtruthdata to if None is given, just read, skip move to device
            "point_gt": {
                "data_keys": [
                    "superpoint_heatmap",
                    "gt_keypoints",
                    "gt_keypoints_scores",
                ],  # heatmap is generated based on keypoints
                "load_points": False,
                "use_score_heatmap": False,
                "max_num_heatmap_keypoints": -1,  # topk keypoints used to create the heatmap (-1 = all are used)
                "max_num_keypoints": 76,  # topk keypoints returned as gt keypoint locations (-1 - return all)
                # -> Can also be set to None to return all points but this can only be used when batchsize=1. Min num kp in minidepth:  76
                "use_deeplsd_lineendpoints_as_kp_gt": False,  # set true to use deep-lsd line endpoints as keypoint groundtruth
                "use_superpoint_kp_gt": True,  # set true to use default HA-Superpoint groundtruth
            },
            "line_gt": {
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
                "enforce_threshold": 5.0,
            },
            "augment": {
                # there is the option to use data augmentation. It is not enlarging dataset but applies the augmentation to an Image with certain probability
                "do": False,
                "type": "dark",  # choose "identity" for no augmentation; other options are "lg", "dark"
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
        # prepare augmentation in case it is needed
        augmentation_map = {
            "dark": augmentations.DarkAugmentation,
            "lg": augmentations.LGAugmentation,
            "identity": augmentations.IdentityAugmentation,
        }
        self.augmentation = augmentation_map[self.conf.load_features.augment.type]()

    def download_minidepth(self):
        logger.info("Downloading the MiniDepth dataset...")
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "minidepth_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url = "https://filedn.com/lt6zb4ORSwapNyVniJf1Pqh/JPL/minidepth.zip"
        zip_name = "minidepth.zip"
        zip_path = tmp_dir / zip_name
        torch.hub.download_url_to_file(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        shutil.move(tmp_dir / zip_name.split(".")[0], str(data_dir))

    def get_dataset(self, split):
        assert split in ["train", "val", "test", "all"]
        return _Dataset(self.conf, split, self.augmentation)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, augmentation):
        super().__init__()
        self.conf = conf
        self.grayscale = bool(conf.grayscale)
        self.max_num_gt_kp = conf.load_features.point_gt.max_num_keypoints
        self.augmentation = augmentation

        # we can configure whether we want to load superpoint kp-gt, deeplsd_lineEP lp-gt or both
        self.use_superpoint_kp_gt = conf.load_features.point_gt.use_superpoint_kp_gt
        self.use_dlsd_ep_as_kp_gt = (
            conf.load_features.point_gt.use_deeplsd_lineendpoints_as_kp_gt
        )

        # Initialize Image Preprocessors for square padding and resizing
        self.preprocessors = {}  # stores preprocessor for each reshape size
        if self.conf.reshape is not None:
            self.register_image_preprocessor_for_size(self.conf.reshape)
        if self.conf.multiscale_learning.do:
            for scale in self.conf.multiscale_learning.scales_list:
                self.register_image_preprocessor_for_size(scale)

        if self.conf.multiscale_learning.do:
            if self.conf.multiscale_learning.scale_selection == "round-robin":
                self.scale_selection_idx = 0
            # Keep track uf how many selected with current scale for batching (all img in same batch need same size)
            self.num_select_with_current_scale = 0
            self.current_scale = None
            # we need to make sure that the appropriate batch size for the dataset conf is set correctly.
            self.relevant_batch_size = self.conf[f"{split}_batch_size"]

        # select split scenes
        self.img_dir = DATA_PATH / conf.data_dir / "images"
        self.line_gt_dir = DATA_PATH / conf.data_dir / "deeplsd_gt"
        self.point_gt_dir = DATA_PATH / conf.data_dir / "keypoint_gt"
        self.dlsd_kp_gt = DATA_PATH / conf.data_dir / "dlsd_keypoint_gt"
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
            # filter out where missing groundtruth
            new_img_path_list = []
            for img_path in self.image_paths:
                h5_file_name = img_path.with_suffix(".hdf5").name
                point_gt_file_path = (
                    self.point_gt_dir
                    / img_path.parent
                    / img_path.stem
                    / "keypoints.npy"
                )
                line_gt_file_path = self.line_gt_dir / img_path.parent / h5_file_name
                dlsd_kp_gt_file = (
                    self.dlsd_kp_gt / img_path.parent / f"{img_path.stem}.npy"
                )
                # perform sanity checks if wanted
                flag = True
                if self.conf.load_features.check_exists:
                    flag = False
                    if self.conf.load_features.check_exists:
                        if point_gt_file_path.exists() and line_gt_file_path.exists():
                            if self.use_dlsd_ep_as_kp_gt:
                                flag = dlsd_kp_gt_file.exists()
                            else:
                                flag = True
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

    def register_image_preprocessor_for_size(self, size: int) -> None:
        """
        We use image preprocessor to reshape images and square pad them. We resize keeping the aspect ratio of images.
        Thus image sizes can be different even when long side scaled to same length. Thus square padding is needed so that
        all images can be stuck together in a batch.
        """
        self.preprocessors[size] = ImagePreprocessor(
            {
                "resize": size,
                "edge_divisible_by": None,
                "side": "long",
                "interpolation": "bilinear",
                "align_corners": None,
                "antialias": True,
                "square_pad": bool(self.conf.square_pad),
                "add_padding_mask": True,
            }
        )

    def _read_groundtruth(self, image_path, shape: int = None) -> dict:
        """
        Reads groundtruth for points and lines from respective h5files.
        We can assume that gt files are existing at this point->filtered in init!

        image_path: path to image as relative to base directory(self.img_path)
        """
        # Load config and local variables
        reshape_scales = None
        reshaped_size_unpadded = None
        heatmap_gt_key_name = self.conf.load_features.point_gt.data_keys[0]
        kp_gt_key_name = self.conf.load_features.point_gt.data_keys[1]
        kp_score_gt_key_name = self.conf.load_features.point_gt.data_keys[2]
        df_gt_key = self.conf.load_features.line_gt.data_keys[0]
        af_gt_key = self.conf.load_features.line_gt.data_keys[1]

        ground_truth = {}
        h5_file_name = image_path.with_suffix(".hdf5").name
        npy_file_subpath = image_path.parent / image_path.stem / "keypoints.npy"
        point_gt_file_path = self.point_gt_dir / npy_file_subpath
        line_gt_file_path = self.line_gt_dir / image_path.parent / h5_file_name
        dlsd_kp_gt_path = self.dlsd_kp_gt / image_path.parent / f"{image_path.stem}.npy"

        # Read data for lines -> stored as tensors
        with h5py.File(line_gt_file_path, "r") as line_file:
            ground_truth = {
                **self.read_datasets_from_h5(
                    self.conf.load_features.line_gt.data_keys, line_file
                ),
                **ground_truth,
            }
        # threshold df if wanted
        df_img = ground_truth[df_gt_key]
        thres = self.conf.load_features.line_gt.enforce_threshold
        if thres is not None:
            df_img = np.where(df_img > thres, thres, df_img)
        df = torch.from_numpy(df_img).to(dtype=torch.float32)
        original_size = df.shape

        af_img = ground_truth[af_gt_key]
        af = af_img.to(dtype=torch.float32)
        # reshape df and af if wanted
        if shape is not None:
            preprocessor_df_out = self.preprocessors[shape](df.unsqueeze(0))
            reshape_scales = preprocessor_df_out[
                "scales"
            ]  # store reshape scales for keypoints later
            reshaped_size_unpadded = preprocessor_df_out["image_size"]
            df = preprocessor_df_out["image"].squeeze(0)
            af = self.preprocessors[shape](af.unsqueeze(0))["image"].squeeze(0)
        ground_truth[af_gt_key] = af
        ground_truth[df_gt_key] = df

        # Read data for points
        keypoints = None
        keypoint_scores = None
        if self.use_superpoint_kp_gt:
            kp_file_content = torch.from_numpy(np.load(point_gt_file_path)).to(
                dtype=torch.float32
            )  # file contains (N, 3) shape np-array -> 1st two cols for kp x,y 3rd for kp-score
            keypoints = kp_file_content[:, [1, 0]]
            keypoint_scores = kp_file_content[:, 2]
        if self.use_dlsd_ep_as_kp_gt:
            # need to clamp values as deeplsd also predicts line endpoints outside the image
            dlsd_kp_gt = torch.from_numpy(np.load(dlsd_kp_gt_path)).to(
                dtype=torch.float32
            )
            dlsd_kp_gt = torch.stack(
                [
                    torch.clamp(dlsd_kp_gt[:, 0], min=0, max=(original_size[1] - 1)),
                    torch.clamp(dlsd_kp_gt[:, 1], min=0, max=(original_size[0] - 1)),
                ],
                dim=1,
            )
            keypoints = (
                torch.vstack([keypoints, dlsd_kp_gt[:, [0, 1]]])
                if keypoints is not None
                else dlsd_kp_gt[:, [0, 1]]
            )
            # for dlsd line endpoints no scores are given thus setting all to one (recommend to not use score heatmap)
            keypoint_scores = (
                torch.hstack([keypoint_scores, torch.ones((dlsd_kp_gt.shape[0]))])
                if keypoint_scores is not None
                else torch.ones((dlsd_kp_gt.shape[0]))
            )

        # scale points and create integer coordinates for heatmap
        heatmap = np.zeros_like(df)
        keypoints = ((keypoints * reshape_scales) if reshape_scales is not None else keypoints)
        coordinates = torch.round(keypoints).to(dtype=torch.int)
        if self.conf.load_features.point_gt.max_num_heatmap_keypoints > 0:
            num_selected_kp = min(
                [self.conf.load_features.point_gt.max_num_heatmap_keypoints, keypoint_scores.shape[0]])
            coordinates = coordinates[:num_selected_kp]
        if reshaped_size_unpadded is not None:
            # if reshaping is done clamp of roundend coordinates is necessary (TODO: possibly only if dlsd line ep kp gt is used)
            coordinates = torch.stack(
                [
                    torch.clamp(
                        coordinates[:, 0], min=0, max=(reshaped_size_unpadded[0] - 1)
                    ),
                    torch.clamp(
                        coordinates[:, 1], min=0, max=(reshaped_size_unpadded[1] - 1)
                    ),
                ],
                dim=1,
            )
        # create heatmap
        if self.conf.load_features.point_gt.use_score_heatmap:
            heatmap[coordinates[:, 1], coordinates[:, 0]] = keypoint_scores
        else:
            heatmap[coordinates[:, 1], coordinates[:, 0]] = 1.0
        heatmap = torch.from_numpy(heatmap).to(dtype=torch.float32)

        ground_truth[heatmap_gt_key_name] = heatmap
        # choose max num keypoints if wanted. Attention: we dont sort by score here! If dlsd kp gt is used all scores of these are 1!
        if self.conf.load_features.point_gt.load_points:
            num_selected_kp = min([self.max_num_gt_kp, keypoint_scores.shape[0]])
            ground_truth[kp_gt_key_name] = (
                keypoints[: num_selected_kp, :]
                if self.max_num_gt_kp > -1
                else keypoints
            )
            ground_truth[kp_score_gt_key_name] = (
                keypoint_scores[: num_selected_kp]
                if self.max_num_gt_kp > -1
                else keypoint_scores
        )

        return ground_truth

    def __getitem__(self, idx):
        """
        Dataloader is usually just returning one datapoint by design. Batching is done in Loader normally.
        """
        path = self.image_paths[idx]
        img = self._read_image(self.img_dir / path)
        if self.conf.load_features.augment.do:
            try:
                img = img.numpy().transpose(1, 2, 0)
                img = self.augmentation(image=img, return_tensor=True)
            except Exception as e:
                logging.error(f"Error in augmentation: {e}")
        orig_shape = img.shape[-1], img.shape[-2]
        size_to_reshape_to = self.select_resize_shape(orig_shape)
        data = {
            "name": str(path),
        }  # add metadata, like transform, image_size etc...
        if size_to_reshape_to == orig_shape:
            data["image"] = img
        else:
            data = {**data, **self.preprocessors[size_to_reshape_to](img)}
        if self.conf.load_features.do:
            gt = self._read_groundtruth(
                path, None if size_to_reshape_to == orig_shape else size_to_reshape_to
            )
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

    def __len__(self):
        return len(self.image_paths)

    def do_change_size_now(self) -> bool:
        """
        Based on current state decides whether to change shape to reshape images to.
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
                assert (
                    len(scales_list) > 1
                )  # need more than one scale for multiscale learning to make sense

                if scale_selection == "random":
                    choice = int(random.choice(scales_list))
                    self.current_scale = choice
                    return choice
                elif scale_selection == "round-robin":
                    current_scale = scales_list[self.scale_selection_idx]
                    self.current_scale = current_scale
                    self.scale_selection_idx += 1
                    self.scale_selection_idx = self.scale_selection_idx % len(
                        scales_list
                    )
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
