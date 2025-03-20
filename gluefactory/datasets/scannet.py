import logging
import math
import random
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

from gluefactory.datasets import BaseDataset
from gluefactory.models.extractors.superpoint import top_k_keypoints
from gluefactory.settings import DATA_PATH, root
from gluefactory.utils.image import ImagePreprocessor, load_image

logger = logging.getLogger(__name__)
DOWNLOAD_URL = "https://polybox.ethz.ch/index.php/s/N3J7SfnE6NEt8Nu/download"


class Scannet(BaseDataset):
    """
    Subset of the full scannet dataset used as dataset for indoor scenes training. The dataset excludes scenes used
    in the scannet-1500 evaluation and samples images from each scene until we reach 12000 images (as oxparis mini).
    """

    default_conf = {
        "data_dir": "scannet",  # as subdirectory of DATA_PATH(defined in settings.py)
        "grayscale": False,
        "train_batch_size": 2,  # prefix must match split
        "test_batch_size": 1,
        "val_batch_size": 1,
        "all_batch_size": 1,
        "split": "train",  # train, val
        "seed": 0,
        "device": None,
        "num_workers": 0,  # number of workers used by the Dataloader
        "prefetch_factor": None,
        "reshape": None,  # ex 800  # if reshape is activated AND multiscale learning is activated -> reshape has prevalence
        "square_pad": True,  # square padding is needed to batch together images with current scaling technique(keeping aspect ratio). Can and should be deactivated on benchmarks
        "multiscale_learning": {
            "do": False,
            "scales_list": [
                1000,
                800,
                600,
                400,
            ],  # use interger scales to have resize keep aspect ratio -> not squashing img by forcing it to square
            "scale_selection": "random",  # random or round-robin
        },
        "load_features": {
            "do": False,
            "check_exists": True,
            "point_gt": {
                "data_keys": [
                    "superpoint_heatmap",
                    "gt_keypoints",
                    "gt_keypoints_scores",
                ],
                "load_points": False,  # load keypoint locations separately and return then as list (heatmap constructed from keypoints is loaded anyway)
                "max_num_keypoints": 63,  # topk keypoints returned as gt keypoint locations (-1 - return all)
                "use_score_heatmap": False,  # the heatmap created from gt kypoints gets probability value assigned instead of 1.0
                "max_num_heatmap_keypoints": -1,  # topk keypoints used to create the heatmap (-1 = all are used)
                # -> Can also be set to None to return all points but this can only be used when batchsize=1. Min num kp in oxparis: 63
            },
            "line_gt": {
                "load_lines": False,
                "data_keys": [
                    "deeplsd_distance_field",
                    "deeplsd_angle_field",
                    "deeplsd_lines",
                ],
                "enforce_threshold": 5.0,  # Enforce values in distance field to be no greater than this value
            },
        },
        "train_scene_list": "gluefactory/datasets/scannetv2_train.txt",
        "val_scene_list": "gluefactory/datasets/scannetv2_val.txt",
        # img list path from repo root -> use checked in file list, it is similar to pold2 file
        "rand_shuffle_seed": None,  # seed to randomly shuffle before split in train and val
        "val_size": 500,  # number of val images
        "train_size": 11500,  # number of train images
    }

    def _init(self, conf):
        # Auto-download the dataset if not existing
        if not (DATA_PATH / conf.data_dir).exists():
            self.download_scannet()
        # load image names
        with open(root / self.conf.train_scene_list, "r") as f:
            self.train_scene_list = [
                file_name.strip("\n") for file_name in f.readlines()
            ]
        with open(root / self.conf.val_scene_list, "r") as f:
            self.val_scene_list = [file_name.strip("\n") for file_name in f.readlines()]

        self.dset_dir = DATA_PATH / self.conf.data_dir
        self.img_dir = self.dset_dir / "images"

        # sample images
        self.sample_images()

        self.images = {
            "train": self.train_images,
            "val": self.val_images,
            "all": self.train_images + self.val_images,
        }
        logger.info(f"DATASET OVERALL(NO-SPLIT) IMAGES: {len(self.images["all"])}")

    def download_scannet(self):
        logger.info("Downloading the Scannet dataset...")
        tmp_dir = self.dset_dir.parent / "scannet_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        zip_name = "scannet_subset.zip"
        zip_path = tmp_dir / zip_name
        torch.hub.download_url_to_file(DOWNLOAD_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        shutil.move(tmp_dir / "scannet_frames_25k", str(self.dset_dir))

    def sample_images(self):
        """
        Read train and val scenes from respective files and sample names. Currently sampling less than one image per
        scene is not supported.

        1. Read all images for all train and val scenes
        2. filter scenes and images by equidistant sampling rules

        Equidistant sampling:
            - for scenes that have <= num of fair share images per scene - take them all
            - the remaining number of images is distributed among other scenes
            - for each of these scenes: take random shuffled images if no equal distance between samples possible
            - otherwise sample with fixed distance.
        """
        self.train_images = self._sample_split(
            self.conf.train_size, self.train_scene_list
        )
        self.val_images = self._sample_split(self.conf.val_size, self.val_scene_list)
        # randomize if wanted
        if self.conf.rand_shuffle_seed is not None:
            np.random.RandomState(self.conf.shuffle_seed).shuffle(self.train_images)
            np.random.RandomState(self.conf.shuffle_seed).shuffle(self.val_images)

    def _sample_split(self, num_images_needed, scenes):
        """
        Given a split of the dataset as scene list (scene list of train or val) sample images based on rules described
        in sample_images() method. Adds those images to the list ref provided
        Args:
            img_list_final_images: list ref to store sampled images in
            num_images: number of images to sample
            scenes: a list of scenes that contain the image candidates
        """
        img_list_final_images = list()
        # initialize scenes with image paths
        remaining_scenes = dict()
        for s in scenes:
            scene_folder = self.img_dir / s / "color"
            scene_image_paths = list(Path(scene_folder).glob("**/*.jpg"))
            remaining_scenes[s] = scene_image_paths

        # sample so we have
        while len(img_list_final_images) < num_images_needed and remaining_scenes:
            remaining_needed = num_images_needed - len(img_list_final_images)
            fair_share = math.ceil(remaining_needed / len(remaining_scenes))
            scenes_to_remove = list()

            for s, paths in remaining_scenes.items():
                if len(paths) <= fair_share:
                    img_list_final_images += paths
                    scenes_to_remove.append(s)
                else:
                    img_list_final_images += self._sample_with_equi_distance(
                        paths, fair_share, delete_taken=True
                    )

            if len(img_list_final_images) >= num_images_needed:
                break

            # remove fully taken scenes
            for s in scenes_to_remove:
                del remaining_scenes[s]
        # Finally remove possible little overshoot
        return img_list_final_images[:num_images_needed]

    @staticmethod
    def _sample_with_equi_distance(
        img_paths: list, num_to_sample: int, delete_taken: bool = True
    ):
        """
        Takes a list of images and samples from this the number of images wanted while having maximum distance between
        them (in terms of index position).
        """
        # Return empty list if nothing to sample
        if not img_paths or num_to_sample <= 0:
            return []
        # Return all elements if
        if num_to_sample >= len(img_paths):
            return img_paths.copy() if not delete_taken else img_paths
        # Calculate the ideal distance between samples
        step = (len(img_paths) - 1) / (num_to_sample - 1) if num_to_sample > 1 else 0
        # Generate the indices we want to sample
        indices = [round(i * step) for i in range(num_to_sample)]
        # Get the images at these indices
        sampled = [img_paths[i] for i in indices]
        # If delete_taken is True, remove the sampled images from the original list
        if delete_taken:
            # Remove items in reverse order to maintain correct indices
            for idx in sorted(indices, reverse=True):
                img_paths.pop(idx)
        return sampled

    def get_dataset(self, split):
        assert split in ["train", "val", "all"]
        return _Dataset(self.conf, self.images[split], split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_sub_paths: list[str], split):
        super().__init__()
        self.split = split
        self.conf = conf
        self.grayscale = bool(conf.grayscale)
        self.max_num_gt_kp = conf.load_features.point_gt.max_num_keypoints

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

        self.dset_dir = DATA_PATH / conf.data_dir
        self.img_dir = self.dset_dir / "images"
        self.dlsd_gt_folder = self.dset_dir / "deeplsd_gt"
        self.sp_gt_folder = self.dset_dir / "superpoint_gt"
        # Extract image paths
        self.image_sub_paths = [
            i.relative_to(self.img_dir) for i in image_sub_paths.copy()
        ]

        # making them relative for system independent names in export files (path used as name in export)
        if len(self.image_sub_paths) == 0:
            raise ValueError(f"Could not find any image in folder: {self.img_dir}.")
        logger.info(f"NUMBER OF IMAGES: {len(self.image_sub_paths)}")
        logger.info(
            f"KNOWN BATCHSIZE FOR MY SPLIT({self.split}) is {self.relevant_batch_size}"
        )
        # Load features
        if conf.load_features.do:
            # filter out where missing groundtruth
            new_img_path_list = []
            for img_sub_path in self.image_sub_paths:
                if not self.conf.load_features.check_exists:
                    new_img_path_list.append(img_sub_path)
                    continue
                # perform checks
                kp_heatmap_gt_file = (
                    self.sp_gt_folder
                    / img_sub_path.parent
                    / f"{img_sub_path.stem}.hdf5"
                )
                kp_points_gt_file = (
                    self.sp_gt_folder / img_sub_path.parent / f"{img_sub_path.stem}.npy"
                )
                df_af_gt_file = (
                    self.dlsd_gt_folder
                    / img_sub_path.parent
                    / f"{img_sub_path.stem}.hdf5"
                )
                dlsd_lines_gt_file = (
                    self.dlsd_gt_folder
                    / img_sub_path.parent
                    / f"{img_sub_path.stem}.npy"
                )
                if (
                    kp_heatmap_gt_file.exists()
                    and kp_points_gt_file.exists()
                    and df_af_gt_file.exists()
                    and dlsd_lines_gt_file.exists()
                ):
                    new_img_path_list.append(img_sub_path)

            self.image_sub_paths = new_img_path_list
            logger.info(f"NUMBER OF IMAGES WITH GT: {len(self.image_sub_paths)}")

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

    def _read_image(self, img_path, enforce_batch_dim=False):
        """
        Read image as tensor and puts it on device
        """
        img = load_image(img_path, grayscale=self.grayscale)
        if enforce_batch_dim:
            if img.ndim < 4:
                img = img.unsqueeze(0)
        assert img.ndim >= 3
        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img

    def read_datasets_from_h5(self, keys: list, file) -> dict:
        """
        Read datasets from h5 file. Expects np arrays in h5py files.
        """
        data = {}
        for key in keys:
            data[key] = torch.from_numpy(
                np.nan_to_num(file[key].__array__())
            )  # nan_to_num needed because of weird sp gt format
        return data

    def _read_groundtruth(self, img_sub_path, shape: int = None) -> dict:
        """
        Reads groundtruth for points and lines from respective files
        We can assume that gt files are existing at this point->checked in init!

        DeepLSD line endpoints can be loaded and used as keypoint groundtruth. This can be configured in the dataset config.
        One can also choose to only use deepLSD line endpoints.

        image_folder_path: image subpath relative from img_dir
        original_img_size: format w,h original image size to be able to create heatmaps.
        shape: shape to reshape img to if it is not None
        """
        # Load config and local variables
        reshape_scales = None
        reshaped_size = None
        heatmap_gt_key_name = self.conf.load_features.point_gt.data_keys[0]
        kp_gt_key_name = self.conf.load_features.point_gt.data_keys[1]
        kp_score_gt_key_name = self.conf.load_features.point_gt.data_keys[2]
        df_gt_key = self.conf.load_features.line_gt.data_keys[0]
        af_gt_key = self.conf.load_features.line_gt.data_keys[1]
        dlsd_lines_key = self.conf.load_features.line_gt.data_keys[2]

        ground_truth = {}
        # load keypoint gt file paths
        # kp_heatmap_gt_file = self.sp_gt_folder / img_sub_path.parent / f"{img_sub_path.stem}.hdf5"
        kp_points_gt_file = (
            self.sp_gt_folder / img_sub_path.parent / f"{img_sub_path.stem}.npy"
        )
        df_af_gt_file = (
            self.dlsd_gt_folder / img_sub_path.parent / f"{img_sub_path.stem}.hdf5"
        )
        dlsd_lines_gt_file = (
            self.dlsd_gt_folder / img_sub_path.parent / f"{img_sub_path.stem}.npy"
        )

        # Load Line GT
        with h5py.File(df_af_gt_file, "r") as line_file:
            ground_truth = {
                **self.read_datasets_from_h5(
                    list(self.conf.load_features.line_gt.data_keys[0:2]),
                    line_file,  # only select af and df here
                ),
                **ground_truth,
            }

        # threshold df if wanted
        df_img = ground_truth[df_gt_key]
        thres = self.conf.load_features.line_gt.enforce_threshold
        if thres is not None:
            df_img = np.where(df_img > thres, thres, df_img)
        df = torch.from_numpy(df_img).to(dtype=torch.float32)

        af_img = ground_truth[af_gt_key]
        af = af_img.to(dtype=torch.float32)

        # loaded lines have shape N x 2 x 2, each line is parametrized as its two endpoints one stored in each row
        if self.conf.load_features.line_gt.load_lines:
            lines = torch.from_numpy(np.load(dlsd_lines_gt_file)).to(
                dtype=torch.float32
            )

        # reshape AF/DF and lines if reshape is activated
        if shape is not None:
            preprocessor_df_out = self.preprocessors[shape](df.unsqueeze(0))
            reshape_scales = preprocessor_df_out["scales"]
            reshaped_size = preprocessor_df_out["image_size"]
            df = preprocessor_df_out["image"].squeeze(
                0
            )  # only store image here as padding map will be stored by preprocessing image
            af = self.preprocessors[shape](af.unsqueeze(0))["image"].squeeze(0)
            if self.conf.load_features.line_gt.load_lines:
                ground_truth[dlsd_lines_key] = (
                    lines * reshape_scales if reshape_scales is not None else lines
                )
                # TODO: is reshaping lines working properly? check!
                # todo: capping lines needed? Generate binary heatmap directly?

        ground_truth[df_gt_key] = df
        ground_truth[af_gt_key] = af

        # Load Keypoint GT hape: N x 2 (one kp per row), we assume them to be already sorted!
        kp_and_scores = torch.from_numpy(np.load(kp_points_gt_file)).to(
            dtype=torch.float32
        )

        original_kp = kp_and_scores[:, [0, 1]]
        keypoint_scores = kp_and_scores[:, 2]

        # rescale keypoints if reshape is activated
        keypoints = (
            original_kp * reshape_scales if reshape_scales is not None else original_kp
        )

        # CREATE HEATMAP
        heatmap = np.zeros_like(df)
        integer_kp_coordinates = torch.round(keypoints).to(dtype=torch.int)

        # select topk keypoints for creation of heatmap if configured like this
        if self.conf.load_features.point_gt.max_num_heatmap_keypoints > 0:
            num_selected_kp = min(
                [
                    self.conf.load_features.point_gt.max_num_heatmap_keypoints,
                    keypoint_scores.shape[0],
                ]
            )
            integer_kp_coordinates = integer_kp_coordinates[:num_selected_kp]
        # if reshaping is done clamp of rounded coordinates is necessary
        if reshaped_size is not None:
            integer_kp_coordinates = torch.stack(
                [
                    torch.clamp(
                        integer_kp_coordinates[:, 0], min=0, max=(reshaped_size[0] - 1)
                    ),
                    torch.clamp(
                        integer_kp_coordinates[:, 1], min=0, max=(reshaped_size[1] - 1)
                    ),
                ],
                dim=1,
            )
        if self.conf.load_features.point_gt.use_score_heatmap:
            heatmap[integer_kp_coordinates[:, 1], integer_kp_coordinates[:, 0]] = (
                keypoint_scores
            )
        else:
            heatmap[integer_kp_coordinates[:, 1], integer_kp_coordinates[:, 0]] = 1.0

        heatmap = torch.from_numpy(heatmap).to(dtype=torch.float32)

        ground_truth[heatmap_gt_key_name] = heatmap
        # TODO: does batching work with loaded points
        # choose max num keypoints if wanted. Attention: we dont sort by score here! If dlsd kp gt is used all scores of these are 1!
        if self.conf.load_features.point_gt.load_points:
            num_selected_kp = min([self.max_num_gt_kp, keypoint_scores.shape[0]])
            ground_truth[kp_gt_key_name] = (
                keypoints[:num_selected_kp, :]
                if self.max_num_gt_kp is not None
                else keypoints
            )
            ground_truth[kp_score_gt_key_name] = (
                keypoint_scores[:num_selected_kp]
                if self.max_num_gt_kp is not None
                else keypoint_scores
            )

        return ground_truth

    def __getitem__(self, idx):
        """
        Dataloader is usually just returning one datapoint by design. Batching is done in Loader normally.
        """
        img_path = self.img_dir / self.image_sub_paths[idx]
        img = self._read_image(img_path)
        orig_shape = img.shape[-1], img.shape[-2]
        size_to_reshape_to = self.select_resize_shape(orig_shape)
        data = {
            "name": str(self.image_sub_paths[idx]),
        }  # keys: 'name', 'scales', 'image_size', 'transform', 'original_image_size', 'image'
        if size_to_reshape_to == orig_shape:
            data["image"] = img
        else:
            data = {**data, **self.preprocessors[size_to_reshape_to](img)}
        if self.conf.load_features.do:
            gt = self._read_groundtruth(
                self.image_sub_paths[idx],
                None if size_to_reshape_to == orig_shape else size_to_reshape_to,
            )
            data = {**data, **gt}
        return data

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

    def __len__(self):
        return len(self.image_sub_paths)
