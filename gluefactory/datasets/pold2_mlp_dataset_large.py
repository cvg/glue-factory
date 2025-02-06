"""
Load the POLD2-MLP dataset (h5 files if they exist, otherwise generate using DeepLSD Outputs).
Usage:
    python -m gluefactory.datasets.pold2_mlp_dataset_large --conf gluefactory/configs/pold2_mlp_dataloader_test.yaml
"""

import argparse
import enum
import glob
import logging
import shutil
from pathlib import Path

import cv2
import h5py as h5
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.models.extractors.joint_point_line_extractor import (
    JointPointLineDetectorDescriptor,
)
from gluefactory.models.lines.pold2_extractor import LineExtractor
from gluefactory.utils.image import load_image

from ..settings import DATA_PATH
from ..utils.tools import fork_rng
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class NegativeType(enum.Enum):
    RANDOM = "random"
    DEEPLSD_RANDOM = "deeplsd_random"
    DEEPLSD_NEIGHBOUR = "deeplsd_neighbour"
    COMBINED = "combined"


class POLD2_MLP_Dataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "pold2_mlp_dataset",  # the top-level directory with the npy files
        # splits
        "train_size": 100,
        "val_size": 10,
        "shuffle_seed": 0,  # or None to skip
        "reseed": False,
        # data generation (None to skip)
        "generate": {
            "regenerate": False,
            "fields_and_lines_path": "DeepLSD-Outputs-OXPA/DeepLSD-Outputs-OXPA.h5",
            "h5_file_name": None,  # Name the file based on the config - NumBands, ImageSize, NumLineSamples, BandWidth
            "deeplsd_line_neighborhood": 5,
            "num_images": 100,
            "num_positive_per_image": 10,  # -1 to use all
            "num_negative_per_image": 10,
            "negative_type": "combined",  # random, deeplsd_random, deeplsd_neighbour, combined
            "combined_ratio": 0.5,
            "negative_neighbour_min_radius": 5,
            "negative_neighbour_max_radius": 10,
            "mlp_config": {
                "has_angle_field": True,
                "has_distance_field": True,
                "num_line_samples": 30,  # number of sampled points between line endpoints
                "brute_force_samples": False,  # sample all points between line endpoints
                "image_size": 800,  # size of the input image, relevant only if brute_force_samples is True
                "num_bands": 1,  # number of bands to sample along the line
                "band_width": 1,  # width of the band to sample along the line
            },
            "debug": False,  # debug the data generation (visualize positive and negative samples)
        },
    }

    def _init(self, conf):
        data_dir = DATA_PATH / conf.data_dir
        self.data_dir = data_dir
        h5_file_name = conf.generate.h5_file_name

        if h5_file_name is None and conf.generate is None:
            raise ValueError(
                "No h5 file name found in the configuration. Either provide a name or regenerate the data by specifying the generate configuration."
            )
        elif h5_file_name is None:
            num_bands = conf.generate.mlp_config.num_bands
            image_size = conf.generate.mlp_config.image_size
            num_line_samples = conf.generate.mlp_config.num_line_samples
            band_width = conf.generate.mlp_config.band_width
            brute_force_samples = conf.generate.mlp_config.brute_force_samples

            if brute_force_samples:
                h5_file_name = f"MLPDataset-{image_size}x{image_size}_{num_bands}Bands_BW{band_width}_BruteForceSamples.h5"
            else:
                h5_file_name = f"MLPDataset-{image_size}x{image_size}_{num_bands}Bands_BW{band_width}_{num_line_samples}Samples.h5"
        self.h5_file_name = h5_file_name

        if not data_dir.exists() or (
            conf.generate is not None and conf.generate.regenerate
        ):
            self.generate_data(conf, data_dir)

        if not (data_dir / self.h5_file_name).exists():
            raise ValueError(
                f"Data file {data_dir / self.h5_file_name} not found. Generate the data first."
            )

        # Get the number of samples from the h5 file
        with h5.File(data_dir / self.h5_file_name, "r") as f:
            all_samples = list(f.keys())
            num_data_samples = len(all_samples)

        if conf.train_size + conf.val_size > num_data_samples:
            raise ValueError(
                f"Train size {conf.train_size} + Val size {conf.val_size} is greater than the total number of samples {num_data_samples}"
            )

        idxs = np.random.permutation(num_data_samples)
        self.train_samples = idxs[: conf.train_size]
        self.val_samples = idxs[conf.train_size : conf.train_size + conf.val_size]

        self.train_samples = [all_samples[i] for i in self.train_samples]
        self.val_samples = [all_samples[i] for i in self.val_samples]

        self.train_idxs = []
        self.val_idxs = []

        for idx in self.train_samples:
            with h5.File(data_dir / self.h5_file_name, "r") as f:
                for i in range(len(f[f"{idx}"].keys())):
                    self.train_idxs.append((idx, i))

        for idx in self.val_samples:
            with h5.File(data_dir / self.h5_file_name, "r") as f:
                for i in range(len(f[f"{idx}"].keys())):
                    self.val_idxs.append((idx, i))

        self.split_data = {
            "train": self.train_idxs,
            "val": self.val_idxs,
        }
        logger.info(
            f"Loaded POLD2-MLP dataset with {len(self.train_idxs)} training lines, {len(self.val_idxs)} validation lines, {len(self.train_samples)} training images, and {len(self.val_samples)} validation images"
        )

    def generate_data(self, conf: OmegaConf, data_dir: Path):

        # DEBUG
        self.gen_debug = False
        if conf.generate.debug:
            import os

            from gluefactory.visualization.viz2d import show_lines, show_points

            self.IMAGE = None
            self.IDX = 0
            self.gen_debug = True

            if os.path.exists("tmp_dataset_debug"):
                shutil.rmtree("tmp_dataset_debug")
            os.makedirs("tmp_dataset_debug")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def get_line_from_image(file_path, fields_and_lines=None):
            img = np.array(fields_and_lines[file_path]["img"])

            if self.gen_debug:
                from copy import deepcopy

                self.IMAGE = np.ascontiguousarray(deepcopy(img), dtype=np.uint8)

            # distance field
            distances = np.array(fields_and_lines[file_path]["jpldd_df"])
            distances = torch.from_numpy(distances)
            distances /= conf.generate.deeplsd_line_neighborhood

            # angle field
            angles = np.array(fields_and_lines[file_path]["jpldd_af"])
            angles = torch.from_numpy(angles)
            angles = angles / torch.pi

            lines = np.array(fields_and_lines[file_path]["deeplsd_lines"])

            return distances, angles, lines, img.shape[:2]

        def generate_random_endpoints(img_shape, gen_conf):
            # Randomly sample negative points and generate lines
            neg_x = np.random.randint(0, img_shape[1], (2 * self.num_negative)).reshape(
                -1, 1
            )
            neg_y = np.random.randint(0, img_shape[0], (2 * self.num_negative)).reshape(
                -1, 1
            )
            neg_lines = np.stack([neg_x, neg_y], axis=1).reshape(-1, 2, 2)

            return neg_lines

        def generate_deeplsd_random_endpoints(lines):
            # Randomly pair up the deeplsd endpoints to generate negative samples
            lines = lines.copy()
            endpoints = lines.reshape(-1, 2)
            np.random.shuffle(endpoints)
            neg_lines = endpoints.reshape(-1, 2, 2)

            return neg_lines

        def generate_deeplsd_neighbour_endpoints(lines, img_shape, gen_conf):
            # Pairup points in the neighbourhood of the deeplsd endpoints to generate hard negative samples
            neg_lines = []
            min_radius = gen_conf.negative_neighbour_min_radius
            max_radius = gen_conf.negative_neighbour_max_radius

            for line in lines:
                radius = np.random.randint(min_radius, max_radius)

                p1, p2 = line
                p1_neigh = p1 + np.random.randint(-radius, radius, 2)
                p2_neigh = p2 + np.random.randint(-radius, radius, 2)
                p1_neigh = np.clip(p1_neigh, 0, img_shape[::-1])
                p2_neigh = np.clip(p2_neigh, 0, img_shape[::-1])
                neg_lines.append(np.stack([p1_neigh, p2_neigh]))

            neg_lines = np.array(neg_lines)

            return neg_lines

        if data_dir.exists() and os.path.exists(self.h5_file_name):
            if conf.generate.regenerate:
                logger.warning(f"HFile already exists. Overwriting {data_dir}")
            else:
                logger.info("Found existing data. Not regenerating")
                return

        data_dir.mkdir(parents=True, exist_ok=True)

        gen_conf = conf.generate
        if gen_conf is None:
            raise ValueError("No data generation configuration found.")

        extractor = LineExtractor(
            OmegaConf.create(
                {
                    "mlp_conf": gen_conf.mlp_config,
                }
            )
        )

        # Generate the data

        fields_and_lines = h5.File(DATA_PATH / gen_conf.fields_and_lines_path, "r")

        fps = list(fields_and_lines.keys())
        print(f"Found {len(fps)} images in the dataset")
        fps = (
            np.random.choice(fps, gen_conf.num_images, replace=False)
            if gen_conf.num_images > 0
            else fps
        )

        for file_path in tqdm(fps, desc="Generating data"):
            distance_map, angle_map, lines, img_shape = get_line_from_image(
                file_path, fields_and_lines
            )

            lines = lines.astype(int)  # convert to int for indexing

            self.num_positive = gen_conf.num_positive_per_image
            self.num_negative = gen_conf.num_negative_per_image

            self.num_positive = min(self.num_positive, self.num_negative)
            self.num_negative = min(self.num_negative, self.num_positive)

            # Postive samples
            if self.num_positive == -1:
                pos_idx = np.arange(len(lines))
                self.num_positive = len(lines)
                self.num_negative = len(lines)
            else:
                num_pos = min(self.num_positive, len(lines))
                pos_idx = np.random.choice(len(lines), num_pos, replace=False)
            pos_lines = lines[pos_idx]

            # Negative samples
            if gen_conf.negative_type == NegativeType.RANDOM.value:
                neg_lines = generate_random_endpoints(img_shape, gen_conf)

            elif gen_conf.negative_type == NegativeType.DEEPLSD_RANDOM.value:
                neg_lines = generate_deeplsd_random_endpoints(lines)

            elif gen_conf.negative_type == NegativeType.DEEPLSD_NEIGHBOUR.value:
                neg_lines = generate_deeplsd_neighbour_endpoints(
                    lines, img_shape, gen_conf
                )

            elif gen_conf.negative_type == NegativeType.COMBINED.value:
                neg_deeplsd_random = generate_deeplsd_random_endpoints(lines)
                neg_deeplsd_neighbour = generate_deeplsd_neighbour_endpoints(
                    lines, img_shape, gen_conf
                )

                num_neg_neigh = int(gen_conf.combined_ratio * self.num_negative)
                num_neg_rand = self.num_negative - num_neg_neigh

                neigh_idx = np.random.choice(
                    len(neg_deeplsd_neighbour), num_neg_neigh, replace=False
                )
                rand_idx = np.random.choice(
                    len(neg_deeplsd_random), num_neg_rand, replace=False
                )

                neg_lines = np.concatenate(
                    [neg_deeplsd_neighbour[neigh_idx], neg_deeplsd_random[rand_idx]]
                )

            else:
                raise ValueError(f"Unknown negative type: {gen_conf.negative_type}")

            num_neg = min(self.num_negative, len(neg_lines))
            neg_idx = np.random.choice(len(neg_lines), num_neg, replace=False)
            neg_lines = neg_lines[neg_idx]

            # DEBUG
            if self.gen_debug:
                dimg = show_lines(
                    self.IMAGE[:, :, ::-1].copy(), pos_lines.astype(int), color="green"
                )
                dimg = show_lines(dimg, neg_lines.astype(int), color="red")
                dimg = show_points(dimg, pos_lines.reshape(-1, 2).astype(int))

                global IDX
                cv2.imwrite(f"tmp_dataset_debug/{self.IDX}.png", dimg)
                self.IDX += 1

            # convert to torch tensors
            device = extractor.device
            pos_lines = torch.tensor(pos_lines, dtype=torch.int, device=device)
            neg_lines = torch.tensor(neg_lines, dtype=torch.int, device=device)

            # Generate positive samples
            positives = (
                extractor.mlp_input_prep(
                    pos_lines.reshape(-1, 2),
                    torch.arange(len(pos_lines) * 2)
                    .reshape(-1, 2)
                    .to(extractor.device),
                    distance_map.to(extractor.device),
                    angle_map.to(extractor.device),
                )
                .cpu()
                .numpy()
            )

            # Generate negative samples
            negatives = (
                extractor.mlp_input_prep(
                    neg_lines.reshape(-1, 2),
                    torch.arange(len(neg_lines) * 2)
                    .reshape(-1, 2)
                    .to(extractor.device),
                    distance_map.to(extractor.device),
                    angle_map.to(extractor.device),
                )
                .cpu()
                .numpy()
            )

            # Save the data
            pos_labels = np.ones(len(positives))
            neg_labels = np.zeros(len(negatives))

            lines = np.concatenate([positives, negatives])
            labels = np.concatenate([pos_labels, neg_labels])

            shuffle_idx = np.random.permutation(len(lines))
            lines = lines[shuffle_idx]
            labels = labels[shuffle_idx]

            with h5.File(data_dir / self.h5_file_name, "a") as f:
                img_group = f.create_group(f"{file_path}")
                for i, (line, label) in enumerate(zip(lines, labels)):
                    line_group = img_group.create_group(f"line_{i}")
                    line_group.create_dataset("sampled_data", data=line)
                    line_group.create_dataset("label", data=label)

    def get_dataset(self, split):
        return _Dataset(
            self.conf, self.split_data[split], split, self.data_dir / self.h5_file_name
        )


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split_data, split, h5_file_name):
        self.conf = conf
        self.split = split
        self.idxs = split_data
        self.h5_file_name = h5_file_name

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):

        with h5.File(DATA_PATH / self.h5_file_name, "r") as f:
            img_idx, line_idx = self.idxs[idx]
            line_data = f[f"{img_idx}"][f"line_{line_idx}"]

            sample = np.array(line_data["sampled_data"])
            sample = torch.from_numpy(sample)
            label = np.array(line_data["label"])
            label = torch.from_numpy(label)

        data = {"input": sample, "label": label}

        return data

    def __len__(self):
        return len(self.idxs)


if __name__ == "__main__":
    from .. import logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default=None)
    args = parser.parse_args()

    conf = (
        OmegaConf.load(args.conf)
        if args.conf is not None
        else POLD2_MLP_Dataset.default_conf
    )
    dataset = POLD2_MLP_Dataset(conf)

    # Test train loader
    train_loader = dataset.get_data_loader("train")
    logger.info(f"The train dataset has {len(train_loader)} elements.")
    for data in train_loader:
        print(f'TRAIN - input: {data["input"].shape}, label: {data["label"].shape}')
        break

    # Test val loader
    val_loader = dataset.get_data_loader("val")
    logger.info(f"The validation dataset has {len(val_loader)} elements.")
    for data in val_loader:
        print(f'VAL - input: {data["input"].shape}, label: {data["label"].shape}')
        break
