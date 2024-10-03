"""
Load the POLD2-MLP dataset (npy files if they exist, otherwise generate using DeepLSD).
Usage:
    python -m gluefactory.datasets.pold2_mlp_dataset --conf gluefactory/configs/pold2_mlp_dataloader_test.yaml
"""

import argparse
import glob
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.models.deeplsd_inference import DeepLSD

from ..settings import DATA_PATH
from ..utils.tools import fork_rng
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


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
            "use_df": True,
            "use_af": True,
            "num_images": 100,
            "num_negative_per_image": 10,
            "num_positive_per_image": 10,  # -1 to use all
            "num_line_samples": 150,  # number of sampled points between line endpoints
            "deeplsd_config": {
                "detect_lines": True,
                "line_detection_params": {
                    "merge": False,
                    "filtering": True,
                    "grad_thresh": 3,
                    "grad_nfa": True,
                },
            },
            "weights": None,  # path to the weights of the DeepLSD model (relative to DATA_PATH)
            "glob": "revisitop1m/jpg/**/base_image.jpeg",  # relative to DATA_PATH
        },
    }

    def _init(self, conf):
        data_dir = DATA_PATH / conf.data_dir
        if not data_dir.exists() or conf.generate is not None:
            self.generate_data(conf, data_dir)

        if (
            not (data_dir / "positives.npy").exists()
            or not (data_dir / "negatives.npy").exists()
        ):
            raise ValueError("No data found in the data directory.")

        # Load the data
        positives = torch.from_numpy(np.load(data_dir / "positives.npy"))
        negatives = torch.from_numpy(np.load(data_dir / "negatives.npy"))
        samples = torch.cat((positives, negatives), axis=0)
        labels = torch.cat(
            (torch.ones(positives.shape[0]), torch.zeros(negatives.shape[0])), axis=0
        )

        if conf.shuffle_seed is not None:
            idxs = torch.randperm(
                samples.shape[0],
                generator=torch.Generator().manual_seed(conf.shuffle_seed),
            )
            samples = samples[idxs]
            labels = labels[idxs]

        train_samples = samples[: conf.train_size]
        val_samples = samples[conf.train_size : conf.train_size + conf.val_size]

        train_labels = labels[: conf.train_size]
        val_labels = labels[conf.train_size : conf.train_size + conf.val_size]

        self.split_data = {
            "train": {"samples": train_samples, "labels": train_labels},
            "val": {"samples": val_samples, "labels": val_labels},
        }
        logger.info(
            f"Loaded POLD2-MLP dataset with total {len(samples)} samples, Train: {len(train_samples)}, Val: {len(val_samples)}"
        )

    def generate_data(self, conf: OmegaConf, data_dir: Path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def get_line_from_image(file_path, net):
            img = cv2.imread(file_path)[:, :, ::-1]
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            inputs = {
                "image": torch.tensor(gray_img, dtype=torch.float, device=device)[
                    None, None
                ]
                / 255.0
            }

            with torch.no_grad():
                out = net(inputs)

            # distance field
            distances = out["df"][0]
            distances /= net.conf.line_neighborhood
            distances = distances.cpu().numpy()

            # angle field
            angles = out["line_level"][0]
            angles = angles.cpu().numpy() / np.pi

            lines = np.array(out["lines"][0])

            return distances, angles, lines, gray_img.shape

        def datasetEntryFromPoints(p1, p2, distance_map, angle_map, blend, image_shape):
            v1 = blend * p1
            v2 = (1 - blend) * p2

            points = (v1 + v2).round().astype(int)

            points[:, 1] = np.clip(points[:, 1], 0, image_shape[0] - 1)
            points[:, 0] = np.clip(points[:, 0], 0, image_shape[1] - 1)

            df_val = distance_map[points[:, 1], points[:, 0]].reshape(-1)
            af_val = angle_map[points[:, 1], points[:, 0]].reshape(-1)

            if conf.generate.use_df and not conf.generate.use_af:
                return df_val
            elif conf.generate.use_af and not conf.generate.use_df:
                return af_val
            else:
                return np.hstack([df_val, af_val])

        if data_dir.exists():
            logger.warning("Data directory already exists. Overwriting.")
            shutil.rmtree(data_dir)

        data_dir.mkdir(parents=True, exist_ok=True)

        gen_conf = conf.generate
        if gen_conf is None:
            raise ValueError("No data generation configuration found.")

        if not gen_conf.use_df and not gen_conf.use_af:
            raise ValueError(
                "At least one of the fields (distance or angle) must be used."
            )

        # Load the DeepLSD model
        ckpt_path = DATA_PATH / gen_conf.weights
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        net = DeepLSD(gen_conf.deeplsd_config)
        net.load_state_dict(ckpt["model"])
        net = net.to(device).eval()

        blend = np.linspace(0, 1, conf.generate.num_line_samples).reshape(-1, 1)

        # Generate the data
        positives = []
        negatives = []

        fps = glob.glob(str(DATA_PATH / gen_conf.glob), recursive=True)
        fps = np.random.choice(fps, gen_conf.num_images, replace=False)

        for file_path in tqdm(fps, desc="Generating data"):
            distance_map, angle_map, lines, img_shape = get_line_from_image(
                file_path, net
            )

            lines = lines.astype(int)  # convert to int for indexing
            lines = lines[: conf.generate.num_positive_per_image]

            # Generate positive samples
            for line in lines:
                positives.append(
                    datasetEntryFromPoints(
                        line[0], line[1], distance_map, angle_map, blend, img_shape
                    )
                )

            # Generate negative samples
            for _ in range(conf.generate.num_negative_per_image):
                p1 = np.array(
                    [
                        np.random.randint(0, img_shape[1]),
                        np.random.randint(0, img_shape[0]),
                    ]
                )
                p2 = np.array(
                    [
                        np.random.randint(0, img_shape[1]),
                        np.random.randint(0, img_shape[0]),
                    ]
                )
                negatives.append(
                    datasetEntryFromPoints(
                        p1, p2, distance_map, angle_map, blend, img_shape
                    )
                )

        # Save the data
        np.save(data_dir / "positives.npy", np.array(positives))
        np.save(data_dir / "negatives.npy", np.array(negatives))

    def get_dataset(self, split):
        return _Dataset(self.conf, self.split_data[split], split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split_data, split):
        self.conf = conf
        self.split = split

        self.samples = split_data["samples"]
        self.labels = split_data["labels"]

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        data = {"input": self.samples[idx], "label": self.labels[idx]}

        return data

    def __len__(self):
        return len(self.samples)


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
