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
import enum

from gluefactory.models.deeplsd_inference import DeepLSD
from gluefactory.utils.image import load_image
from gluefactory.models.extractors.joint_point_line_extractor import JointPointLineDetectorDescriptor
from gluefactory.models.lines.pold2_extractor import LineExtractor

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

            "num_images": 100,
            "num_negative_per_image": 10,
            "negative_type": "random",  # random, deeplsd_random, deeplsd_neighbour, combined
            "combined_ratio": 0.5,
            "negative_neighbour_min_radius": 5,
            "negative_neighbour_max_radius": 10,
            "num_positive_per_image": 10,  # -1 to use all

            "mlp_config": {
                "has_angle_field": True,
                "has_distance_field": True, 
                
                "num_line_samples": 30,    # number of sampled points between line endpoints
                "brute_force_samples": False,  # sample all points between line endpoints
                "image_size": 800,         # size of the input image, relevant only if brute_force_samples is True

                "num_bands": 1,            # number of bands to sample along the line
                "band_width": 1,           # width of the band to sample along the line
            },

            "deeplsd_config": {
                "detect_lines": True,
                "line_detection_params": {
                    "merge": False,
                    "filtering": True,
                    "grad_thresh": 3,
                    "grad_nfa": True,
                },
                "weights": None,  # path to the weights of the DeepLSD model (relative to DATA_PATH)
            },

            "jpldd_config" : {
                "name": "joint_point_line_extractor",
                "max_num_keypoints": 500,  # setting for training, for eval: -1
                "timeit": True,  # override timeit: False from BaseModel
                "line_df_decoder_channels": 32,
                "line_af_decoder_channels": 32,
                "line_detection": {
                        "do": False,
                    },
                "checkpoint": None,
            },
            "glob": "revisitop1m/jpg/**/base_image.jpg",  # relative to DATA_PATH
            
            "debug": False, # debug the data generation (visualize positive and negative samples)
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
        logger.info(f"Loaded POLD2-MLP dataset with {positives.shape} positives and {negatives.shape} negatives")

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

        def get_line_from_image(file_path, deeplsd_net, jpldd_net, reshape_image=None):
            img = cv2.imread(file_path)[:, :, ::-1]
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if reshape_image is not None:
                gray_img = cv2.resize(gray_img, (reshape_image, reshape_image))
                img = cv2.resize(img, (reshape_image, reshape_image))

            if self.gen_debug:
                from copy import deepcopy
                self.IMAGE = np.ascontiguousarray(deepcopy(img), dtype=np.uint8)

            inputs = {
                "image": torch.tensor(gray_img, dtype=torch.float, device=device)[
                    None, None
                ]
                / 255.0
            }
            inputs_jpldd = {
                "image": torch.tensor(img.copy(), dtype=torch.float, device=device)[None].permute(0, 3, 1, 2)/255.0
            }

            with torch.no_grad():
                out_deeplsd = deeplsd_net(inputs)
                out_jpldd = jpldd_net(inputs_jpldd)


            # distance field
            distances = out_jpldd["line_distancefield"][0]
            distances /= deeplsd_net.conf.line_neighborhood

            # angle field
            angles = out_jpldd["line_anglefield"][0]
            angles = angles / np.pi

            lines = np.array(out_deeplsd["lines"][0])

            return distances, angles, lines, gray_img.shape
            
        def generate_random_endpoints(img_shape, gen_conf):
            # Randomly sample negative points and generate lines
            neg_x = np.random.randint(0, img_shape[1], (2 * self.num_negative)).reshape(-1, 1)
            neg_y = np.random.randint(0, img_shape[0], (2 * self.num_negative)).reshape(-1, 1)
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

        if data_dir.exists():
            if conf.generate.regenerate:
                logger.warning(f"Data directory already exists. Overwriting {data_dir}")
                shutil.rmtree(data_dir)
            else:
                logger.info("Found existing data. Not regenerating")
                return

        data_dir.mkdir(parents=True, exist_ok=True)

        gen_conf = conf.generate
        if gen_conf is None:
            raise ValueError("No data generation configuration found.")

        # Load the DeepLSD model
        ckpt_path = DATA_PATH / gen_conf.deeplsd_config.weights
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        deeplsd_net = DeepLSD(gen_conf.deeplsd_config)
        deeplsd_net.load_state_dict(ckpt["model"])
        deeplsd_net = deeplsd_net.to(device).eval()
        jpldd_net = JointPointLineDetectorDescriptor(gen_conf.jpldd_config)
        jpldd_net = jpldd_net.to(device).eval()
        
        extractor = LineExtractor(
            OmegaConf.create(
                {
                    "mlp_conf": gen_conf.mlp_config,
                }
            )
        )

        # Generate the data
        positives = []
        negatives = []

        fps = glob.glob(str(DATA_PATH / gen_conf.glob), recursive=True)
        print(f"Found {len(fps)} images for glob {str(DATA_PATH / gen_conf.glob)}")
        fps = np.random.choice(fps, gen_conf.num_images, replace=False)

        for file_path in tqdm(fps, desc="Generating data"):
            distance_map, angle_map, lines, img_shape = get_line_from_image(
                file_path, deeplsd_net, jpldd_net, gen_conf.mlp_config.image_size
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
                neg_lines = generate_deeplsd_neighbour_endpoints(lines, img_shape, gen_conf)

            elif gen_conf.negative_type == NegativeType.COMBINED.value:
                neg_deeplsd_random = generate_deeplsd_random_endpoints(lines)
                neg_deeplsd_neighbour = generate_deeplsd_neighbour_endpoints(lines, img_shape, gen_conf)

                num_neg_neigh = int(gen_conf.combined_ratio * self.num_negative)
                num_neg_rand = self.num_negative - num_neg_neigh

                neigh_idx = np.random.choice(len(neg_deeplsd_neighbour), min(num_neg_neigh, len(neg_deeplsd_neighbour)), replace=False)
                rand_idx = np.random.choice(len(neg_deeplsd_random), min(num_neg_rand, len(neg_deeplsd_random)), replace=False)

                neg_lines = np.concatenate([neg_deeplsd_neighbour[neigh_idx], neg_deeplsd_random[rand_idx]])

            else:
                raise ValueError(f"Unknown negative type: {gen_conf.negative_type}")
            
            num_neg = min(self.num_negative, len(neg_lines))
            neg_idx = np.random.choice(len(neg_lines), num_neg, replace=False)
            neg_lines = neg_lines[neg_idx]
            
            # DEBUG
            if self.gen_debug:
                dimg = show_lines(self.IMAGE[:,:,::-1].copy(), pos_lines.astype(int), color='green')
                dimg = show_lines(dimg, neg_lines.astype(int), color='red')
                dimg = show_points(dimg, pos_lines.reshape(-1, 2).astype(int))

                global IDX
                cv2.imwrite(f"tmp_dataset_debug/{self.IDX}.png", dimg)
                self.IDX += 1

            # convert to torch tensors
            device = extractor.device
            pos_lines = torch.tensor(pos_lines, dtype=torch.int, device=device)
            neg_lines = torch.tensor(neg_lines, dtype=torch.int, device=device)

            # Generate positive samples
            positives.append(
                extractor.mlp_input_prep(
                    pos_lines.reshape(-1, 2),
                    torch.arange(len(pos_lines)*2).reshape(-1, 2),
                    distance_map,
                    angle_map
                ).cpu().numpy()
            )

            # Generate negative samples
            negatives.append(
                extractor.mlp_input_prep(
                    neg_lines.reshape(-1, 2),
                    torch.arange(len(neg_lines)*2).reshape(-1, 2),
                    distance_map,
                    angle_map
                ).cpu().numpy()
            )

        positives = np.concatenate(positives)
        negatives = np.concatenate(negatives)

        # Save the data
        np.save(data_dir / "positives.npy", positives)
        np.save(data_dir / "negatives.npy", negatives)

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
