"""Composed dataset that combines multiple datasets."""

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from ..geometry.reconstruction import PerspectiveCamera
from ..utils import misc, preprocess
from . import get_dataset
from .augmentations import augmentations
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ComposedDataset(BaseDataset):
    default_conf = {
        "childs": {},  # dict of dataset configurations
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
        "target_length": "min",  # min, max, <dataset_name>, number
        "sample_from": None,  # list of dataset names to sample from, None means all
        "weights": None,  # Same order and length as sample_from
        "photometric": {"name": "identity", "p": 0.75},
        "force_perspective_camera": False,
    }

    def _init(self, conf):
        child_confs = conf.childs
        self.datasets = {
            name: get_dataset(name)(c)
            for name, c in child_confs.items()
            if name in (conf.sample_from or child_confs.keys())
        }

    def get_dataset(self, split: str, epoch: int = 0):
        return ComposedSplit(self.conf, self.datasets, split, epoch)


class ComposedSplit(torch.utils.data.Dataset):
    def __init__(self, conf, datasets, split: str, epoch: int = 0):
        self.conf = conf = copy.deepcopy(conf)
        if split != "train":
            OmegaConf.set_readonly(self.conf.preprocessing, False)
            # Only perform homography augmentation during training
            self.conf.preprocessing.homography.p = 0.0
            OmegaConf.set_readonly(self.conf.preprocessing, True)

        self.dataset_names = (
            conf.get(f"{split}_split") or conf.sample_from or list(datasets.keys())
        )
        self.datasets = [
            datasets[name].get_dataset(split, epoch) for name in self.dataset_names
        ]

        self.dataset_valid = [d.conf.get("valid_geometry", True) for d in self.datasets]
        logger.info(f"[{split}] Dataset valid_geometry flags: {self.dataset_valid}")
        self.sizes = np.array([len(d) for d in self.datasets])

        logger.info(
            "[%s] Composed dataset with datasets: %s", split, self.dataset_names
        )

        if conf.weights is not None:
            assert conf.sample_from is not None, "sample_from required to use weights."
            assert len(conf.weights) == len(
                conf.sample_from
            ), "Number of weights must match number of datasets in sample_from."
            self.weights = conf.weights
            # Reorder weights to match dataset_names
            self.weights = [
                self.weights[conf.sample_from.index(name)]
                for name in self.dataset_names
            ]
        else:
            # Alternative, use weight per dataset in its config
            self.weights = [
                d.conf.get(f"{split}_weight", d.conf.get("weight", None))
                for d in self.datasets
            ]
        if all(w is None for w in self.weights):
            self.weights = None
        else:
            self.weights = np.array([1.0 if w is None else w for w in self.weights])
        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)

        augmentor_name = "identity" if split != "train" else conf.photometric.name
        self.photometric_augmentor = augmentations[augmentor_name](conf.photometric)

        target_length = conf.get(f"{split}_target_length", conf.target_length)
        if self.weights is not None:
            weights = np.array(self.weights)
            weights = weights / weights.sum()
            if target_length == "min":
                ref_length = min(self.sizes)
                ref_weight = weights[np.argmin(self.sizes)]
            elif target_length == "max":
                ref_length = max(self.sizes)
                ref_weight = weights[np.argmax(self.sizes)]
            elif isinstance(target_length, str) and target_length in self.dataset_names:
                ref_idx = self.dataset_names.index(target_length)
                ref_length = self.sizes[ref_idx]
                ref_weight = weights[ref_idx]
            elif isinstance(target_length, int):
                ref_length = target_length
                ref_weight = 1.0
            else:
                raise ValueError(f"Unknown target_length {target_length}")
            actual_sizes = (weights * ref_length / ref_weight).astype(int)

            self.sample_idxs = []
            for i, (dataset, actual_size) in enumerate(
                zip(self.datasets, actual_sizes)
            ):
                if actual_size > len(dataset):
                    idxs = np.random.default_rng(conf.seed + epoch + i).choice(
                        len(dataset), actual_size, replace=True
                    )
                elif actual_size < len(dataset):
                    idxs = np.random.default_rng(conf.seed + epoch + i).choice(
                        len(dataset), actual_size, replace=False
                    )
                else:
                    idxs = np.arange(len(dataset))
                self.sample_idxs.append(idxs)
            self.sizes = actual_sizes
        for name, size in zip(self.dataset_names, self.sizes):
            logger.info(f"[{split}] Dataset {name}: {size} samples")
        logger.info(f"[{split}] Total: {self.sizes.sum()} samples")
        self.cum_sizes = np.cumsum([0] + self.sizes.tolist())

    def get_idxs(self, idx):
        dataset_idx = np.where(idx < self.cum_sizes)[0][0] - 1
        sample_idx = idx - self.cum_sizes[dataset_idx]
        if self.weights is not None:
            sample_idx = self.sample_idxs[dataset_idx][sample_idx]
        return dataset_idx, sample_idx

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        element = self.datasets[dataset_idx][sample_idx]
        element["dataset"] = self.dataset_names[dataset_idx.item()]

        for i, view in enumerate(misc.iterelements(element)):
            element[f"view{i}"].update(self.preprocessor(view["image"]))
            element[f"view{i}"]["image"] = self.photometric_augmentor(
                element[f"view{i}"]["image"].permute(1, 2, 0), return_tensor=True
            )
            if "depth" in view:
                element[f"view{i}"]["depth"] = self.preprocessor.interpolate(
                    view["depth"][None],
                    element[f"view{i}"]["transform"],
                    element[f"view{i}"]["image"].shape[-2:],
                    mode="nearest",
                )[0]
            if "camera" in view:
                if self.conf.force_perspective_camera and not isinstance(
                    view["camera"], PerspectiveCamera
                ):
                    view["camera"] = PerspectiveCamera.from_pinhole(view["camera"])
                element[f"view{i}"]["camera"] = view["camera"].compose_image_transform(
                    element[f"view{i}"]["transform"]
                )
        element["valid_geometry"] = self.dataset_valid[dataset_idx]
        return element

    def stats(self):
        metrics, figures = {}, {}

        fig, ax = plt.subplots()

        dataset_names = self.dataset_names
        ax.bar(dataset_names, self.sizes, alpha=0.5, label="sampled")
        ax.bar(
            dataset_names, [len(d) for d in self.datasets], alpha=0.5, label="original"
        )
        ax.legend()
        ax.set_title("Dataset sizes")

        figures["dataset_sizes"] = fig

        for name, d in zip(self.dataset_names, self.datasets):
            if hasattr(d, "stats"):
                dmetrics, dfigures = d.stats()
                metrics.update({f"{name}/{k}": v for k, v in dmetrics.items()})
                figures.update({f"{name}/{k}": v for k, v in dfigures.items()})

        return metrics, figures


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from ..visualization.viz2d import plot_image_grid

    conf = {
        "name": "composed",
        "preprocessing": {
            "resize": (512, 512),
        },
        "childs": {
            "doppelgangers": {
                "root": "doppelgangerspp",
                "subset": 100,
                "add_dummy_pose_depth": True,
                "only_negatives": True,
            },
            "megadepth": {
                "name": "megadepth",
                "train_num_per_scene": 100,
                "test_num_per_scene": 10,
            },
        },
        "target_length": 50,
        "seed": 42,
        "batch_size": 4,
    }
    dataset = get_dataset("composed")(conf)
    loader = dataset.get_data_loader("test", shuffle=True)

    metrics, figs = loader.dataset.stats()
    for k, fig in figs.items():
        fig.show()

    images = []
    for i, data in tqdm(enumerate(loader)):
        images.append(
            [view["image"][0].permute(1, 2, 0) for view in misc.iterelements(data)]
        )
        if i > 3:
            print(misc.print_summary(data))
            break

    axes = plot_image_grid(images, dpi=200)
    plt.savefig("composed.png")
    plt.show()
