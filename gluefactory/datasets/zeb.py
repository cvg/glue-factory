"""
Zeroshot Evaluation Benchmark Dataset (ZEB).
Source: https://arxiv.org/abs/2402.11095
Code: https://github.com/xuelunshen/gim/
"""

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import tqdm

from ..settings import DATA_PATH
from ..utils.preprocess import ImagePreprocessor, load_image
from ..visualization import viz2d
from .base_dataset import BaseDataset
from .image_pairs import parse_camera, parse_relative_pose

logger = logging.getLogger(__name__)


def read_pair_data(pairs_file: Path) -> list[str]:
    with open(pairs_file, "r") as f:
        pair_data = f.readlines()[0].rstrip().split(" ")
    return pair_data


def parse_overlap(pair_data: list[str]) -> tuple[float, float]:
    """Parse overlap from pair data."""
    if len(pair_data) < 2:
        raise ValueError(f"Pair data {pair_data} does not contain overlap information.")
    return float(pair_data[0]), float(pair_data[1])


def parse_pairs(pairs_file: Path) -> tuple[Path, Path, str]:
    """Parse pairs file and return a list of pairs."""
    pair_data = read_pair_data(pairs_file)
    file_name = pairs_file.stem

    img_name0, img_name1 = pair_data[:2]
    img_name0 = img_name0.split(".")[0]
    img_name1 = img_name1.split(".")[0]

    subscene_name = file_name.replace(f"{img_name0}-{img_name1}", "")
    subscene_name = subscene_name.replace(f"{img_name0}_{img_name1}", "")
    subscene_name, sep = subscene_name[:-1], subscene_name[-1]
    img_path0 = list(pairs_file.parent.glob(f"{subscene_name}{sep}{img_name0}.*"))[0]
    img_path1 = list(pairs_file.parent.glob(f"{subscene_name}{sep}{img_name1}.*"))[0]

    assert img_path0.exists(), f"Image {img_path0} does not exist."
    assert img_path1.exists(), f"Image {img_path1} does not exist."
    return img_path0, img_path1, pair_data[2:]


class ZEBPairs(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "root": "???",
        "preprocessing": ImagePreprocessor.default_conf,
        "scene_list": None,  # ToDo: add scenes interface
        "exclude_scenes": None,  # scenes to exclude
        "shuffle": False,
        "seed": 42,
        "max_per_scene": None,  # maximum number of pairs per scene
        "min_overlap": 0.0,  # minimum overlap for pairs
        "max_overlap": 1.0,  # maximum overlap for pairs
        "check": False,  # check if pairs files are valid
    }

    def _init(self, conf):
        self.root = DATA_PATH / conf.root
        assert self.root.exists()
        # we first read the scenes
        if isinstance(conf.scene_list, Iterable):
            self.scenes = list(conf.scene_list)
        elif isinstance(conf.scene_list, str):
            scenes_path = self.root / conf.scene_list
            self.scenes = scenes_path.read_text().rstrip("\n").split("\n")
        else:
            self.scenes = [s.name for s in self.root.glob("*")]
        if conf.exclude_scenes is not None:
            self.scenes = [
                scene for scene in self.scenes if scene not in conf.exclude_scenes
            ]
        logger.info(f"Found scenes {self.scenes}.")
        # read posed views, check if images exist

        self.items = []
        for i, scene in enumerate(sorted(self.scenes)):
            pair_files = list((self.root / scene).glob("*.txt"))
            if conf.check:
                for pair_file in tqdm.tqdm(
                    pair_files[:900], desc=f"Check pairs in {scene}"
                ):
                    parse_pairs(pair_file)  # check if pairs file is valid (asserts)
            if conf.min_overlap > 0.0 or conf.max_overlap < 1.0:
                overlaps = np.array(
                    [
                        min(*parse_overlap(read_pair_data(pair_file)[2:4]))
                        for pair_file in pair_files
                    ]
                )
                valid = overlaps >= conf.min_overlap
                valid &= overlaps <= conf.max_overlap
                logger.info(
                    "Filtering pairs in %s with overlap in [%f, %f]: %d/%d valid.",
                    scene,
                    conf.min_overlap,
                    conf.max_overlap,
                    valid.sum(),
                    len(pair_files),
                )
                valid_idx = np.where(valid)[0]
                pair_files = [pair_files[idx.item()] for idx in valid_idx]
            if conf.max_per_scene is not None and len(pair_files) > conf.max_per_scene:
                pair_files = sorted(pair_files, key=lambda x: x.stem)
                pair_files = np.random.RandomState(i).choice(
                    pair_files, conf.max_per_scene, replace=False
                )
            self.items.extend(pair_files)
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        if conf.shuffle:
            logger.info("Shuffling pairs.")
            self.items = sorted(self.items, key=lambda x: x.stem)
            np.random.RandomState(conf.seed).shuffle(self.items)

    def get_dataset(self, split: str, epoch: int = 0):
        assert split == "test", "ZEBPairs dataset does not have train/val splits."
        return self

    def _read_view(self, path):
        img = load_image(path)
        data = self.preprocessor(img)
        data["name"] = path.name
        return data

    def __getitem__(self, idx):
        pair_file = self.items[idx]
        img_path0, img_path1, pair_data = parse_pairs(pair_file)
        data0 = self._read_view(img_path0)
        data1 = self._read_view(img_path1)

        data = {
            "view0": data0,
            "view1": data1,
        }
        data["view0"]["camera"] = parse_camera(pair_data[2:11]).scale(data0["scales"])
        data["view1"]["camera"] = parse_camera(pair_data[11:20]).scale(data1["scales"])
        data["T_0to1"] = parse_relative_pose(pair_data[20:])
        data["T_1to0"] = data["T_0to1"].inv()
        data["scene"] = pair_file.parent.name

        data["name"] = data["scene"] + "/" + pair_file.stem
        data["overlap"] = min(*parse_overlap(pair_data[1:3]))
        return data

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    config = {
        "root": "zeb",
        "scene_list": None,  # ["blendedmvs", "scenenet"]
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "shuffle": False,
        "max_per_scene": 1,
    }

    dataset = ZEBPairs(config)
    loader = dataset.get_data_loader("test")
    logger.info("The dataset has %d elements.", len(loader))
    images = []

    ds_iter = iter(loader)
    for i in range(12):
        batch = next(ds_iter)
        images.append(
            [
                batch["view0"]["image"][0].permute(1, 2, 0).numpy(),
                batch["view1"]["image"][0].permute(1, 2, 0).numpy(),
            ]
        )

    viz2d.plot_image_grid(images)
    import matplotlib.pyplot as plt

    plt.savefig("zeb_pairs.png", dpi=300, bbox_inches="tight")
