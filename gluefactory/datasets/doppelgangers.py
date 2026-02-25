"""DataLoader for the Doppelgangers dataset."""

import numpy as np
import torch

from .. import settings
from ..geometry import reconstruction
from ..utils import misc, preprocess
from .base_dataset import BaseDataset


class DoppelgangersDataset(BaseDataset):

    default_conf = {
        "root": "???",
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
        "load_images": True,
        "subset": None,
        "add_dummy_pose_depth": False,  # for compatibility with some pipelines
        "only_negatives": False,
        "visym": False,
        "add_visym_pairs": False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split: str, epoch: int = 0):
        return DoppelgangersSplit(self.conf, split, epoch)

    def download(self):
        raise NotImplementedError(
            "Please download the doppelgangers dataset from the official repository."
        )


class DoppelgangersSplit(torch.utils.data.Dataset):

    def __init__(self, conf, split: str, epoch: int = 0):
        self.conf = conf
        if conf.visym:
            pairs_name = {
                "test": "pairs_metadata/test_pairs_visym.npy",
                "val": "pairs_metadata/test_pairs_visym.npy",
                "train": "pairs_metadata/train_pairs_visym_fix.npy",
            }[split]
        else:
            pairs_name = {
                "test": "pairs_metadata/test_pairs.npy",
                "val": "pairs_metadata/test_pairs.npy",
                "train": "pairs_metadata/train_pairs_flip.npy",
            }[split]
        pair_f = settings.DATA_PATH / conf.root / pairs_name
        self.items = np.load(pair_f, allow_pickle=True)

        if conf.add_visym_pairs and split == "train":
            visym_pairs_name = "pairs_metadata/train_pairs_visym_fix.npy"
            visym_pair_f = settings.DATA_PATH / conf.root / visym_pairs_name
            visym_items = np.load(visym_pair_f, allow_pickle=True)
            self.items = np.concatenate([self.items, visym_items], axis=0)

        self.items = np.array(
            [x for x in self.items if ".gif" not in x[0] and ".gif" not in x[1]]
        )
        if self.conf.only_negatives:
            self.items = self.items[self.items[:, 2] < 1.0e-3]
        if conf.subset is not None:
            seed = conf.seed + epoch if "train" in split else 42
            self.items = np.random.default_rng(seed).choice(
                self.items, self.conf.subset, replace=False
            )
        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)
        self.split = split

        image_dir = {
            "test": "doppelgangers/images/test_set/",
            "val": "doppelgangers/images/test_set/",
            "train": "doppelgangers/images/train_set_flip/",
        }[split]
        self.image_dir = settings.DATA_PATH / conf.root / image_dir
        # The training images of visym are stored with the test set
        self.alt_image_dir = (
            settings.DATA_PATH / conf.root / "doppelgangers/images/test_set/"
        )

    def _read_view(self, name):
        path = self.image_dir / name
        if not path.exists():
            path = self.alt_image_dir / name
        img = preprocess.load_image(path)
        data = self.preprocessor(img)
        data["name"] = name
        return data

    def __getitem__(self, idx):
        name0, name1, has_overlap, num_matches = self.items[idx].tolist()
        has_overlap = int(has_overlap)
        data = {
            "name": "/".join([name0, name1]),
            "scene": name0.split("/")[0],
            "has_overlap": has_overlap,
            "num_sift_matches": num_matches,
        }

        if self.conf.load_images:
            data["view0"] = self._read_view(name0)
            data["view1"] = self._read_view(name1)

        if self.conf.add_dummy_pose_depth:
            # Unify it with other datasets by adding dummy pose and depth
            for i in range(2):
                data[f"view{i}"]["T_w2cam"] = reconstruction.Pose.identity()
                data[f"view{i}"]["camera"] = reconstruction.Camera.from_image(
                    data[f"view{i}"]["image"]
                ).compose_image_transform(data[f"view{i}"]["transform"])
                data[f"view{i}"]["depth"] = (
                    torch.zeros_like(data[f"view{i}"]["image"][0]) + 1e-2
                )
                data[f"view{i}"]["scene"] = data["scene"]

            data["T_0to1"] = data["view1"]["T_w2cam"].compose(
                data["view0"]["T_w2cam"].inv()
            )
            data["T_1to0"] = data["T_0to1"].inv()
            data["overlap_0to1"] = float(data["has_overlap"])
            data["overlap_1to0"] = float(data["has_overlap"])

            data["overlap"] = np.eye(2, dtype=np.float32)
            if data["has_overlap"]:
                data["overlap"][0, 1] = 1.0
                data["overlap"][1, 0] = 1.0
            data["idx"] = idx
            data.pop("has_overlap")
            data.pop("num_sift_matches")
        return data

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from ..visualization.viz2d import plot_image_grid

    conf = {
        "root": "doppelgangerspp",
        "preprocessing": {
            "resize": 512,
            "side": "long",
            "interpolation": "area",
            "antialias": False,
        },
        "num_workers": 1,
        "visym": True,
    }

    dataset = DoppelgangersDataset(conf)

    loader = dataset.get_data_loader("train")

    images = []
    for i, data in tqdm(enumerate(loader)):
        print(data["has_overlap"])
        images.append([data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)])
        if i > 3:
            print(misc.print_summary(data))
            break

    axes = plot_image_grid(images, dpi=200)
    plt.savefig("doppelgangers.png")
    plt.show()
