import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..datasets import get_dataset
from ..models import get_model
from ..settings import DATA_PATH
from ..utils.export_predictions import export_predictions

resize = 1600

sp_keys = ["keypoints", "descriptors", "keypoint_scores"]

# SuperPoint
n_kpts = 2048
configs = {
    "sp": {
        "name": f"r{resize}_SP-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "gluefactory_nonfree.superpoint",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
    "sift": {
        "name": f"r{resize}_SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "sift",
            "max_num_keypoints": n_kpts,
            "options": {
                "peak_threshold": 0.001,
            },
            "peak_threshold": 0.001,
            "device": "cpu",
        },
    },
    "disk": {
        "name": f"r{resize}_DISK-k{n_kpts}-nms6",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "disk",
            "max_num_keypoints": n_kpts,
        },
    },
}


def run_export(feature_file, images, args):
    conf = {
        "data": {
            "name": "image_folder",
            "grayscale": configs[args.method]["gray"],
            "preprocessing": {
                "resize": resize,
            },
            "images": str(images),
            "batch_size": 1,
            "num_workers": args.num_workers,
        },
        "split": "train",
        "model": configs[args.method]["conf"],
    }

    conf = OmegaConf.create(conf)

    keys = configs[args.method]["keys"]
    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader(conf.split or "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(conf.model.name)(conf.model).eval().to(device)

    export_predictions(loader, model, feature_file, as_half=True, keys=keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--export_prefix", type=str, default="")
    parser.add_argument("--method", type=str, default="sp")
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    export_name = configs[args.method]["name"]

    if args.dataset == "megadepth":
        data_root = Path(DATA_PATH, "megadepth/Undistorted_SfM")
        export_root = Path(DATA_PATH, "exports", "megadepth-undist-" + export_name)
        export_root.mkdir(parents=True, exist_ok=True)

        if args.scenes is None:
            scenes = [p.name for p in data_root.iterdir() if p.is_dir()]
        else:
            with open(DATA_PATH / "megadepth" / args.scenes, "r") as f:
                scenes = f.read().split()
        for i, scene in enumerate(scenes):
            # print(f'{i} / {len(scenes)}', scene)
            print(scene)
            feature_file = export_root / (scene + ".h5")
            if feature_file.exists():
                continue
            if not (data_root / scene / "images").exists():
                logging.info("Skip " + scene)
                continue
            logging.info(f"Export local features for scene {scene}")
            run_export(feature_file, data_root / scene / "images", args)
    else:
        data_root = Path(DATA_PATH, args.dataset)
        feature_file = Path(
            DATA_PATH, "exports", args.export_prefix + export_name + ".h5"
        )
        feature_file.parent.mkdir(exist_ok=True, parents=True)
        logging.info(
            f"Export local features for dataset {args.dataset} "
            f"to file {feature_file}"
        )
        run_export(feature_file, data_root)
