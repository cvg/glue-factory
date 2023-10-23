import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..datasets import get_dataset
from ..geometry.depth import sample_depth
from ..models import get_model
from ..settings import DATA_PATH
from ..utils.export_predictions import export_predictions

resize = 1024
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
    "sp_open": {
        "name": f"r{resize}_SP-open-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "extractors.superpoint_open",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
    "cv2-sift": {
        "name": f"r{resize}_opencv-SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": 4096,
            "backend": "opencv",
        },
    },
    "pycolmap-sift": {
        "name": f"r{resize}_pycolmap-SIFT-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
            "backend": "pycolmap",
        },
    },
    "pycolmap-sift-gpu": {
        "name": f"r{resize}_pycolmap_SIFTGPU-nms3-fixed-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.sift",
            "max_num_keypoints": n_kpts,
            "backend": "pycolmap_cuda",
            "nms_radius": 3,
        },
    },
    "keynet-affnet-hardnet": {
        "name": f"r{resize}_KeyNetAffNetHardNet-k{n_kpts}",
        "keys": ["keypoints", "descriptors", "keypoint_scores", "oris", "scales"],
        "gray": True,
        "conf": {
            "name": "extractors.keynet_affnet_hardnet",
            "max_num_keypoints": n_kpts,
        },
    },
    "disk": {
        "name": f"r{resize}_DISK-k{n_kpts}-nms5",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "extractors.disk_kornia",
            "max_num_keypoints": n_kpts,
        },
    },
    "aliked": {
        "name": f"r{resize}_ALIKED-k{n_kpts}-n16",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": False,
        "conf": {
            "name": "extractors.aliked",
            "max_num_keypoints": n_kpts,
        },
    },
}


def get_kp_depth(pred, data):
    d, valid = sample_depth(pred["keypoints"], data["depth"])
    return {"depth_keypoints": d, "valid_depth_keypoints": valid}


def run_export(feature_file, scene, args):
    conf = {
        "data": {
            "name": "megadepth",
            "views": 1,
            "grayscale": configs[args.method]["gray"],
            "preprocessing": {
                "resize": resize,
                "side": "long",
            },
            "batch_size": 1,
            "num_workers": args.num_workers,
            "read_depth": True,
            "train_split": [scene],
            "train_num_per_scene": None,
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

    if args.export_sparse_depth:
        callback_fn = get_kp_depth  # use this to store the depth of each keypoint
        keys = keys + ["depth_keypoints", "valid_depth_keypoints"]
    else:
        callback_fn = None
    export_predictions(
        loader, model, feature_file, as_half=True, keys=keys, callback_fn=callback_fn
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_prefix", type=str, default="")
    parser.add_argument("--method", type=str, default="sp")
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--export_sparse_depth", action="store_true")
    args = parser.parse_args()

    export_name = configs[args.method]["name"]

    data_root = Path(DATA_PATH, "megadepth/Undistorted_SfM")
    export_root = Path(DATA_PATH, "exports", "megadepth-undist-depth-" + export_name)
    export_root.mkdir(parents=True, exist_ok=True)

    if args.scenes is None:
        scenes = [p.name for p in data_root.iterdir() if p.is_dir()]
    else:
        with open(DATA_PATH / "megadepth" / args.scenes, "r") as f:
            scenes = f.read().split()
    for i, scene in enumerate(scenes):
        print(f"{i} / {len(scenes)}", scene)
        feature_file = export_root / (scene + ".h5")
        if feature_file.exists() and False:
            continue
        if not (data_root / scene / "images").exists():
            logging.info("Skip " + scene)
            continue
        logging.info(f"Export local features for scene {scene}")
        run_export(feature_file, scene, args)
