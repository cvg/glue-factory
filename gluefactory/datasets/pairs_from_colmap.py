"""
Simply load images from a folder or nested folders (does not have any split).
"""

import collections
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pycolmap
from omegaconf import OmegaConf
from tqdm import tqdm

from .. import settings
from .posed_images import PosedImageDataset, names_to_pair

logger = logging.getLogger(__name__)


def write_colmap_views(path: Path, sfm: pycolmap.Reconstruction):
    with open(path, "w") as f:
        for image_id, image in sfm.images.items():
            c_t_w = image.cam_from_world()
            camera = sfm.cameras[image.camera_id]
            c_r_w = c_t_w.rotation.matrix()
            c_t_w = c_t_w.translation
            data = [image.name] + c_r_w.flatten().tolist() + c_t_w.tolist()
            data += [
                camera.model.name,
                camera.width,
                camera.height,
            ] + camera.params.tolist()
            data = [str(d) for d in data]
            f.write(" ".join(data) + "\n")


def extract_covisible_pairs(sfm: pycolmap.Reconstruction, unique: bool = True):
    covisibility = collections.defaultdict(int)
    for image_id, image in tqdm(sfm.images.items()):
        # Load image data
        for p2d in image.get_observation_points2D():
            p3d: pycolmap.Point3D = sfm.points3D[p2d.point3D_id]
            for track_el in p3d.track.elements:
                if track_el.image_id == image_id:
                    continue
                covisibility[(image_id, track_el.image_id)] += 1

    covisibility = {
        (idx0, idx1): (
            min(n_cov01, covisibility[(idx1, idx0)]),
            min(
                n_cov01 / sfm.images[idx0].num_points3D,
                covisibility[(idx1, idx0)] / sfm.images[idx1].num_points3D,
            ),
        )
        for (idx0, idx1), n_cov01 in covisibility.items()
        if idx0 < idx1 or not unique
    }

    return covisibility


class ColmapImagePairsDataset(PosedImageDataset):
    default_conf = {
        **PosedImageDataset.default_conf,
        "sfm": "???",
        "min_covisible_points": 5,
        "min_overlap": 0.0,
        "max_overlap": 1.0,
        "overwrite": False,
        "max_per_scene": None,
        "view_groups": "{scene}/covisibility/{num_pairs}pairs_{num_covisible}-{min_overlap}-{max_overlap}.txt",  # noqa: E501
    }

    def _init(self, conf):
        self.root = settings.DATA_PATH / conf.root
        assert self.root.exists()
        # we first read the scenes
        if isinstance(conf.scene_list, Iterable):
            self.scenes = list(conf.scene_list)
        elif isinstance(conf.scene_list, str):
            scenes_path = self.root / conf.scene_list
            self.scenes = scenes_path.read_text().rstrip("\n").split("\n")
        else:
            self.scenes = [s.name for s in self.root.glob("*")]

        OmegaConf.set_readonly(conf, False)
        conf.view_groups = conf.view_groups.format(
            num_covisible=conf.min_covisible_points,
            min_overlap=conf.min_overlap,
            max_overlap=conf.max_overlap,
            num_pairs=conf.max_per_scene or "",
            scene="{scene}",  # Do not format scene here
        )
        self.conf = conf
        OmegaConf.set_readonly(conf, True)

        logger.info(f"Found scenes {self.scenes}.")
        # read posed views, check if images exist
        for i, scene in enumerate(sorted(self.scenes)):
            scene_sfm_path = self.root / conf.sfm.format(scene=scene)

            view_path = self.root / conf.views.format(scene=scene)
            if not view_path.exists() or conf.overwrite:
                # We cache the views for faster loading
                sfm = pycolmap.Reconstruction(scene_sfm_path)
                write_colmap_views(view_path, sfm)
            else:
                sfm = None

            if (self.root / scene / "pairs.txt").exists():
                (self.root / scene / "pairs.txt").unlink()

            pairs_path = self.root / conf.view_groups.format(scene=scene)
            pairs_path.parent.mkdir(exist_ok=True)

            if pairs_path.exists() and not conf.overwrite:
                continue

            # Extract covisible pairs
            sfm = sfm if sfm is not None else pycolmap.Reconstruction(scene_sfm_path)
            logger.info(f"Extracting covisible pairs for scene {scene}...")
            covisibility = extract_covisible_pairs(sfm, unique=True)

            # Filter pairs based on covisibility & overlap
            valid_pairs = [
                (idx0, idx1)
                for (idx0, idx1), (n_cov, overlap) in covisibility.items()
                if n_cov >= self.conf.min_covisible_points
                and overlap >= self.conf.min_overlap
                and overlap <= self.conf.max_overlap
            ]

            logger.info(f"Found {len(valid_pairs)} valid pairs in scene {scene}.")

            # Sort for reproducibility
            valid_pairs = sorted(valid_pairs)

            if self.conf.max_per_scene and len(valid_pairs) > self.conf.max_per_scene:
                seed = self.conf.seed + i
                rng = np.random.default_rng(seed)
                valid_idxs = rng.choice(
                    len(valid_pairs), self.conf.max_per_scene, replace=False
                )
                valid_pairs = [valid_pairs[idx] for idx in valid_idxs.tolist()]
                logger.info(f"Sampled {len(valid_pairs)} pairs with seed {seed}")

            with open(pairs_path, "w") as f:
                for idx0, idx1 in valid_pairs:
                    name0 = sfm.images[idx0].name
                    name1 = sfm.images[idx1].name
                    f.write(f"{name0} {name1}\n")

            overlap_data_path = pairs_path.with_suffix(".json")
            logger.info(f"Writing overlap data for {scene} to {overlap_data_path}")
            overlap_dict = {}
            for idx0, idx1 in valid_pairs:
                name0 = sfm.images[idx0].name
                name1 = sfm.images[idx1].name
                num_covisible, overlap = covisibility[(idx0, idx1)]
                overlap_dict[names_to_pair(name0, name1)] = {
                    "num_covisible": num_covisible,
                    "overlap": overlap,
                }

            with open(overlap_data_path, "w") as f:
                json.dump(overlap_dict, f, indent=2)

        super()._init(conf)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        overlap_data_path = (
            self.root / self.conf.view_groups.format(scene=data["scene"])
        ).with_suffix(".json")
        if overlap_data_path.exists():
            with open(overlap_data_path, "r") as f:
                data.update(json.load(f)[data["name"]])
        return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from ..visualization.viz2d import plot_heatmaps, plot_image_grid

    conf = {
        "root": "ETH3D_undistorted_resizedx2",
        "image_dir": "{scene}/images",
        "depth_dir": "{scene}/ground_truth_depth_dense",
        "depth_format": "h5",
        "scene_list": None,
        "sfm": "{scene}/dslr_calibration_undistorted/",
        "views": "{scene}/views.txt",  # To cache poses & cameras
        "overwrite": True,
        "min_overlap": 0.1,
        "max_overlap": 0.9,
        "max_per_scene": 100,
        "preprocessing": {
            "side": "long",
            "interpolation": "area",
            "antialias": False,
        },
        "num_workers": 1,
    }

    dataset = ColmapImagePairsDataset(conf)

    loader = dataset.get_data_loader("test")

    images, depths = [], []
    for i, data in tqdm(enumerate(loader)):
        images.append(
            [
                data[f"view{i}"]["image"][0].permute(1, 2, 0)
                for i in range(data["nviews"][0])
            ]
        )
        depths.append([data[f"view{i}"]["depth"][0] for i in range(data["nviews"][0])])
        if i > 3:
            break

    axes = plot_image_grid(images, dpi=200)
    for i in range(len(images)):
        plot_heatmaps(depths[i], axes=axes[i])
    plt.savefig("posed_images.png")
    plt.show()
