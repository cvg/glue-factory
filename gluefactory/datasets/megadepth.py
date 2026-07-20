import argparse
import logging
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .. import settings
from ..geometry import reconstruction
from ..geometry import transforms as gtr
from ..models import cache_loader
from ..utils import preprocess, tools
from . import base_dataset

logger = logging.getLogger(__name__)
scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class MegaDepth(base_dataset.BaseDataset):
    default_conf = {
        # paths
        "data_dir": "megadepth/",
        "depth_subpath": "depth_undistorted/",
        "image_subpath": "Undistorted_SfM/",
        "info_dir": "scene_info/",  # @TODO: intrinsics problem?
        # Training
        "train_split": "train_scenes_clean.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "valid_scenes_clean.txt",
        "val_num_per_scene": None,
        "val_pairs": None,
        # Test
        "test_split": "test_scenes_clean.txt",
        "test_num_per_scene": None,
        "test_pairs": None,
        # data sampling
        "views": 2,
        "min_overlap": 0.3,  # only with D2-Net format
        "max_overlap": 1.0,  # only with D2-Net format
        "num_overlap_bins": 1,
        "sort_by_overlap": False,
        "triplet_enforce_overlap": False,  # only with views==3
        # image options
        "read_depth": True,
        "read_image": True,
        "grayscale": False,
        "allow_distractors": False,
        "squeeze_single_view": False,
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0,
        # features from cache
        "load_features": {
            "do": False,
            **cache_loader.CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        if not (settings.DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the MegaDepth dataset.")
            self.download()

    def download(self):
        data_dir = settings.DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "megadepth_tmp"
        if tmp_dir.exists():  # The previous download failed.
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://cvg-data.inf.ethz.ch/megadepth/"
        for tar_name, out_name in (
            ("Undistorted_SfM.tar.gz", self.conf.image_subpath),
            ("depth_undistorted.tar.gz", self.conf.depth_subpath),
            ("scene_info.tar.gz", self.conf.info_dir),
        ):
            tar_path = tmp_dir / tar_name
            torch.hub.download_url_to_file(url_base + tar_name, tar_path)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=tmp_dir)
            tar_path.unlink()
            shutil.move(tmp_dir / tar_name.split(".")[0], tmp_dir / out_name)
        shutil.move(tmp_dir, data_dir)

    def get_dataset(self, split: str, epoch: int = 0):
        assert self.conf.views in [1, 2, 3]
        seed = self.conf.seed
        if split == "train":
            seed = seed + epoch
        return _MegaDepthSplit(self.conf, split, seed=seed)


class _MegaDepthSplit(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True, seed: int | None = None):
        self.root = settings.DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf
        self.seed = seed if seed is not None else conf.seed

        split_conf = conf[split + "_split"]
        if isinstance(split_conf, (str, Path)):
            scenes_path = scene_lists_path / split_conf
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            scenes = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        scenes = sorted(set(scenes))

        if conf.load_features.do:
            self.feature_loader = cache_loader.CacheLoader(conf.load_features)

        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)

        self.images = {}
        self.depths = {}
        self.poses = {}
        self.intrinsics = {}

        # load metadata
        self.info_dir = self.root / self.conf.info_dir
        self.scenes = []
        for scene in scenes:
            path = self.info_dir / (scene + ".npz")
            try:
                info = np.load(str(path), allow_pickle=True)
            except Exception:
                logger.warning(
                    "Cannot load scene info for scene %s at %s.", scene, path
                )
                continue
            self.images[scene] = info["image_paths"]
            self.depths[scene] = info["depth_paths"]
            self.poses[scene] = info["poses"]
            self.intrinsics[scene] = info["intrinsics"]
            self.scenes.append(scene)

        self.items = []
        if load_sample:
            self.sample_new_items(self.seed)
            assert len(self.items) > 0

    def sample_new_items(self, seed):
        logger.info("Sampling new %s data with seed %d.", self.split, seed)
        self.items = []
        split = self.split
        num_per_scene = self.conf[self.split + "_num_per_scene"]
        if isinstance(num_per_scene, Iterable):
            num_pos, num_neg = num_per_scene
        else:
            num_pos = num_per_scene
            num_neg = None
        if split != "train" and self.conf[split + "_pairs"] is not None:
            # Fixed validation or test pairs
            assert num_pos is None
            assert num_neg is None
            pairs_path = scene_lists_path / self.conf[split + "_pairs"]
            for line in pairs_path.read_text().rstrip("\n").split("\n"):
                imnames = line.split(" ")
                scene = imnames[0].split("/")[0]
                idxs = []
                for imname in imnames:
                    assert imname.split("/")[0] == scene
                    impath = self.conf.image_subpath + imname
                    assert impath in self.images[scene], (impath, scene)
                    idx = np.where(self.images[scene] == impath)[0][0]
                    idxs.append(idx)
                self.items.append(
                    (scene, idxs, np.ones((len(idxs), len(idxs)), dtype=np.float32))
                )
        elif self.conf.views == 1:
            for scene in self.scenes:
                if scene not in self.images:
                    continue
                valid = (self.images[scene] != None) | (  # noqa: E711
                    self.depths[scene] != None  # noqa: E711
                )
                ids = np.where(valid)[0]
                if num_pos and len(ids) > num_pos:
                    ids = np.random.RandomState(seed).choice(
                        ids, num_pos, replace=False
                    )
                ids = [(scene, (i,), None) for i in ids]
                self.items.extend(ids)
        else:
            logger.info("Sampling new %s data with seed %d.", self.split, seed)
            for i, scene in enumerate(
                tqdm(
                    sorted(self.scenes),
                    desc="Sampling groups",
                    disable=not self.conf.get("use_pbar", True),
                )
            ):
                self.sample_groups(seed + i, scene, num_pos, num_neg)

        if self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: np.mean(i[-1]), reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def sample_groups(self, seed, scene, num_pos, num_neg):
        if self.conf.views == 2:
            return self.sample_pairs(seed, scene, num_pos, num_neg)
        elif self.conf.views == 3:
            return self.sample_triplets(seed, scene, num_pos, num_neg)
        else:
            raise ValueError(self.conf.views)

    def sample_pairs(self, seed, scene, num_pos, num_neg):
        path = self.info_dir / (scene + ".npz")
        assert path.exists(), path
        info = np.load(str(path), allow_pickle=True)
        valid = (self.images[scene] != None) & (  # noqa: E711
            self.depths[scene] != None  # noqa: E711
        )
        ind = np.where(valid)[0]
        mat = info["overlap_matrix"][valid][:, valid]

        if num_pos is not None:
            # Sample a subset of pairs, binned by overlap.
            num_bins = self.conf.num_overlap_bins
            assert num_bins > 0
            bin_width = (self.conf.max_overlap - self.conf.min_overlap) / num_bins
            num_per_bin = num_pos // num_bins
            pairs_all = []
            for k in range(num_bins):
                bin_min = self.conf.min_overlap + k * bin_width
                bin_max = bin_min + bin_width
                pairs_bin = (mat > bin_min) & (mat <= bin_max)
                pairs_bin = np.stack(np.where(pairs_bin), -1)
                pairs_all.append(pairs_bin)
            # Skip bins with too few samples
            has_enough_samples = [len(p) >= num_per_bin * 2 for p in pairs_all]
            num_per_bin_2 = num_pos // max(1, sum(has_enough_samples))
            pairs = []
            for pairs_bin, keep in zip(pairs_all, has_enough_samples):
                if keep:
                    pairs.append(sample_n(pairs_bin, num_per_bin_2, seed))
            if len(pairs) == 0:
                logger.warning(
                    "No pairs found for scene %s with overlap in [%.2f, %.2f].",
                    scene,
                    self.conf.min_overlap,
                    self.conf.max_overlap,
                )
                return
            pairs = np.concatenate(pairs, 0)
        else:
            pairs = (mat > self.conf.min_overlap) & (mat <= self.conf.max_overlap)
            pairs = np.stack(np.where(pairs), -1)

        pairs = [(scene, (ind[i], ind[j]), mat[[i, j]][:, [i, j]]) for i, j in pairs]
        if num_neg is not None:
            neg_pairs = np.stack(np.where(mat <= 0.0), -1)
            neg_pairs = sample_n(neg_pairs, num_neg, seed)
            pairs += [(scene, (ind[i], ind[j]), np.zeros((2, 2))) for i, j in neg_pairs]
        self.items.extend(pairs)

    def sample_triplets(self, seed, scene, num_pos, num_neg):
        path = self.info_dir / (scene + ".npz")
        assert path.exists(), path
        info = np.load(str(path), allow_pickle=True)
        if self.conf.num_overlap_bins > 1:
            raise NotImplementedError("TODO")
        valid = (self.images[scene] != None) & (  # noqa: E711
            self.depths[scene] != None  # noqa: E711
        )
        ind = np.where(valid)[0]
        mat = info["overlap_matrix"][valid][:, valid]
        good = (mat > self.conf.min_overlap) & (mat <= self.conf.max_overlap)
        triplets = []
        if self.conf.triplet_enforce_overlap:
            pairs = np.stack(np.where(good), -1)
            for i0, i1 in pairs:
                for i2 in pairs[pairs[:, 0] == i0, 1]:
                    if good[i1, i2]:
                        triplets.append((i0, i1, i2))
            if len(triplets) > num_pos:
                selected = np.random.RandomState(seed).choice(
                    len(triplets), num_pos, replace=False
                )
                selected = range(num_pos)
                triplets = np.array(triplets)[selected]
        else:
            # we first enforce that each row has >1 pairs
            non_unique = good.sum(-1) > 1
            ind_r = np.where(non_unique)[0]
            good = good[non_unique]
            pairs = np.stack(np.where(good), -1)
            if num_pos is not None and len(pairs) > num_pos:
                selected = np.random.RandomState(seed).choice(
                    len(pairs), num_pos, replace=False
                )
                pairs = pairs[selected]
            for idx, (k, i) in enumerate(pairs):
                # We now sample a j from row k s.t. i != j
                possible_j = np.where(good[k])[0]
                possible_j = possible_j[possible_j != i]
                selected = np.random.RandomState(seed + idx).choice(
                    len(possible_j), 1, replace=False
                )[0]
                triplets.append((ind_r[k], i, possible_j[selected]))
            triplets = [
                (scene, (ind[k], ind[i], ind[j]), mat[[k, i, j]][:, [k, i, j]])
                for k, i, j in triplets
            ]
            self.items.extend(triplets)

    def _read_view(self, scene, idx):
        if idx is None and self.conf.allow_distractors:
            scenes = list(set(self.scenes) - {scene})
            scene = np.random.choice(scenes).item()
            valid = self.images[scene] != None
            idx = np.random.choice(len(self.images[scene]), p=valid / valid.sum())
        path = self.root / self.images[scene][idx]

        # read pose data
        K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        T = self.poses[scene][idx].astype(np.float32, copy=False)

        # read image
        if self.conf.read_image:
            img = preprocess.load_image(
                self.root / self.images[scene][idx], self.conf.grayscale
            )
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()

        # read depth
        if self.conf.read_depth:
            depth_path = (
                self.root / self.conf.depth_subpath / scene / (path.stem + ".h5")
            )
            with h5py.File(str(depth_path), "r") as f:
                depth_map = f["/depth"].__array__().astype(np.float32, copy=False)
                depth_map = torch.as_tensor(depth_map)[None]
            assert depth_map.shape[-2:] == img.shape[-2:]
        else:
            depth_map = None

        # add random rotations
        do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = torch.rot90(img, k=-k, dims=[1, 2])
                if self.conf.read_depth:
                    depth_map = torch.rot90(depth_map, k=-k, dims=[1, 2]).clone()
                K = gtr.rotate_intrinsics(K, img.shape, k + 2)
                T = gtr.rotate_pose_inplane(T, k + 2)

        name = path.name

        data = self.preprocessor(img)
        if depth_map is not None:
            data["depth"] = self.preprocessor.interpolate(
                depth_map,
                data["transform"],
                data["image"].shape[-2:],
                mode="nearest",
            )[0]
        # Scale intrinsics
        data = {
            "name": name,
            "scene": scene,
            "T_w2cam": reconstruction.Pose.from_4x4mat(T),
            "depth": depth_map,
            "camera": reconstruction.Camera.from_calibration_matrix(K)
            .float()
            .compose_image_transform(data["transform"]),
            **data,
        }

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            if do_rotate and k != 0:
                # ang = np.deg2rad(k * 90.)
                kpts = features["keypoints"].clone()
                x, y = kpts[:, 0].clone(), kpts[:, 1].clone()
                w, h = data["image_size"]
                if k == 1:
                    kpts[:, 0] = w - y
                    kpts[:, 1] = x
                elif k == -1:
                    kpts[:, 0] = y
                    kpts[:, 1] = h - x
                else:
                    raise ValueError
                features["keypoints"] = kpts

            data = {"cache": features, **data}
        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with tools.fork_rng(self.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        if isinstance(idx, list):
            scene, idxs, overlap = idx
        else:
            scene, idxs, overlap = self.items[idx]
        views = [self._read_view(scene, i) for i in idxs]
        nviews = len(views)
        data = {f"view{i}": view for i, view in enumerate(views)}

        for k in range(nviews):
            for l in range(k + 1, nviews):
                data[f"T_{k}to{l}"] = views[l]["T_w2cam"] @ views[k]["T_w2cam"].inv()
                data[f"T_{l}to{k}"] = views[k]["T_w2cam"] @ views[l]["T_w2cam"].inv()
                if isinstance(overlap, np.ndarray):
                    data[f"overlap_{k}to{l}"] = overlap[k, l]
                    data[f"overlap_{l}to{k}"] = overlap[k, l]
                elif isinstance(overlap, (float, int)):
                    data[f"overlap_{k}to{l}"] = overlap
                    data[f"overlap_{l}to{k}"] = overlap
        data["name"] = f"{scene}/{'_'.join([v['name'] for v in views])}"

        if overlap is not None:
            data["overlap"] = overlap

        if nviews == 1 and self.conf.squeeze_single_view:
            view_data = data.pop("view0")
            data = {**data, **view_data}
        data["scene"] = scene
        data["idx"] = idx
        return data

    def __len__(self):
        return len(self.items)


def visualize(args):
    conf = {
        "min_overlap": 0.1,
        "max_overlap": 0.7,
        "num_overlap_bins": 3,
        "sort_by_overlap": False,
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
    }
    from ..visualization import viz2d

    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = MegaDepth(conf)
    loader = dataset.get_data_loader(args.split)
    # logger.info("The dataset has %d elements.", len(loader))

    # train_iter = iter(loader)
    # for _ in tqdm(range(100), smoothing=0.0):
    #     data = next(train_iter)
    # exit()

    with tools.fork_rng(seed=dataset.conf.seed):
        images, depths = [], []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [
                    data[f"view{i}"]["image"][0].permute(1, 2, 0)
                    for i in range(dataset.conf.views)
                ]
            )
            depths.append(
                [data[f"view{i}"]["depth"][0] for i in range(dataset.conf.views)]
            )
    axes = viz2d.plot_image_grid(images, dpi=args.dpi)
    for i in range(len(images)):
        viz2d.plot_heatmaps(depths[i], axes=axes[i])
    plt.show()
    plt.savefig("megadepth.png")


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
