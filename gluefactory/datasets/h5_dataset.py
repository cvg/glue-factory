import hashlib
import json
import logging
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # registers custom compression filters with h5py
import numpy as np
import torch
from tqdm import tqdm

from .. import settings
from ..geometry import reconstruction
from ..utils import preprocess, tools
from . import base_dataset

logger = logging.getLogger(__name__)


# Supported H5 layouts:
#
# Single file (data_path points to a .h5 file):
#   dataset.h5
#   └── {scene}/
#       ├── views/
#       │   └── 0, 1, 2, ...
#       │       ├── image           # JPEG bytes
#       │       ├── name            # original filename, string
#       │       ├── depth           # float32 [H, W], optional
#       │       ├── K               # [3, 3] float32
#       │       └── c_T_w          # [4, 4] float32
#       ├── overlaps                # int32 [M, 3]: (id0, id1, raw_overlap_pct), directional, unnormalized
#       ├── depth_coverage          # int32 [N], percent, optional (default: ones)
#       └── pairs                   # int32 [M, 3]: (id0, id1, overlap_pct), precomputed symmetric
#
# Directory of per-scene files (data_path points to a directory):
#   data_path/
#   └── <arbitrary>.h5
#       └── {scene}/            # scene name as top-level key (same structure as above)
#           ├── views/...
#           ├── overlaps            # int32 [M, 3]: (id0, id1, raw_overlap_pct)
#           ├── depth_coverage      # int32 [N], percent
#           └── pairs               # int32 [M, 3]: (id0, id1, overlap_pct)


class H5Dataset(base_dataset.BaseDataset):
    default_conf = {
        "data_path": "???",  # path to .h5 file or directory containing per-scene .h5 files
        "train_scenes": None,  # txt file path or list of scene names, None = all
        "val_scenes": None,
        "test_scenes": None,
        "train_num_per_scene": None,
        "val_num_per_scene": None,
        "test_num_per_scene": None,
        "use_pairs": True,  # use precomputed pairs key if available, else use overlaps
        "balance_views": False,  # sample per-row in overlap matrix for uniform view coverage
        "balance_overlap": False,  # inverse-frequency weighting for uniform overlap distribution
        "min_overlap": 0.0,
        "max_overlap": 1.0,
        "read_depth": True,
        "read_image": True,
        "use_valid_mask": False,
        "cache_sampling": True,  # cache per-scene sampling to {scene}_sample_cache.npz
        "preload_files": True,  # pre-open all h5 handles and view groups at worker start
        "preprocessing": preprocess.ImagePreprocessor.default_conf,
        "reseed": False,
        "seed": 0,
    }

    def _init(self, conf):
        if Path(conf.data_path).exists():
            # Handle absolute or relative path as-is
            self.h5_path = Path(conf.data_path)
        else:
            self.h5_path = settings.DATA_PATH / conf.data_path
        assert self.h5_path.exists(), self.h5_path

    def get_dataset(self, split: str, epoch: int = 0):
        seed = self.conf.seed + (epoch if split == "train" else 0)
        return _H5Split(self.conf, self.h5_path, split, seed=seed)


def _balance_overlap_sample(overlap_vals, num_pairs, rng):
    """Sample indices with inverse-frequency weighting for a uniform overlap distribution."""
    n_bins = 10
    hist, bin_edges = np.histogram(overlap_vals, bins=n_bins)
    bin_idx = np.clip(np.digitize(overlap_vals, bin_edges) - 1, 0, n_bins - 1)
    weights = 1.0 / np.maximum(hist[bin_idx], 1).astype(np.float64)
    weights /= weights.sum()
    n = min(num_pairs, len(overlap_vals))
    return rng.choice(len(overlap_vals), n, replace=False, p=weights)


def _balance_views_sample(pair_ids, n_views, num_pairs, rng, overlap_vals=None):
    """Sample indices with per-view stratification for uniform view coverage."""
    k = max(1, int(np.ceil(2 * num_pairs / n_views)))
    view_to_pairs = defaultdict(list)
    for idx, (id0, id1) in enumerate(pair_ids):
        view_to_pairs[id0].append(idx)
        view_to_pairs[id1].append(idx)
    selected = set()
    for pair_idxs in view_to_pairs.values():
        if len(pair_idxs) > k:
            if overlap_vals is not None:
                chosen = _balance_overlap_sample(overlap_vals[pair_idxs], k, rng)
                pair_idxs = np.array(pair_idxs)[chosen].tolist()
            else:
                pair_idxs = rng.choice(pair_idxs, k, replace=False).tolist()
        selected.update(pair_idxs)
    sel = np.array(sorted(selected))
    if len(sel) > num_pairs:
        sel = rng.choice(sel, num_pairs, replace=False)
    return sel


def _sample_stratified_pairs_from_overlaps(
    overlap_min, overlap_max, min_ov, max_ov, num_pairs, rng, balance_overlap=False
):
    """Returns (rows, cols) sampling k pairs per view row for uniform view coverage."""
    n = overlap_min.shape[0]
    k = max(1, int(np.ceil(2 * num_pairs / n))) if num_pairs is not None else 1
    sel_rows, sel_cols = [], []
    for i in range(n):
        valid = np.where(
            (overlap_min[i, i + 1 :] >= min_ov) & (overlap_max[i, i + 1 :] <= max_ov)
        )[0] + (i + 1)
        if len(valid) == 0:
            continue
        if k is not None and len(valid) > k:
            if balance_overlap:
                chosen = _balance_overlap_sample(overlap_min[i, valid], k, rng)
                valid = valid[chosen]
            else:
                valid = rng.choice(valid, k, replace=False)
        sel_rows.append(np.full(len(valid), i, dtype=np.intp))
        sel_cols.append(valid)
    if not sel_rows:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    rows = np.concatenate(sel_rows)
    cols = np.concatenate(sel_cols)
    if num_pairs is not None and len(rows) > num_pairs:
        sel = rng.choice(len(rows), num_pairs, replace=False)
        rows, cols = rows[sel], cols[sel]
    return rows, cols


def _build_overlap_matrix(sg, n):
    """Builds the normalised symmetric overlap matrix from a scene group."""
    from scipy.sparse import coo_matrix

    raw = sg["overlaps"][()].astype(np.float32)
    raw[:, 2] /= 100.0
    dcoverage = (
        sg["depth_coverage"][()].astype(np.float32) / 100.0
        if "depth_coverage" in sg
        else np.ones(n, dtype=np.float32)
    )
    rc = raw[:, :2].astype(int)
    mat = coo_matrix((raw[:, 2], (rc[:, 0], rc[:, 1])), shape=(n, n)).tocsr()
    mat /= dcoverage.clip(min=1e-12)[:, None]
    return mat.toarray() + np.eye(n, dtype=np.float32)


def _scene_cache_key(conf, split, seed, h5_file):
    keys = [
        "use_pairs",
        "balance_views",
        "balance_overlap",
        "min_overlap",
        "max_overlap",
        f"{split}_num_per_scene",
        f"{split}_scenes",
    ]
    conf_dict = {k: conf.get(k) for k in keys}
    st = h5_file.stat()
    blob = json.dumps(
        {
            "conf": conf_dict,
            "split": split,
            "seed": seed,
            "file": str(h5_file),
            "size": st.st_size,
            "mtime": st.st_mtime,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _scene_cache_path(scene, h5_file):
    return h5_file.parent / f"{scene}_sample_cache.npz"


class _H5Split(torch.utils.data.Dataset):
    def __init__(self, conf, h5_path, split, seed=None):
        self.conf = conf
        self.h5_path = h5_path
        self.split = split
        self.seed = seed if seed is not None else conf.seed
        self._h5_files = {}  # lazy open per worker, keyed by file path string
        self._views_groups = {}  # cached "views" group per scene, keyed by scene name
        self._is_dir = h5_path.is_dir()

        scenes_conf = conf.get(f"{split}_scenes")
        # Read scene list from txt file, list of strings, or None = all scenes in h5 file(s)
        if scenes_conf is None:
            self.scenes = None  # resolved in _build_scene_index
        elif isinstance(scenes_conf, (str, Path)):
            self.scenes = sorted(
                set(Path(scenes_conf).read_text().rstrip("\n").split("\n"))
            )
        else:
            self.scenes = sorted(set(scenes_conf))

        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)

        logger.info(
            "Building scene index for %s (%d scenes)",
            conf.data_path,
            len(self.scenes) if self.scenes is not None else -1,
        )
        self.scene_to_h5 = self._build_scene_index()

        self.items = []
        self.scene_num_views = {}  # scene → total view count, populated during sampling

        self.sample_groups(self.seed)

    def _build_scene_index(self):
        """Returns dict mapping scene name → h5 file path."""
        if not self._is_dir:
            # single file: scan top-level keys
            with h5py.File(self.h5_path, "r") as h5:
                keys = h5.keys()
                return {
                    s: self.h5_path
                    for s in keys
                    if self.scenes is None or s in self.scenes
                }
        # directory: scan all .h5 files and collect their top-level keys
        scene_to_h5 = {}
        # easy way
        if self._is_dir:
            for scene in self.scenes or []:
                if (self.h5_path / f"{scene}.h5").exists():
                    scene_to_h5[scene] = self.h5_path / f"{scene}.h5"
        if self.scenes is not None and len(scene_to_h5) == len(self.scenes):
            return scene_to_h5
        logger.info(
            "Scene index incomplete (%d/%d), scanning all .h5 files in %s",
            len(scene_to_h5),
            len(self.scenes) if self.scenes is not None else -1,
            self.h5_path,
        )
        # fallback: scan all .h5 files and collect their top-level keys
        for h5_file in sorted(self.h5_path.glob("*.h5")):
            if h5_file.stem in self.scenes:
                scene_to_h5[h5_file.stem] = h5_file
                continue
            with h5py.File(h5_file, "r") as h5:
                for scene in h5.keys():
                    if self.scenes is None or scene in self.scenes:
                        scene_to_h5[scene] = h5_file
        return scene_to_h5

    def _get_scene_group(self, scene):
        h5_file = self.scene_to_h5[scene]
        key = str(h5_file)
        if key not in self._h5_files:
            self._h5_files[key] = h5py.File(h5_file, "r", rdcc_nbytes=0)
        return self._h5_files[key][scene]

    def _get_views_group(self, scene):
        if scene not in self._views_groups:
            self._views_groups[scene] = self._get_scene_group(scene)["views"]
        return self._views_groups[scene]

    def sample_scene(self, scene, sg, seed):
        """Sample pairs for a single scene group. Returns (items, num_views)."""
        num_per_scene = self.conf.get(f"{self.split}_num_per_scene")
        balance_views = self.conf.get("balance_views", False)
        balance_overlap = self.conf.get("balance_overlap", False)

        if self.conf.use_pairs and "pairs" in sg:
            n = int(sg.attrs["num_views"])
            pairs = sg["pairs"][()].astype(np.float32)
            pairs[:, 2] /= 100.0
            ov = pairs[:, 2]
            mask = (ov >= self.conf.min_overlap) & (ov <= self.conf.max_overlap)
            pairs = pairs[mask]
            if len(pairs) == 0:
                return [], n
            if num_per_scene is not None and len(pairs) > num_per_scene:
                rng = np.random.RandomState(seed)
                if balance_views:
                    sel = _balance_views_sample(
                        pairs[:, :2].astype(int),
                        n,
                        num_per_scene,
                        rng,
                        overlap_vals=pairs[:, 2] if balance_overlap else None,
                    )
                elif balance_overlap:
                    sel = _balance_overlap_sample(pairs[:, 2], num_per_scene, rng)
                else:
                    sel = rng.choice(len(pairs), num_per_scene, replace=False)
                pairs = pairs[sel]
            items = [
                (
                    scene,
                    (int(id0), int(id1)),
                    np.array([[1.0, ov], [ov, 1.0]], dtype=np.float32),
                )
                for id0, id1, ov in pairs
            ]
            return items, n

        elif "overlaps" in sg:
            n = int(sg.attrs["num_views"])
            mat = _build_overlap_matrix(sg, n)
            overlap_min = np.minimum(mat, mat.T)
            overlap_max = overlap_min
            rng = np.random.RandomState(seed)
            if balance_views:
                rows, cols = _sample_stratified_pairs_from_overlaps(
                    overlap_min,
                    overlap_max,
                    self.conf.min_overlap,
                    self.conf.max_overlap,
                    num_per_scene,
                    rng,
                    balance_overlap=balance_overlap,
                )
            else:
                rows, cols = np.where(
                    (overlap_min >= self.conf.min_overlap)
                    & (overlap_max <= self.conf.max_overlap)
                )
                mask = rows < cols
                rows, cols = rows[mask], cols[mask]
                if num_per_scene is not None and len(rows) > num_per_scene:
                    if balance_overlap:
                        sel = _balance_overlap_sample(
                            overlap_min[rows, cols], num_per_scene, rng
                        )
                    else:
                        sel = rng.choice(len(rows), num_per_scene, replace=False)
                    rows, cols = rows[sel], cols[sel]
            if len(rows) == 0:
                return [], n
            overlap_mats = np.empty((len(rows), 2, 2), dtype=np.float32)
            overlap_mats[:, 0, 0] = 1.0
            overlap_mats[:, 1, 1] = 1.0
            overlap_mats[:, 0, 1] = mat[rows, cols]
            overlap_mats[:, 1, 0] = mat[cols, rows]
            items = [
                (scene, (int(i), int(j)), om)
                for i, j, om in zip(rows, cols, overlap_mats)
            ]
            return items, n

        else:
            logger.warning("Scene %s has no pairs or overlaps, skipping.", scene)
            return [], 0

    def sample_scene_cached(self, scene, h5_file, seed):
        """Load sampling for one scene from npz cache. Returns (None, None) on miss."""
        from filelock import FileLock

        key = _scene_cache_key(self.conf, self.split, seed, h5_file)
        cp = _scene_cache_path(scene, h5_file)
        if not cp.exists():
            return None, None
        # Same lock as save_scene_cache's atomic rename, so a read can never
        # interleave with a concurrent writer swapping the underlying inode
        # (which otherwise surfaces as a stale-file-handle error on NFS).
        with FileLock(str(cp) + ".lock"):
            if not cp.exists():
                return None, None
            try:
                npz = np.load(cp, allow_pickle=False)
            except (EOFError, ValueError, zipfile.BadZipFile):
                logger.warning(
                    "Corrupt sampling cache %s — deleting and recomputing.", cp
                )
                cp.unlink(missing_ok=True)
                return None, None
            if f"{key}_ids" not in npz:
                return None, None
            items = [
                (scene, (int(i[0]), int(i[1])), ov)
                for i, ov in zip(npz[f"{key}_ids"], npz[f"{key}_overlaps"])
            ]
            return items, int(npz[f"{key}_num_views"])

    def save_scene_cache(self, scene, h5_file, seed, items, num_views):
        from filelock import FileLock

        key = _scene_cache_key(self.conf, self.split, seed, h5_file)
        cp = _scene_cache_path(scene, h5_file)
        with FileLock(str(cp) + ".lock"):
            try:
                data = dict(np.load(cp, allow_pickle=False)) if cp.exists() else {}
            except (EOFError, ValueError, zipfile.BadZipFile):
                logger.warning("Corrupt sampling cache %s — overwriting.", cp)
                data = {}
            data[f"{key}_ids"] = (
                np.array([[it[1][0], it[1][1]] for it in items], dtype=np.int32)
                if items
                else np.zeros((0, 2), dtype=np.int32)
            )
            data[f"{key}_overlaps"] = (
                np.array([it[2] for it in items], dtype=np.float32)
                if items
                else np.zeros((0, 2, 2), dtype=np.float32)
            )
            data[f"{key}_num_views"] = np.int32(num_views)
            tmp = cp.with_name(cp.stem + ".tmp.npz")
            np.savez(tmp, **data)
            tmp.replace(cp)  # atomic on POSIX — readers never see a partial write

    def sample_groups(self, seed):
        self.items = []
        for scene, h5_file in tqdm(
            self.scene_to_h5.items(), desc=f"Sampling {self.conf.data_path} groups"
        ):
            scene_items, num_views = None, None
            if self.conf.cache_sampling:
                scene_items, num_views = self.sample_scene_cached(scene, h5_file, seed)
            if scene_items is None:
                with h5py.File(h5_file, "r", rdcc_nbytes=0) as f:
                    scene_items, num_views = self.sample_scene(scene, f[scene], seed)
                if self.conf.cache_sampling:
                    self.save_scene_cache(scene, h5_file, seed, scene_items, num_views)
            if num_views > 0:
                self.scene_num_views[scene] = num_views
            self.items.extend(scene_items)
        np.random.RandomState(seed).shuffle(self.items)

    def worker_init(self, worker_id):
        if not self.conf.preload_files:
            return
        logger.info(f"Worker {worker_id} preloading h5 files and view groups...")
        self._h5_files = {}
        self._views_groups = {}
        for scene, h5_file in self.scene_to_h5.items():
            key = str(h5_file)
            if key not in self._h5_files:
                self._h5_files[key] = h5py.File(h5_file, "r", rdcc_nbytes=0)
            self._views_groups[scene] = self._h5_files[key][scene]["views"]

    def _read_view(self, scene, idx):
        view = self._get_views_group(scene)[str(idx)]

        if self.conf.read_image:
            raw = view["image"][()]
            if raw.ndim == 1:  # JPEG bytes
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:  # raw uint8 [H, W, C]
                img = raw[..., ::-1].copy()  # BGR → RGB
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        else:
            img = torch.zeros(3, 1, 1)

        K = view["K"][()].astype(np.float32)

        if "K_pinhole" in view:
            # then K is a full perspective matrix!
            camera = reconstruction.PerspectiveCamera.from_calibration_matrix(
                K, hw=img.shape[-2:]
            )
        else:
            camera = reconstruction.Camera.from_calibration_matrix(K)

        c_T_w = view["c_T_w"][()].astype(np.float32)
        name = view["name"][()].decode() if "name" in view else str(idx)

        depth = None
        if self.conf.read_depth and "depth" in view:
            depth = torch.from_numpy(view["depth"][()].astype(np.float32)).unsqueeze(0)
            if self.conf.use_valid_mask:
                if "valid" not in view:
                    raise ValueError(
                        f"use_valid_mask=True but no 'valid' dataset in view {name}"
                    )
                valid = torch.from_numpy(view["valid"][()].astype(bool)).unsqueeze(0)
                depth = depth * valid

        data = self.preprocessor(img)
        if depth is not None:
            data["depth"] = self.preprocessor.interpolate(
                depth, data["transform"], data["image"].shape[-2:], mode="nearest"
            )[0]
        else:
            data["depth"] = None

        data.update(
            {
                "name": name,
                "scene": scene,
                "T_w2cam": reconstruction.Pose.from_4x4mat(c_T_w),
                "camera": camera.float().compose_image_transform(
                    data["transform"], hw=data["image"].shape[-2:]
                ),
            }
        )
        return data

    def _getitem(self, idx):
        scene, (id0, id1), overlap = self.items[idx]
        v0, v1 = self._read_view(scene, id0), self._read_view(scene, id1)
        data = {
            "view0": v0,
            "view1": v1,
            "T_0to1": v1["T_w2cam"] @ v0["T_w2cam"].inv(),
            "T_1to0": v0["T_w2cam"] @ v1["T_w2cam"].inv(),
            "overlap_0to1": overlap[0, 1],
            "overlap_1to0": overlap[1, 0],
            "overlap": overlap,
            "scene": scene,
            "idx": idx,
            "name": f"{scene}/{v0['name']}_{v1['name']}",
        }
        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with tools.fork_rng(self.seed + idx, False):
                return self._getitem(idx)
        return self._getitem(idx)

    def __len__(self):
        return len(self.items)
