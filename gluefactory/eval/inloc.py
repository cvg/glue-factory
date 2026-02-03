import collections
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np
import pycolmap
import torch
from hloc import localize_inloc
from hloc.utils.parsers import names_to_pair, parse_retrieval
from tqdm import tqdm

from .. import datasets, models, settings
from ..models.cache_loader import CacheLoader, recursive_load
from ..utils import export, misc, tools
from . import io, utils
from .eval_pipeline import EvalPipeline

logger = logging.getLogger(__name__)


def pose_from_cluster(dataset_dir, q, retrieved, pred_h5, skip=None):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    cx = 0.5 * width
    cy = 0.5 * height
    focal_length = 4032.0 * 28.0 / 36.0

    all_mkpq = []
    all_mkpr = []
    all_mkp3d = []
    all_indices = []

    data = collections.defaultdict(list)
    for i, r in enumerate(retrieved):
        pair = names_to_pair(q, r)
        grp = pred_h5[pair]
        pkeys = grp.keys() | grp.attrs.keys()
        pred = recursive_load(grp, pkeys)

        kpq = pred["keypoints0"].__array__()
        kpr = pred["keypoints1"].__array__()
        m = pred["matches0"].__array__()
        v = m > -1

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)
        scan_r = localize_inloc.loadmat(Path(dataset_dir, r + ".mat"))["XYZcut"]
        # scan_r = scans[i]
        mkp3d, valid = localize_inloc.interpolate_scan(scan_r, mkpr)
        Tr = localize_inloc.get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        data["mkpq"].append(mkpq[valid])
        data["mkpr"].append(mkpr[valid])
        data["mkp3d"].append(mkp3d[valid])
        data["indices"].append(np.full(np.count_nonzero(valid), i))

    cat_data = {k: np.concatenate(v, 0) for k, v in data.items()}

    cam = {
        "model": "SIMPLE_PINHOLE",
        "width": width,
        "height": height,
        "params": [focal_length, cx, cy],
    }
    estimation_options = pycolmap.AbsolutePoseEstimationOptions()
    estimation_options.ransac.max_error = 48  # default: 12
    # estimation_options.ransac.min_inlier_ratio = 0.01  # default: 0.1
    # estimation_options.ransac.min_num_trials = 1_000  # default: 100
    # estimation_options.ransac.max_num_trials = 100_000  # default: 10000

    # refinement_options = pycolmap.AbsolutePoseRefinementOptions()
    # refinement_options.gradient_tolerance = 1e-12  # default: 1e-4
    # refinement_options.print_summary = True
    ret = pycolmap.estimate_and_refine_absolute_pose(
        cat_data["mkpq"], cat_data["mkp3d"], cam, estimation_options
    )
    ret["cfg"] = cam
    return ret, data


class InLocPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "inloc_pairs",
            "preprocessing": {
                "side": "long",
            },
            "pairs": "inloc/pairs/pairs-query-netvlad40-temporal.txt",
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": ["pycolmap"],
            "ransac_th": 12.0,  # -1 runs a bunch of thresholds and selects the best
            "n_processes": None,  # 0 is sequential
            "max_tasks": 500,  # max tasks in the pool
        },
    }

    main_metric = "rel_pose_error_mAA"

    export_keys = (
        "keypoints0",
        "keypoints1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    )
    optional_export_keys = ()

    def _init(self, conf):
        logger.info("InLoc Pipeline initialized.")

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = datasets.get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test", num_samples=self.num_samples)

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = io.load_model(self.conf.model, self.conf.checkpoint)
            export.export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = collections.defaultdict(list)
        h5f = h5py.File(str(pred_file), "r")

        retrieval_dict = parse_retrieval(self.conf.data.pairs)
        results = []
        poses = []

        for q, dbs in tqdm(
            retrieval_dict.items(), desc="Evaluation: ", total=len(retrieval_dict)
        ):
            q_ret, q_data = pose_from_cluster(
                settings.DATA_PATH / self.conf.data.root,
                q,
                dbs,
                h5f,
                skip=conf.get("min_matches", None),
            )
            poses[q] = q_ret["cam_from_world"]

        pose_file = pred_file.parent / "poses.txt"
        logger.info(f"Writing poses to {pose_file}...")
        localize_inloc.write_poses(poses, pose_file, prepend_camera_name=False)
        summaries = {}
        figures = {}
        return summaries, figures, results
