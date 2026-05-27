"""InLoc Evaluation Pipeline.

Uses the hloc reconstruction pipeline to extract features per-image and match
pairs, avoiding redundant image loads.  The eval loop iterates over *queries*
(not pairs): ``num_samples`` limits the number of queries evaluated.
"""

import collections
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import pycolmap
from hloc import localize_inloc
from hloc.utils.io import write_poses
from hloc.utils.parsers import names_to_pair, parse_retrieval
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from .. import datasets, pipelines, settings
from ..datasets.inloc_pairs import names_to_pair as pair_name_for_dataset
from ..pipelines.reconstruction.base import ReconstructionPipeline
from ..utils import types
from ..utils.export import write_tree_h5
from ..visualization.two_view_frame import TwoViewFrame
from . import io
from .eval_pipeline import EvalPipeline, exists_eval, load_eval, save_eval

logger = logging.getLogger(__name__)


def pose_from_cluster(
    dataset_dir,
    q,
    retrieved,
    match_h5,
    feature_h5,
    skip=None,
    db_keypoints=None,
    query_keypoints=None,
    threshold: float | None = 0.7,
):
    """Estimate absolute pose for query *q* from a cluster of DB images.

    Args:
        db_keypoints: If set, read this key from match_h5[pair] instead of
            feature_h5[db_image]["keypoints"] (refined positions per pair).
        query_keypoints: If set, read this key from match_h5[pair] instead of
            feature_h5[query]["keypoints"] (refined positions per pair).

    Returns:
        ret: PnP result dict (or None if no valid matches).
        data: defaultdict(list) with per-DB-image arrays.
        cat_data: concatenated version of *data* (or empty dict).
        num_matches: total number of raw matches across all DB images.
    """
    # height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    width, height = Image.open(dataset_dir / q).size
    cx = 0.5 * width
    cy = 0.5 * height
    focal_length = 4032.0 * 28.0 / 36.0

    data = collections.defaultdict(list)
    num_matches = 0

    # Default query keypoints from feature file (may be overridden per-pair below)
    kpq_default = feature_h5[q]["keypoints"].__array__()

    for i, r in enumerate(retrieved):
        pair = names_to_pair(q, r)
        if pair not in match_h5:
            continue

        # Use overridden keypoints from match_h5 if configured, else feature_h5
        kpq = (
            match_h5[pair][query_keypoints].__array__()
            if query_keypoints and query_keypoints in match_h5[pair]
            else kpq_default
        )
        kpr = (
            match_h5[pair][db_keypoints].__array__()
            if db_keypoints and db_keypoints in match_h5[pair]
            else feature_h5[r]["keypoints"].__array__()
        )
        m = match_h5[pair]["matches0"].__array__()
        v = m > -1

        if threshold is not None:
            ms = match_h5[pair]["matching_scores0"].__array__()
            v = v & (ms > threshold)

        if skip and (np.count_nonzero(v) < skip):
            continue

        v_indices = np.where(v)[0]
        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)
        scan_r = localize_inloc.loadmat(Path(dataset_dir, r + ".mat"))["XYZcut"]
        mkp3d, valid = localize_inloc.interpolate_scan(scan_r, mkpr)
        Tr = localize_inloc.get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        data["mkpq"].append(mkpq[valid])
        data["mkpr"].append(mkpr[valid])
        data["mkp3d"].append(mkp3d[valid])
        data["indices"].append(np.full(np.count_nonzero(valid), i))
        data["query_kp_idx"].append(v_indices[valid])

    if not data["mkpq"]:
        return None, data, {}, num_matches

    cat_data = {k: np.concatenate(v, 0) for k, v in data.items()}

    cam = {
        "model": "SIMPLE_PINHOLE",
        "width": width,
        "height": height,
        "params": [focal_length, cx, cy],
    }
    estimation_options = pycolmap.AbsolutePoseEstimationOptions()
    estimation_options.ransac.max_error = 48

    ret = pycolmap.estimate_and_refine_absolute_pose(
        cat_data["mkpq"], cat_data["mkp3d"], cam, estimation_options
    )
    if ret is not None:
        ret["cfg"] = cam
    return ret, data, cat_data, num_matches


class InLocPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "root": "inloc",
            "num_workers": 8,
            "preprocessing": {
                "side": "long",
            },
            "pairs": "inloc/pairs/pairs-query-netvlad40-temporal.txt",
        },
        "model": {
            "ground_truth": {
                "name": None,
            },
            "allow_no_extract": True,
        },
        "eval": {
            "skip_matches": None,
            "db_keypoints": None,
            "query_keypoints": None,
            "scene": None,  # e.g. "DUC1" or "DUC2" to restrict to one scene
        },
        "pipeline": {
            "name": "reconstruction.hloc",
        },
    }

    main_metric = "num_localized"
    child_frame = TwoViewFrame
    default_x = "pnp_inliers"
    default_y = "num_matches"
    default_plot = "pnp_inliers"

    def _init(self, conf):
        self.root_dir = settings.DATA_PATH / conf.data.root
        pipeline_conf = OmegaConf.merge(
            conf.pipeline,
            {"data": conf.data},
        )
        self.pipeline: ReconstructionPipeline = pipelines.get_pipeline(
            ReconstructionPipeline, conf.pipeline.name
        )(pipeline_conf)

        print(self.pipeline.conf)

    def _load_retrieval(self):
        pairs_path = Path(self.conf.data.pairs)
        if not pairs_path.exists():
            pairs_path = settings.DATA_PATH / self.conf.data.pairs
        return parse_retrieval(pairs_path)

    def _build_reconstruction_data(self, queries, retrieval_dict, output_dir):
        """Build ReconstructionData for the given queries only.

        Writes a filtered pairs file to the location the hloc pipeline expects
        and collects only the images needed.
        """
        from ..pipelines.reconstruction.hloc import HlocPipeline

        all_images = set()
        for q in queries:
            all_images.add(q)
            all_images.update(retrieval_dict[q])

        # Write pairs file directly where the pipeline expects it
        pairs_file = HlocPipeline.PathConfig(output_dir).pairs_file
        pairs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pairs_file, "w") as f:
            for q in queries:
                for db in retrieval_dict[q]:
                    f.write(f"{q} {db}\n")

        return types.ReconstructionData(
            image_dir=self.root_dir,
            image_list=sorted(all_images),
        )

    @classmethod
    def get_dataloader(cls, data_conf=None):
        data_conf = data_conf if data_conf else cls.default_conf["data"]
        dataset = datasets.get_dataset("inloc_pairs")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(
        self, experiment_dir, queries, retrieval_dict, model=None, overwrite=False
    ):
        """Extract features per-image, then match pairs via hloc pipeline."""
        experiment_dir.mkdir(exist_ok=True, parents=True)

        from ..pipelines.reconstruction.hloc import HlocPipeline

        paths = HlocPipeline.PathConfig(experiment_dir)

        if (
            not overwrite
            and paths.feature_file.exists()
            and paths.matches_file.exists()
        ):
            return

        if model is None:
            model = io.load_model(self.conf.model, self.conf.checkpoint)

        data = self._build_reconstruction_data(queries, retrieval_dict, experiment_dir)

        # Step 1: extract features (one forward pass per unique image)
        self.pipeline.extract_features(experiment_dir, model, data)

        # Step 2: match pairs (loads cached features from h5)
        conf = self.conf.eval
        optional_keys = [
            k for k in [conf.get("db_keypoints"), conf.get("query_keypoints")] if k
        ]
        self.pipeline.match_features(
            experiment_dir, model, data, optional_keys=optional_keys
        )

    def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False):
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )

        retrieval_dict = self._load_retrieval()
        queries = list(retrieval_dict.keys())
        scene = self.conf.eval.get("scene", None)
        if scene is not None:
            queries = [q for q in queries if any(scene in d for d in retrieval_dict[q])]
        if self.conf.num_samples is not None:
            queries = queries[: self.conf.num_samples]

        self.get_predictions(
            experiment_dir,
            queries,
            retrieval_dict,
            model=model,
            overwrite=overwrite,
        )

        # Update conf to point to the filtered pairs file for inspect compat
        from ..pipelines.reconstruction.hloc import HlocPipeline

        pairs_file = HlocPipeline.PathConfig(experiment_dir).pairs_file
        if pairs_file.exists():
            OmegaConf.update(self.conf, "data.pairs", str(pairs_file))
            OmegaConf.save(self.conf, experiment_dir / "conf.yaml")

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            s, f, r = self.run_eval(queries, retrieval_dict, experiment_dir)
            save_eval(experiment_dir, s, f, r)
        s, r = load_eval(experiment_dir)
        return s, f, r

    def run_eval(self, queries, retrieval_dict, pred_dir):
        """Iterate over queries, run PnP, write pair-level predictions."""
        from ..pipelines.reconstruction.hloc import HlocPipeline

        paths = HlocPipeline.PathConfig(pred_dir)
        assert paths.feature_file.exists(), f"Missing {paths.feature_file}"
        assert paths.matches_file.exists(), f"Missing {paths.matches_file}"

        feature_h5 = h5py.File(str(paths.feature_file), "r")
        match_h5 = h5py.File(str(paths.matches_file), "r")

        conf = self.conf.eval
        poses = {}
        results = collections.defaultdict(list)

        pred_file = pred_dir / "predictions.h5"
        eval_pred_file = pred_dir / "eval_predictions.h5"
        pred_h5 = h5py.File(str(pred_file), "w")
        eval_h5 = h5py.File(str(eval_pred_file), "w")

        db_kp_key = conf.get("db_keypoints", None)
        query_kp_key = conf.get("query_keypoints", None)

        for q in tqdm(queries, desc="Evaluating queries"):
            dbs = retrieval_dict[q]
            ret, _, cat_data, _ = pose_from_cluster(
                self.root_dir,
                q,
                dbs,
                match_h5,
                feature_h5,
                skip=conf.get("skip_matches", None),
                db_keypoints=db_kp_key,
                query_keypoints=query_kp_key,
            )

            if ret is not None and ret.get("num_inliers", 0) > 0:
                poses[q] = ret["cam_from_world"]

            # Get inlier mask (over cat_data) from PnP result
            inlier_mask = None
            if ret is not None and cat_data and ret.get("num_inliers", 0) > 0:
                inlier_mask = np.asarray(ret["inlier_mask"], dtype=bool)

            kpq_default = feature_h5[q]["keypoints"].__array__()

            # Write per-pair data
            for i, r in enumerate(dbs):
                hloc_pair = names_to_pair(q, r)
                ds_pair = pair_name_for_dataset(q, r)

                if hloc_pair not in match_h5:
                    continue

                # Use overridden keypoints from match_h5 if configured
                kpq = (
                    match_h5[hloc_pair][query_kp_key].__array__()
                    if query_kp_key and query_kp_key in match_h5[hloc_pair]
                    else kpq_default
                )
                kpr = (
                    match_h5[hloc_pair][db_kp_key].__array__()
                    if db_kp_key and db_kp_key in match_h5[hloc_pair]
                    else feature_h5[r]["keypoints"].__array__()
                )
                m = match_h5[hloc_pair]["matches0"].__array__()
                ms = match_h5[hloc_pair]["matching_scores0"].__array__()

                pair_num_matches = int(np.count_nonzero(m > -1))

                # PnP inlier mask for this pair
                pnp_inlier = np.zeros(len(kpq), dtype=bool)
                pair_pnp_inliers = 0
                if inlier_mask is not None and cat_data:
                    pair_rows = cat_data["indices"] == i
                    if pair_rows.any():
                        pair_qkp_idx = cat_data["query_kp_idx"][pair_rows]
                        pair_inlier = inlier_mask[pair_rows]
                        pnp_inlier[pair_qkp_idx[pair_inlier]] = True
                        pair_pnp_inliers = int(pair_inlier.sum())

                # Write predictions.h5
                grp = pred_h5.create_group(ds_pair)
                write_tree_h5(
                    grp,
                    {
                        "keypoints0": kpq,
                        "keypoints1": kpr,
                        "matches0": m,
                        "matching_scores0": ms,
                    },
                )

                # Write eval_predictions.h5
                egrp = eval_h5.create_group(ds_pair)
                write_tree_h5(egrp, {"pnp_inlier0": pnp_inlier})

                # Pair-level results
                results["names"].append(ds_pair)
                results["query"].append(q)
                results["num_matches"].append(pair_num_matches)
                results["pnp_inliers"].append(pair_pnp_inliers)
                results["success"].append(
                    1 if ret is not None and ret.get("num_inliers", 0) > 0 else 0
                )

        pred_h5.close()
        eval_h5.close()
        feature_h5.close()
        match_h5.close()

        # Write poses
        pose_file = pred_dir / "poses.txt"
        logger.info(f"Writing {len(poses)} poses to {pose_file}")
        write_poses(poses, pose_file, prepend_camera_name=False)

        n_pairs = len(results["names"])

        # Query-level aggregates (sum matches/inliers per query)
        query_matches = collections.defaultdict(int)
        query_inliers = collections.defaultdict(int)
        for q_name, nm, ni in zip(
            results["query"], results["num_matches"], results["pnp_inliers"]
        ):
            query_matches[q_name] += nm
            query_inliers[q_name] += ni

        summaries = {
            "num_queries": len(queries),
            "num_localized": len(poses),
            "localization_rate": round(len(poses) / max(len(queries), 1), 3),
            "num_pairs": n_pairs,
            "mean_num_matches": (
                round(np.mean(list(query_matches.values())).item(), 1)
                if query_matches
                else 0
            ),
            "mean_num_inliers": (
                round(np.mean(list(query_inliers.values())).item(), 1)
                if query_inliers
                else 0
            ),
        }

        return summaries, {}, dict(results)


if __name__ == "__main__":
    io.run_cli(InLocPipeline, name=Path(__file__).stem)
