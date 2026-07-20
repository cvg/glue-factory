"""Overlap Evaluation from Reconstruction.

Uses ETH3D COLMAP reconstructions to derive GT overlap labels from shared 3D
point tracks, then evaluates a model's overlap prediction.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import sklearn.metrics as skm
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils import misc
from ..utils.export import export_predictions
from ..utils.preprocess import ImagePreprocessor, load_image
from . import eval_pipeline, io
from .eth3d_reconstruction import ETH3D_SCENES


def build_covisibility(rec):
    """Build an (N x N) shared-point-count matrix from a COLMAP reconstruction.

    Args:
        rec: pycolmap.Reconstruction

    Returns:
        covis: np.ndarray of shape (N, N) with shared 3D point counts
        image_ids: list of image IDs (dense index corresponds to matrix row/col)
    """
    image_ids = list(rec.reg_image_ids())
    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}
    n = len(image_ids)
    covis = np.zeros((n, n), dtype=np.int32)

    for point in rec.points3D.values():
        track_img_ids = [el.image_id for el in point.track.elements]
        # Map to dense indices, skip unregistered
        idxs = [id_to_idx[iid] for iid in track_img_ids if iid in id_to_idx]
        for i, a in enumerate(idxs):
            for b in idxs[i + 1 :]:
                covis[a, b] += 1
                covis[b, a] += 1

    return covis, image_ids


class _PairDataset(torch.utils.data.Dataset):
    """Simple dataset that loads and preprocesses image pairs."""

    def __init__(self, pairs, preprocessor):
        self.pairs = pairs
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        view0 = self.preprocessor(load_image(p["view0_path"]))
        view1 = self.preprocessor(load_image(p["view1_path"]))
        view0["name"] = p["view0_name"]
        view1["name"] = p["view1_name"]
        return {
            "view0": view0,
            "view1": view1,
            "name": p["name"],
            "has_overlap": p["has_overlap"],
            "scene": p["scene"],
        }


class OverlapReconstructionPipeline(eval_pipeline.EvalPipeline):
    default_conf = {
        "data": {
            "root": "ETH3D_undistorted_resized",
            "batch_size": 1,
            "num_workers": 8,
            "preprocessing": {
                "resize": 1024,
                "side": "long",
                "square_pad": True,
            },
        },
        "model": {
            "ground_truth": {
                "name": None,
            },
        },
        "eval": {
            "score_key": "overlaps",
            "overlap_threshold": 5,
            "subset_idxs": None,
            "negatives_only": False,
        },
    }

    export_keys = (
        "keypoints0",
        "keypoints1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    )

    default_x: str | None = "score"
    default_y: str | None = "label"

    optional_export_keys = (
        "matchability0",
        "matchability1",
        "overlap_score0",
        "overlap_score1",
    )

    scenes = sum(ETH3D_SCENES.values(), [])

    def _init(self, conf):
        self.export_keys += [conf.eval.score_key]
        self.root_dir = DATA_PATH / conf.data.root
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory {self.root_dir} does not exist.")

    def _build_pairs(self):
        """Build all image pairs with GT overlap labels from COLMAP reconstructions.

        If ``conf.eval.negatives_only`` is True, only pairs *without* overlap
        (shared points < threshold) are returned.
        """
        threshold = self.conf.eval.overlap_threshold
        negatives_only = self.conf.eval.negatives_only
        pairs = []
        for scene in self.scenes:
            ref_sfm_dir = self.root_dir / scene / "dslr_calibration_undistorted"
            image_dir = self.root_dir / scene / "images"
            rec = pycolmap.Reconstruction(ref_sfm_dir)
            covis, image_ids = build_covisibility(rec)
            n = len(image_ids)

            # Map image_id -> filename
            id_to_name = {img.image_id: img.name for img in rec.images.values()}

            for i in range(n):
                for j in range(i + 1, n):
                    shared = int(covis[i, j])
                    has_overlap = int(shared >= threshold)
                    name_i = id_to_name[image_ids[i]]
                    name_j = id_to_name[image_ids[j]]
                    pair_name = f"{scene}/{name_i}_{name_j}"
                    if negatives_only and has_overlap:
                        continue
                    pairs.append(
                        {
                            "name": pair_name,
                            "scene": scene,
                            "has_overlap": has_overlap,
                            "shared_points": shared,
                            "view0_path": str(image_dir / name_i),
                            "view1_path": str(image_dir / name_j),
                            "view0_name": f"{scene}/{name_i}",
                            "view1_name": f"{scene}/{name_j}",
                        }
                    )
        return pairs

    @classmethod
    def get_dataloader(cls, data_conf=None):
        raise NotImplementedError("Use get_predictions / run_eval directly.")

    def _get_dataloader(self, pairs):
        """Build a DataLoader from the pair list."""
        preprocessor = ImagePreprocessor(self.conf.data.preprocessing)
        dataset = _PairDataset(pairs, preprocessor)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.conf.data.batch_size,
            num_workers=self.conf.data.num_workers,
            shuffle=False,
            collate_fn=None,  # default collate
        )

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = io.load_model(self.conf.model, self.conf.checkpoint)
            pairs = self._build_pairs()
            loader = self._get_dataloader(pairs)
            export_predictions(
                loader,
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False):
        """Run export+eval loop (overridden to pass pairs to run_eval)."""
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite
        )

        f = {}
        if (
            not eval_pipeline.exists_eval(experiment_dir)
            or overwrite_eval
            or overwrite
        ):
            pairs = self._build_pairs()
            loader = self._get_dataloader(pairs)
            s, f, r = self.run_eval(loader, pred_file)
            eval_pipeline.save_eval(experiment_dir, s, f, r)
        s, r = eval_pipeline.load_eval(experiment_dir)
        return s, f, r

    def run_eval(self, loader, pred_file):
        assert pred_file.exists()
        results = defaultdict(list)

        conf = self.conf.eval
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            # Remove batch dimension
            data = misc.map_tensor(data, lambda t: torch.squeeze(t, dim=0))

            results_i = {}
            scores = {**data, **pred}[conf.score_key]
            if scores.ndim == 1:
                results_i["score"] = scores[-1].mean().item()
                if conf.subset_idxs is not None:
                    for idx in conf.subset_idxs:
                        ii = idx if idx >= 0 else scores.shape[0] + idx
                        results_i[f"score{ii}"] = scores[ii].item()
            else:
                results_i["score"] = scores.item()

            results_i["label"] = data["has_overlap"].item()
            results_i["names"] = data["name"][0]
            results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        figures = {}

        # Summarize results
        summaries = {}
        for k in list(results.keys()):
            v = results[k]
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = np.mean(arr)
            if "score" in k:
                suffix = k.replace("score", "")
                precision, recall, ths = skm.precision_recall_curve(
                    results["label"], v
                )
                summaries[f"auprc{suffix}"] = skm.auc(recall, precision)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1)
                summaries[f"best_threshold{suffix}"] = ths[best_idx]
                results[f"pred{suffix}"] = (v > ths[best_idx]).astype(np.float32)
                results[f"discrepancy{suffix}"] = (
                    results[f"pred{suffix}"] - results["label"]
                )
                summaries[f"f1{suffix}"] = f1[best_idx]
                summaries[f"precision{suffix}"] = precision[best_idx]
                summaries[f"recall{suffix}"] = recall[best_idx]
                summaries[f"ap{suffix}"] = skm.average_precision_score(
                    results["label"], v
                )
                is_positive = results[f"pred{suffix}"] > 0.5
                gt_is_positive = np.array(results["label"]) > 0.5
                summaries[f"false_positive{suffix}"] = (
                    is_positive & ~gt_is_positive
                ).mean()
                summaries[f"false_negative{suffix}"] = (
                    gt_is_positive & ~is_positive
                ).mean()
                summaries[f"auroc{suffix}"] = np.nan_to_num(
                    skm.roc_auc_score(results["label"], v), 0.0
                )

                fig, ax = plt.subplots(1, 1)
                labels = np.array(results["label"])
                ax.hist(
                    arr[labels == 0],
                    bins=20,
                    alpha=0.5,
                    label="no overlap",
                    range=(0, 1),
                )
                ax.hist(
                    arr[labels == 1],
                    bins=20,
                    alpha=0.5,
                    label="overlap",
                    range=(0, 1),
                )
                ax.legend()
                figures[f"hist{suffix}"] = fig

        summaries = {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in summaries.items()
        }

        return summaries, figures, results


if __name__ == "__main__":
    io.run_cli(OverlapReconstructionPipeline, name=Path(__file__).stem)
