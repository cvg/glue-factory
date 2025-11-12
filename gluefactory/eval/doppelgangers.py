"""Doppelgangers Evaluation Pipeline."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import torch
from tqdm import tqdm

from .. import datasets
from ..models.cache_loader import CacheLoader
from ..utils import misc
from ..utils.export import export_predictions
from . import eval_pipeline, io


class DoppelgangersPipeline(eval_pipeline.EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "doppelgangers",
            "root": "doppelgangerspp",
            "num_workers": 16,
            "preprocessing": {
                "resize": 1024,  # we also resize during eval to have comparable metrics
                "side": "long",
                "crop_if_short_side": True,
            },
            "seed": 42,
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {"score_key": "overlaps"},
    }

    export_keys = (
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    )

    # For plotting
    default_x: str | None = "score"
    default_y: str | None = "label"

    optional_export_keys = (
        "matchability0",
        "matchability1",
        "overlap_score0",
        "overlap_score1",
    )

    def _init(self, conf):
        self.export_keys += [conf.eval.score_key]

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = datasets.get_dataset("doppelgangers")(data_conf)
        return dataset.get_data_loader("test", num_samples=self.num_samples)

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = io.load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

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
                results_i["score"] = scores[0].mean().item()
                for i in range(0, len(scores)):
                    results_i[f"score{i}"] = scores[i].item()
            else:
                results_i["score"] = scores.item()

            results_i["label"] = data["has_overlap"].item()
            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        figures = {}

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k in list(results.keys()):
            v = results[k]
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = np.mean(arr)
            if "score" in k:
                suffix = k.replace("score", "")
                precision, recall, ths = skm.precision_recall_curve(results["label"], v)
                summaries[f"auprc{suffix}"] = skm.auc(recall, precision)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1)
                summaries[f"best_threshold{suffix}"] = ths[best_idx]
                results[f"pred{suffix}"] = (v > ths[best_idx]).astype(np.float32)
                results[f"discrepancy{suffix}"] = (
                    results[f"pred{suffix}"] - results["label"]
                )  # +1: False Positive, -1: False Negative
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
                    arr[labels == 1], bins=20, alpha=0.5, label="overlap", range=(0, 1)
                )
                ax.legend()
                figures[f"hist{suffix}"] = fig

        summaries = {
            k: round(v, 3) if isinstance(v, float) else v for k, v in summaries.items()
        }

        return summaries, figures, results


if __name__ == "__main__":
    io.run_cli(DoppelgangersPipeline, name=Path(__file__).stem)
