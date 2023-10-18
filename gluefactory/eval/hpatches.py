from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tensor import map_tensor
from ..utils.tools import AUCMetric
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import (
    eval_homography_dlt,
    eval_homography_robust,
    eval_matches_homography,
    eval_poses,
)


class HPatchesPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "hpatches",
            "num_workers": 16,
            "preprocessing": {
                "resize": 480,  # we also resize during eval to have comparable metrics
                "side": "short",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,  # -1 runs a bunch of thresholds and selects the best
        },
    }
    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]

    optional_export_keys = [
        "lines0",
        "lines1",
        "orig_lines0",
        "orig_lines1",
        "line_matches0",
        "line_matches1",
        "line_matching_scores0",
        "line_matching_scores1",
    ]

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset("hpatches")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
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

        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            # Remove batch dimension
            data = map_tensor(data, lambda t: torch.squeeze(t, dim=0))
            # add custom evaluations here
            if "keypoints0" in pred:
                results_i = eval_matches_homography(data, pred)
                results_i = {**results_i, **eval_homography_dlt(data, pred)}
            else:
                results_i = {}
            for th in test_thresholds:
                pose_results_i = eval_homography_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.median(arr), 3)

        auc_ths = [1, 3, 5]
        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=auc_ths, key="H_error_ransac", unit="px"
        )
        if "H_error_dlt" in results.keys():
            dlt_aucs = AUCMetric(auc_ths, results["H_error_dlt"]).compute()
            for i, ath in enumerate(auc_ths):
                summaries[f"H_error_dlt@{ath}px"] = dlt_aucs[i]

        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "homography_recall": plot_cumulative(
                {
                    "DLT": results["H_error_dlt"],
                    self.conf.eval.estimator: results["H_error_ransac"],
                },
                [0, 10],
                unit="px",
                title="Homography ",
            )
        }

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(HPatchesPipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = HPatchesPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
