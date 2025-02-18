import os
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torch.utils.data
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.datasets.homographies_deeplsd import warp_lines
from gluefactory.eval.eval_pipeline import (
    EvalPipeline,
    exists_eval,
    load_eval,
    save_eval,
)
from gluefactory.eval.io import get_eval_parser, load_model, parse_eval_args
from gluefactory.models import BaseModel
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.models.lines.line_utils import H_estimation
from gluefactory.models.utils.metrics_lines import (
    compute_loc_error,
    compute_repeatability,
)
from gluefactory.settings import EVAL_PATH
from gluefactory.utils.export_predictions import export_predictions
from gluefactory.utils.tensor import map_tensor
from gluefactory.visualization.viz2d import (
    plot_images,
    plot_keypoints,
    plot_lines,
    save_plot,
)


class HPatchesPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "hpatches",
            "num_workers": 2,
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
        "use_points": False,
        "use_lines": True,
        "H_err_thresh": [1, 3, 5],
    }
    export_keys = []

    optional_export_keys = [
        "lines0",
        "lines1",
        "orig_lines0",
        "orig_lines1",
        "line_matches0",
        "line_matches1",
        "line_matching_scores0",
        "line_matching_scores1",
        "line_distances",
    ]

    def _init(self, conf):
        if conf.use_points:
            self.export_keys += [
                "keypoints0",
                "keypoints1",
                "keypoint_scores0",
                "keypoint_scores1",
                "matches0",
                "matches1",
                "matching_scores0",
                "matching_scores1",
            ]
        if conf.use_lines:
            self.export_keys += [
                "lines0",
                "lines1",
                "line_matches0",
                "line_matches1",
                "orig_lines0",
                "orig_lines1",
            ]

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset("hpatches")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(
        self, experiment_dir: Path, model: BaseModel = None, overwrite: bool = False
    ) -> Path:
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

    def run(
        self,
        experiment_dir: Path,
        model: BaseModel = None,
        overwrite=False,
        overwrite_eval=False,
        plot=False,
    ):
        """Run export+eval loop"""
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite
        )

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            s, f, r = self.run_eval(self.get_dataloader(), pred_file, plot)
            save_eval(experiment_dir, s, f, r)
        s, r = load_eval(experiment_dir)
        return s, f, r

    def run_eval(
        self, loader: torch.utils.data.DataLoader, pred_file: Path, plot: bool
    ):
        assert pred_file.exists()
        results = defaultdict(list)

        conf = self.conf.eval
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            # if i in range(360,365):
            #     continue
            pred = cache_loader(data)
            # Remove batch dimension
            data = map_tensor(data, lambda t: torch.squeeze(t, dim=0))
            # add custom evaluations here

            segs1, segs2, matched_idx1, matched_idx2 = (
                pred["lines0"],
                pred["lines1"],
                pred["line_matches0"].to(torch.int64),
                pred["line_matches1"].to(torch.int64),
            )
            results_i = {}

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            results_i["scenes"] = data["scene"][0]
            H = data["H_0to1"].cpu().numpy()

            for thresh in [1, 3, 5]:
                if len(matched_idx1) < 3:
                    results_i[f"H_err@{thresh}"] = 0
                else:
                    matched_seg1 = segs1[matched_idx1].cpu().numpy()
                    matched_seg2 = warp_lines(segs2.cpu().numpy(), H)[matched_idx2]
                    score = H_estimation(
                        matched_seg1,
                        matched_seg2,
                        H,
                        data["view0"]["image"].shape[1:],
                        reproj_thresh=thresh,
                    )[0]
                    results_i[f"H_err@{thresh}"] = score * 1

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        figures = {}

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
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
        plot=args.plot,
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
