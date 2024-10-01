import os
import pickle
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.utils.desc_evaluation import compute_homography, compute_matching_score
from gluefactory.utils.kp_evaluation import compute_rep_loc_H
from gluefactory.utils.ls_evaluation import (
    compute_loc_error,
    compute_repeatability,
    match_segments_to_distance,
)
from gluefactory.visualization.viz2d import (
    plot_images,
    plot_keypoints,
    plot_lines,
    save_plot,
)

from ..datasets import get_dataset
from ..datasets.wireframe import get_lines, get_lines_gt
from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH

# from ..utils.export_predictions import export_predictions
from ..utils.tensor import batch_to_device, map_tensor
from ..utils.tools import AUCMetric
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import (
    eval_homography_dlt,
    eval_homography_robust,
    eval_matches_homography,
    eval_poses,
    get_matches_scores,
)

from gluefactory.visualization.viz2d import (
    plot_images,
    plot_keypoints,
    plot_lines,
    save_plot,
)

from ..utils.ls_evaluation import get_structural_dist, get_area_line_dist, get_orth_dist

# from .ls_evaluation import

"""

USAGE:

python -m gluefactory.eval.wireframe --conf ./gluefactory/configs/superpoint+lsd+gluestick-no-matcher.yaml --overwrite

TODO eval size and config integration
    For now you can edit datasets/wireframe.py at ln 70
"""

IMAGE_SHAPE = [640, 480]
PLOT_IMG = True


@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = model.to(device).eval()

    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        out = model(data)
        pred = {**out, **data}

        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("lines"):
                idx = k.replace("lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                if pred[k].shape[1] >= 1:
                    pred[k] = pred[k] * scales[None]
                else:
                    pred[k] = torch.arange(16).reshape(1, 4,2,2)

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        try:
            name = data["name"][0]
            grp = hfile.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
        except RuntimeError:
            continue

        del pred
    hfile.close()
    return output_file


class WireframePipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "wireframe",
            "num_workers": 1,
            "preprocessing": {
                "resize": IMAGE_SHAPE  # [320, 240],  # we also resize during eval to have comparable metrics
                # "side": "short",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            },
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": -1.0,  # -1 runs a bunch of thresholds and selects the best
            "distance": "structural",
            "distance_thresh": [1, 3, 5, 7, 100, 150, 200]
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "lines0",
        "lines1",
        "line_ends",
    ]

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset("wireframe")(data_conf)
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

        # pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)

            # Load Prediction and Ground Truth
            lines_pred = pred["lines0"]
            lines_gt = pred["line_ends"]

            scales = data[f"view0"]["scales"]
            lines_gt = lines_gt * scales[None]


            # Make Plot Directory
            if PLOT_IMG:
                os.makedirs("./wireframe_plots", exist_ok=True)

            # Get Line Distances
            dist_name = conf.distance
            if dist_name == "structural":
                line_dist = get_structural_dist(lines_gt, lines_pred)
            elif dist_name == "orthogonal":
                line_dist = get_orth_dist(lines_gt, lines_pred)
            elif dist_name == "area":
                line_dist = get_area_line_dist(lines_gt, lines_pred)
            else:
                raise NotImplementedError(f"{dist_name} is not an implemented Distance Measure")
            

            if PLOT_IMG:
                plot_images([data['view0']['image'][0].permute(1,2,0)/data['view0']['image'].max(), data['view1']['image'][0].permute(1,2,0)/data['view1']['image'].max()], ['Pred', 'GT'])
                plot_keypoints(kpts=[pred['keypoints0'], pred['keypoints1']])
                if pred['lines0'].shape[1] > 0:
                    plot_lines(lines= [pred['lines0'], lines_gt])
                save_plot(os.path.join('./wireframe_plots/', f'{i}.jpg'))

            # Get Closest Line Distances
            best_match = line_dist.min(axis=1)

            # Get Scores at Thresholds
            for thresh in conf.distance_thresh:
                TP = (best_match < thresh).sum().item()
                results[f"TP_@{thresh}"].append(TP)
                results[f"FN_@{thresh}"].append(len(best_match) - TP)
                results[f"Recall_@{thresh}"].append(TP/len(best_match))


        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.median(arr), 3)
            summaries[f"M{k}"] = round(np.mean(arr), 3)


        # figures = {
        #     "homography_recall": plot_cumulative(
        #         {
        #             "DLT": results["H_error_ransac"],
        #             self.conf.eval.estimator: results["H_error_ransac"],
        #         },
        #         [0, 10],
        #         unit="px",
        #         title="Homography ",
        #     )
        # }
        figures = {}

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(WireframePipeline.default_conf)

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

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint).parent
        experiment_dir = experiment_dir / ckpt_dir.name

    experiment_dir.mkdir(exist_ok=True)

    pipeline = WireframePipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
