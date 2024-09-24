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
)

# from .ls_evaluation import

IMAGE_SHAPE = [640, 480]


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
    model = model.to(device).eval()

    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
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
                # comment for POLD2, uncomment for SP+LSD
                if (len(pred[f"keypoints{idx}"][0].shape)) == 3:
                    pred[k] = pred[k] * scales[None]
                else:
                    pred[k] = (
                        #For JPLDD Line detection
                        #(pred[f"keypoints{idx}"][0][pred[k][0][:,2].to(torch.int32)].reshape(1, -1, 2))
                        (pred[f"keypoints{idx}"][0][pred[k]].reshape(1, -1, 2))
                        .reshape(-1, 2, 2)
                        .unsqueeze(0)
                    )
            if k.startswith("orig_lines"):
                idx = k.replace("orig_lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]

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


class HPatchesPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "hpatches",
            "num_workers": 16,
            "preprocessing": {
                "resize": IMAGE_SHAPE  # [320, 240],  # we also resize during eval to have comparable metrics
                # "side": "short",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            },
            "matcher": {"name": "nn_point_line"},
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": -1.0,  # -1 runs a bunch of thresholds and selects the best
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
        "keypoint_scores0",
        "keypoint_scores1",
        "descriptors0",
        "descriptors1",
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
            # if 'keypoints0' in pred:
            #     del pred['keypoints0']
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

            if "lines0" in pred or "keypoints0" in pred:

                # Individual KP metrics
                m0 = pred["matches0"] > -1
                m1 = pred["matches0"][m0]
                kp0 = pred["keypoints0"][m0]
                kp1 = pred["keypoints1"][m1]
                desc0 = pred["descriptors0"][m0]
                desc1 = pred["descriptors1"][m1]

                rep, loc = compute_rep_loc_H(
                    np.concatenate(
                        [kp0, pred["keypoint_scores0"][m0].reshape(-1, 1)], axis=-1
                    ),
                    np.concatenate(
                        [kp1, pred["keypoint_scores1"][m1].reshape(-1, 1)], axis=-1
                    ),
                    data["H_0to1"].numpy(),
                    IMAGE_SHAPE,
                )

                correctness1, correctness3, correctness5 = compute_homography(
                    np.concatenate(
                        [kp0, pred["keypoint_scores0"][m0].reshape(-1, 1)], axis=-1
                    ),
                    np.concatenate(
                        [kp1, pred["keypoint_scores1"][m1].reshape(-1, 1)], axis=-1
                    ),
                    desc0,
                    desc1,
                    data["H_0to1"].numpy(),
                    IMAGE_SHAPE,
                )

                match_score = compute_matching_score(
                    pred["keypoints0"].numpy(),
                    pred["keypoints1"].numpy(),
                    m0,
                    m1,
                    data["H_0to1"].numpy(),
                    IMAGE_SHAPE,
                    keep_k_points=500,
                    thresh=[1, 3, 5],
                )
                """
                if match_score == 0:
                    plot_images([data['view0']['image'].permute(1,2,0), data['view1']['image'].permute(1,2,0)], ['H0', 'H1'])
                    plot_keypoints(kpts=[pred['keypoints0'][m0], pred['keypoints1'][m1]])  
                    save_plot(os.path.join('./match_score/', f'{i}.jpg'))
                """

                results["Desc_correctness1"].append(float(correctness1))
                results["Desc_correctness3"].append(float(correctness3))
                results["Desc_correctness5"].append(float(correctness5))
                results["Match_score1"].append(float(match_score[0]))
                results["Match_score3"].append(float(match_score[1]))
                results["Match_score5"].append(float(match_score[2]))
                results["Point_repeatability"].append(rep)
                results["Point_localization"].append(loc)

                if "lines0" in pred:

                    # Individual Line metrics
                    m0 = pred["line_matches0"] > -1
                    m1 = pred["line_matches0"][m0]
                    line_seg0 = pred["lines0"][m0]
                    line_seg1 = pred["lines1"][m1]
                    # line_seg0 = np.concatenate([pred['keypoints0'][pred['lines0'][m0][0]], pred['keypoints0'][pred['lines0'][m0][1]]], axis = 1).reshape(-1, 2, 2)
                    # line_seg1 = np.concatenate([pred['keypoints1'][pred['lines1'][m1][0]], pred['keypoints1'][pred['lines1'][m1][1]]], axis = 1).reshape(-1, 2, 2)
                    distances = match_segments_to_distance(
                        line_seg0, line_seg1, data["H_0to1"]
                    )
                    line_rep = compute_repeatability(
                        line_seg0, line_seg1, distances, [1, 3, 5]
                    )
                    line_loc_error = compute_loc_error(distances, [1, 3, 5])

                    ## TODO : ADD CONFIG FOR SAVING IMAGES

                    # if line_rep[0] == 0:
                    #     print("Rep 0")
                    #     plot_images([data['view0']['image'].permute(1,2,0), data['view1']['image'].permute(1,2,0)], ['H0', 'H1'])
                    #     plot_keypoints(kpts=[pred['keypoints0'], pred['keypoints1']])
                    #     plot_lines(lines= [pred['lines0'], pred['lines1']])
                    #     save_plot(os.path.join('./match_score/', f'{i}.jpg'))
                    results["line_rep_1px"].append(line_rep[0])
                    results["line_rep_3px"].append(line_rep[1])
                    results["line_rep_5px"].append(line_rep[2])
                    results["line_loc_error_1px"].append(line_loc_error[0])
                    results["line_loc_error_3px"].append(line_loc_error[1])
                    results["line_loc_error_5px"].append(line_loc_error[2])

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
                    "DLT": results["H_error_ransac"],
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

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint).parent
        experiment_dir = experiment_dir / ckpt_dir.name

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
