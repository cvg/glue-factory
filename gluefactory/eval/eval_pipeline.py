import collections
import json
import logging
import multiprocessing as mp
from typing import Any, Callable, Iterable, Sequence

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from .. import datasets
from ..models.cache_loader import CacheLoader
from ..utils import export, misc
from ..visualization import viz2d
from . import io, utils

logger = logging.getLogger(__name__)


def load_eval(dir):
    summaries, results = {}, {}
    with h5py.File(str(dir / "results.h5"), "r") as hfile:
        for k in hfile.keys():
            r = np.array(hfile[k])
            if len(r.shape) < 3:
                results[k] = r
        for k, v in hfile.attrs.items():
            summaries[k] = v
    with open(dir / "summaries.json", "r") as f:
        s = json.load(f)
    summaries = {k: v if v is not None else np.nan for k, v in s.items()}
    return summaries, results


def save_eval(dir, summaries, figures, results):
    with h5py.File(str(dir / "results.h5"), "w") as hfile:
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                arr = arr.astype("object")
            hfile.create_dataset(k, data=arr)
        # just to be safe, not used in practice
        for k, v in summaries.items():
            hfile.attrs[k] = v
    s = {
        k: float(v) if np.isfinite(v) else None
        for k, v in summaries.items()
        if not isinstance(v, list)
    }
    s = {**s, **{k: v for k, v in summaries.items() if isinstance(v, list)}}
    with open(dir / "summaries.json", "w") as f:
        json.dump(s, f, indent=4)

    for fig_name, fig in figures.items():
        fig.savefig(dir / f"{fig_name}.png")


def exists_eval(dir):
    return (dir / "results.h5").exists() and (dir / "summaries.json").exists()


class EvalPipeline:
    default_conf = {}

    export_keys = []
    optional_export_keys = []

    main_metric = "???"  # You need to define this.

    def __init__(self, conf):
        """Assumes"""
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self._init(self.conf)

    def _init(self, conf):
        pass

    @classmethod
    def get_dataset(self, data_conf=None):
        """Returns a dataset with samples for each eval datapoint"""
        return self.get_dataloader(data_conf).dataset

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        raise NotImplementedError

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        raise NotImplementedError

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        raise NotImplementedError

    def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False):
        """Run export+eval loop"""
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        logger.info(f"Running eval pipeline {self.__class__.__name__}.")
        logger.info(f'Loop 1: Exporting predictions to "{experiment_dir}".')
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite
        )
        logger.info(f"Loop 1 finished. Predictions saved to {pred_file}.")

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            logger.info(f"Loop 2: Evaluating predictions in {pred_file}.")
            s, f, r = self.run_eval(self.get_dataloader(), pred_file)
            save_eval(experiment_dir, s, f, r)
            logger.info(f"Loop 2 finished. Results saved to {experiment_dir}.")
        s, r = load_eval(experiment_dir)
        return s, f, r

    def save_conf(self, experiment_dir, overwrite=False, overwrite_eval=False):
        # store config
        conf_output_path = experiment_dir / "conf.yaml"
        if conf_output_path.exists():
            saved_conf = OmegaConf.load(conf_output_path)
            if (saved_conf.data != self.conf.data) or (
                saved_conf.model != self.conf.model
            ):
                assert (
                    overwrite
                ), "configs changed, add --overwrite to rerun experiment with new conf"
            if saved_conf.eval != self.conf.eval:
                assert (
                    overwrite or overwrite_eval
                ), "eval configs changed, add --overwrite_eval to rerun evaluation"
        OmegaConf.save(self.conf, experiment_dir / "conf.yaml")


# ----------------------------------------------------------------------------
# Common specializations (Relative Pose, Homography, SfM, ...)
# ----------------------------------------------------------------------------


class RelativePosePipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "???",
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": ["poselib", "opencv"],
            "ransac_th": -1.0,  # -1 runs a bunch of thresholds and selects the best
            "n_processes": None,  # 0 is sequential
        },
    }

    main_metric = "rel_pose_error_mAA"

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
    optional_export_keys = []

    # Add custom evals here (dataset specific)
    eval_hooks: Sequence[Callable[[Any, Any], dict[str, float]]] = ()

    def __init__(self, conf):
        """Assumes"""
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(
            RelativePosePipeline.default_conf, self.default_conf, conf
        )
        self._init(self.conf)

    def _init(self, conf):
        raise NotImplementedError("Add download instructions here")

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = datasets.get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

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

    def load_evaluate_relative_pose(self, cache_loader: CacheLoader, data):
        pred = cache_loader(data)
        pose_results = self.evaluate_relative_pose(data, pred)
        pose_results["names"] = data["name"][0]
        return pose_results

    def evaluate_relative_pose(
        self,
        data: dict[str, Any],
        pred: dict[str, Any],
    ):
        conf = self.conf.eval
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results_i = collections.defaultdict(dict)
        estimators = (
            [conf.estimator] if isinstance(conf.estimator, str) else conf.estimator
        )
        for estimator in estimators:
            for th in test_thresholds:
                pose_results_i[estimator][th] = utils.eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": estimator, "ransac_th": th},
                )
        pose_results_i["names"] = data["name"][0]
        return dict(pose_results_i)

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = collections.defaultdict(list)
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        pose_results = []

        if conf.n_processes != 0:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=conf.n_processes)

        results = []
        for i, data in enumerate(tqdm(loader, desc="Evaluation: ")):
            pred = cache_loader(data)
            # add custom evaluations here
            results_i = utils.eval_matches_epipolar(data, pred)
            if "depth" in data["view0"].keys():
                results_i.update(utils.eval_matches_depth(data, pred))

            for eval_hook in self.eval_hooks:
                results_i.update(eval_hook(data, pred))

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            if "overlap" in data.keys():
                results_i["overlap"] = data["overlap"][0].item()

            if conf.n_processes == 0:
                pose_results.append(self.evaluate_relative_pose(data, pred))
            else:
                pose_results.append(
                    pool.apply_async(
                        self.evaluate_relative_pose,
                        args=(data, pred),
                    )
                )
            results.append(results_i)

        results = misc.pack_tree(results)
        if conf.n_processes != 0:
            pose_results = [
                p.get() for p in tqdm(pose_results, desc="Pose Estimation: ")
            ]
            pool.close()
            pool.join()
            pool.terminate()
        pose_results = misc.pack_tree(pose_results, sep=None)
        pose_names = pose_results.pop("names")  # To fix order

        # Fix the order to be consistent (usually a no-op, but better safe)
        if conf.n_processes != 0:
            reorder = [pose_names.index(pname) for pname in results["names"]]
            pose_results = misc.flat_map(
                pose_results,
                lambda k, v: [v[i] for i in reorder],
                sep=None,
                unflatten=True,
            )

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose = {}
        for estimator, pose_results_e in pose_results.items():
            prefix = f"est_{estimator}:"
            best_summary_e, best_th = utils.eval_poses(
                pose_results_e, auc_ths=[5, 10, 20], key="rel_pose_error"
            )
            results = {
                **results,
                **misc.add_prefix(pose_results_e[best_th], prefix),
            }
            best_pose[best_summary_e["rel_pose_error_mAA"]] = (
                best_summary_e,
                pose_results_e[best_th],
            )
            if len(pose_results) > 1:
                summaries = {
                    **summaries,
                    **{
                        f"est_{estimator}:{k}": v
                        for k, v in best_summary_e.items()
                        if k.startswith("rel_pose_error")
                    },
                }

        best_pose_summary, best_pose_results = best_pose[max(best_pose.keys())]
        summaries = {**summaries, **best_pose_summary}
        results = {**results, **best_pose_results}

        figures = {
            "pose_recall": viz2d.plot_cumulative(
                {est: results[f"est_{est}:rel_pose_error"] for est in pose_results},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }
        return summaries, figures, results
