import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from omegaconf import OmegaConf

from .. import models, pipelines, settings
from ..datasets import base_dataset
from ..geometry import reconstruction as reconstruction
from ..models import cache_loader
from ..pipelines.reconstruction.base import ReconstructionPipeline
from ..utils import export, misc, types
from . import eval_pipeline, io

ETH3D_SCENES = {
    "outdoor": ["courtyard", "electro", "facade", "meadow", "playground", "terrace"],
    "indoor": [
        "delivery_area",
        "kicker",
        "office",
        "pipes",
        "relief",
        "relief_2",
        "terrains",
    ],
    "test": [
        "botanical_garden",
        "boulders",
        "bridge",
        "door",
        "exhibition_hall",
        "lecture_room",
        "living_room",
        "lounge",
        "observatory",
        "old_computer",
        "statue",
        "terrace_2",
    ],
}


class ETH3DReconstructionPipeline(eval_pipeline.EvalPipeline):
    scenes = sum(ETH3D_SCENES.values(), [])

    default_conf = {
        "data": {
            "root": "ETH3D_undistorted_resized",
            "num_workers": 4,
        },
        "model": {
            "allow_no_extract": True,
        },
        "eval": {
            "thresholds": [1, 3, 5, 10, 20],  # degrees
            "write_scene_summaries": False,
        },
        "pipeline": {
            "name": "reconstruction.hloc",
        },
    }

    root_dir = settings.DATA_PATH / default_conf["data"]["root"]

    def _init(self, conf) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory {self.root_dir} does not exist.")

        pipeline_conf = OmegaConf.merge(
            self.default_conf.pipeline,
            {"data": self.conf.data},
        )
        self.pipeline: ReconstructionPipeline = pipelines.get_pipeline(
            ReconstructionPipeline, conf.pipeline.name
        )(pipeline_conf)

    @classmethod
    def get_dataset(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        scene_dataset = {}
        for scene in self.scenes:
            ref_sfm_dir = self.root_dir / scene / "dslr_calibration_undistorted"
            ref_sfm = pycolmap.Reconstruction(ref_sfm_dir)
            scene_dataset[scene] = types.ReconstructionData(
                image_list=[image.name for image in ref_sfm.images.values()],
                image_dir=self.root_dir / scene / "images",
                reference_sfm=ref_sfm_dir,
            )

        scene_dataset = [
            {"name": scene, "reconstruction": data, "scene": scene}
            for scene, data in scene_dataset.items()
        ]
        return scene_dataset

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        return iter(self.get_dataset(data_conf))

    def run_scene(
        self,
        scene: str,
        experiment_dir: Path,
        model: models.BaseModel,
        data: types.ReconstructionData,
    ) -> tuple[reconstruction.Reconstruction, dict]:
        output_dir = experiment_dir / scene
        output_dir.mkdir(exist_ok=True, parents=True)

        # Export priors
        self.pipeline.export_priors(
            output_dir,
            model,
            data,
        )

        # Run reconstruction
        rec, preds = self.pipeline.run_reconstruction(
            output_dir,
            model,
            data,
        )

        preds["output_dir"] = str(output_dir)
        return rec, preds

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        if model is None:
            model = io.load_model(self.conf.model, self.conf.checkpoint)

        experiment_dir.mkdir(exist_ok=True, parents=True)

        if overwrite and experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        elif experiment_dir.exists():
            return experiment_dir

        all_preds = {}
        predictions_file = experiment_dir / "predictions.h5"

        loader = self.get_dataloader()
        for data in loader:
            scene = data["name"]
            rec, pred = self.run_scene(
                scene, experiment_dir, model, data["reconstruction"]
            )
            all_preds[scene] = pred
        export.dict_to_h5(predictions_file, all_preds)
        return predictions_file

    def eval_scene(
        self,
        scene: str,
        experiment_dir: Path,
        pred: dict,
        data: types.ReconstructionData,
    ):

        # add custom evaluations here
        results = defaultdict(dict)
        output_dir = experiment_dir / scene
        gt_model = reconstruction.Reconstruction.from_colmap(data.reference_sfm)
        estimated_colmap_model = pycolmap.Reconstruction(output_dir)
        results["track_length"] = estimated_colmap_model.compute_mean_track_length()
        results["reprojection_error"] = (
            estimated_colmap_model.compute_mean_reprojection_error()
        )
        results["observations_per_reg_image"] = (
            estimated_colmap_model.compute_mean_observations_per_reg_image()
        )
        estimated_model = reconstruction.Reconstruction.from_colmap(
            estimated_colmap_model
        )
        thresholds = self.conf.eval.thresholds
        _, auc = estimated_model.compare_poses_to(
            gt_model,
            thresholds=thresholds,
        )

        for i, th in enumerate(thresholds):
            results[f"AUC@{th}"] = auc[i]

        # scene avg results, e.g. from multiview tool
        results["num_images"] = len(gt_model.image_names)
        results["num_reg_images"] = estimated_model.num_reg_images
        return results

    def run_eval(self, loader, pred_file):
        results = {}
        cache = cache_loader.CacheLoader(
            {"path": str(pred_file), "collate": None}
        ).eval()
        for data in loader:
            pred = cache(base_dataset.collate([data]))
            scene = data["name"]
            results[scene] = self.eval_scene(
                scene,
                pred_file.parent,
                pred,
                data["reconstruction"],
            )

        summaries = {}

        if self.conf.eval.write_scene_summaries:
            # write scene summaries
            for scene, result in results.items():
                summaries[scene] = {k: v for k, v in result.items()}

        groups = {
            "all": self.scenes,
            **ETH3D_SCENES,
        }
        for group_name, group_scenes in groups.items():
            group_summary = defaultdict(list)
            for scene in group_scenes:
                if scene in results.keys():
                    [group_summary[k].append(v) for k, v in results[scene].items()]
            summaries[group_name] = {
                k: np.mean(v).round(3) for k, v in group_summary.items()
            }

        summaries = misc.flatten_dict(summaries)

        out_results = {
            metric: [results[scene][metric] for scene in self.scenes]
            for metric in results[next(iter(results))].keys()
        }
        out_results["names"] = self.scenes
        out_results["scenes"] = [
            ",".join([k for k, v in ETH3D_SCENES.items() if scene in v])
            for scene in self.scenes
        ]
        return summaries, {}, out_results


if __name__ == "__main__":
    parser = io.get_eval_parser()
    parser.add_argument(
        "--scenes", nargs="+", type=str, default=ETH3DReconstructionPipeline.scenes
    )
    io.run_cli(ETH3DReconstructionPipeline, Path(__file__).stem, parser)
