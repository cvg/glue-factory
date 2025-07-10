import logging
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from ..settings import DATA_PATH, EVAL_PATH
from .io import get_eval_parser, parse_eval_args
from .scannet1500 import ScanNet1500Pipeline

logger = logging.getLogger(__name__)


class ZeroshotEvaluationBenchmarkPipeline(ScanNet1500Pipeline):
    default_conf = {
        "data": {
            "name": "zeb",
            "scene_list": None,
            "root": "zeb",
            "shuffle": False,
            "max_per_scene": 200,  # maximum number of pairs per scene
            "min_overlap": 0.0,  # minimum overlap for pairs
            "max_overlap": 1.0,  # maximum overlap for pairs
            "preprocessing": {
                "side": "long",
                "resize": 1024,  # resize to 1024px on the long side
            },
            "num_workers": 14,
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "opencv",
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
    optional_export_keys = []

    def _init(self, conf):
        if not (DATA_PATH / "zeb").exists():
            logger.info("Please manually download the ZEB dataset following GIM:")
            logger.info("%s", "https://github.com/xuelunshen/gim/tree/main")
            logger.info("Target format: data/zeb/<scene>/*")


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(ZeroshotEvaluationBenchmarkPipeline.default_conf)

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

    pipeline = ZeroshotEvaluationBenchmarkPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
