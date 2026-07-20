import logging
import zipfile
from pathlib import Path

import torch

from .. import settings
from . import eval_pipeline, io

logger = logging.getLogger(__name__)


class MegaDepth1500ValPipeline(eval_pipeline.RelativePosePipeline):
    default_conf = {
        "data": {
            "name": "megadepth",
            "preprocessing": {
                "side": "long",
                "resize": 768,
            },
            "num_workers": None,
            # "test_pairs": "valid_pairs.txt",
            "test_num_per_scene": 3,
            "min_overlap": 0.1,
            "max_overlap": 0.7,
            "test_split": "debug_scenes_clean.txt",
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": eval_pipeline.RelativePosePipeline.default_conf["eval"],
    }

    default_x: str = "gt_match_precision@3px"
    default_y: str = "gt_match_recall@3px"

    def _init(self, conf):
        if not (settings.DATA_PATH / "megadepth").exists():
            logger.info("Please download the MegaDepth dataset.")


if __name__ == "__main__":
    io.run_cli(MegaDepth1500ValPipeline, name=Path(__file__).stem)
