"""
Zeroshot Evaluation Benchmark (ZEB) Pipeline.

Source: https://github.com/xuelunshen/gim
Paper: GIM: Learning Generalizable Image Matcher From Internet Videos
Arxiv: https://arxiv.org/abs/2402.11095
"""

import logging
from pathlib import Path

from ..settings import DATA_PATH
from . import io
from .eval_pipeline import RelativePosePipeline

logger = logging.getLogger(__name__)


class ZeroshotEvaluationBenchmarkPipeline(RelativePosePipeline):
    """Pipeline for Zeroshot Evaluation Benchmark (ZEB)."""

    default_conf = {
        "data": {
            "name": "zeb",
            "scene_list": None,
            "root": "zeb",
            "shuffle": False,
            "exclude_scenes": ["eth3di", "eth3do"],
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
        "eval": RelativePosePipeline.default_conf["eval"],
    }

    def _init(self, conf):  # pylint: disable=redefined-outer-name
        if not (DATA_PATH / "zeb").exists():
            logger.info("Please manually download the ZEB dataset following GIM:")
            logger.info("%s", "https://github.com/xuelunshen/gim/tree/main")
            logger.info("Target format: data/zeb/<scene>/*")


if __name__ == "__main__":
    io.run_cli(ZeroshotEvaluationBenchmarkPipeline, name=Path(__file__).stem)
