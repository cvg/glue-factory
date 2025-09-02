"""Air-Ground Evaluation proposed in RDD.  # noqa: E501

Source: https://github.com/xtcpete/rdd
Paper: RDD: Robust Feature Detector and Descriptor using Deformable Transformer
Arxiv: https://arxiv.org/abs/2505.08013

License: Apache 2.0 (https://github.com/xtcpete/rdd/blob/main/LICENSE)

Bibtex:
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Gonglin and Fu, Tianwen and Chen, Haiwei and Teng, Wenbin and Xiao, Hanyuan and Zhao, Yajie},
    title     = {RDD: Robust Feature Detector and Descriptor using Deformable Transformer},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6394-6403}
}
"""

import logging
from pathlib import Path

from ..settings import DATA_PATH
from . import io
from .eval_pipeline import RelativePosePipeline

logger = logging.getLogger(__name__)


class RDDAirGroundBenchmark(RelativePosePipeline):
    """Pipeline for Air-Ground Benchmark (proposed in RDD)."""

    default_conf = {
        "data": {
            "name": "image_pairs_npz",
            "root": "air_ground",
            "images": "images",
            "pairs": "indices.npz",
            "preprocessing": {
                "side": "long",  # RDD resizes to 1600 for inference, NOT for pose est.
                "interpolation": "area",
                "antialias": True,
            },
            "num_workers": 8,
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": RelativePosePipeline.default_conf["eval"],
    }

    def _init(self, conf):  # pylint: disable=redefined-outer-name
        if not (DATA_PATH / "air_ground").exists():
            logger.info("Please manually download the air_ground dataset from RDD:")
            logger.info(
                "https://drive.google.com/drive/folders/1byc8JAxatpOyjO3CGa0r5Q00-RC-1dzR",  # noqa: E501
            )
            logger.info("Target format: data/air_ground/*")


if __name__ == "__main__":
    io.run_cli(RDDAirGroundBenchmark, name=Path(__file__).stem)
