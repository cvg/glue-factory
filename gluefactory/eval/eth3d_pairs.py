import logging
from pathlib import Path

from ..settings import DATA_PATH
from . import io
from .eval_pipeline import RelativePosePipeline

logger = logging.getLogger(__name__)


class ETH3DPairsPipeline(RelativePosePipeline):
    default_conf = {
        "data": {
            "name": "pairs_from_colmap",
            "root": "ETH3D_undistorted_resizedx2",
            "image_dir": "{scene}/images",
            "depth_dir": "{scene}/ground_truth_depth_dense",
            "depth_format": "h5",
            "scene_list": None,
            "sfm": "{scene}/dslr_calibration_undistorted/",
            "views": "{scene}/views.txt",  # To cache poses & cameras
            "overwrite": False,
            "min_overlap": 0.1,
            "max_overlap": 0.7,
            "max_per_scene": 10,
            "preprocessing": {
                "side": "long",
            },
            "num_workers": 14,
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
    }

    def _init(self, conf):
        if not (DATA_PATH / conf.data.root).exists():
            raise FileNotFoundError(
                f"Data root {DATA_PATH / conf.data.root} does not exist. "
                "Please download the ETH3D dataset and set the correct path."
            )


if __name__ == "__main__":
    io.run_cli(ETH3DPairsPipeline, Path(__file__).stem)
