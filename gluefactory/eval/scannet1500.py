import logging
import zipfile
from pathlib import Path

import torch

from .. import settings
from . import io
from .eval_pipeline import RelativePosePipeline

logger = logging.getLogger(__name__)


class ScanNet1500Pipeline(RelativePosePipeline):
    default_conf = {
        "data": {
            "name": "image_pairs",
            "pairs": "scannet1500/pairs_calibrated.txt",
            "root": "scannet1500/",
            "extra_data": "relative_pose",
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
        "eval": RelativePosePipeline.default_conf["eval"],
    }

    def _init(self, conf):
        if not (settings.DATA_PATH / "scannet1500").exists():
            logger.info("Downloading the ScanNet-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/scannet/scannet1500.zip"
            zip_path = settings.DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(settings.DATA_PATH)
            zip_path.unlink()


if __name__ == "__main__":
    io.run_cli(ScanNet1500Pipeline, name=Path(__file__).stem)
