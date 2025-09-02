import logging
import zipfile
from pathlib import Path

import torch

from .. import settings
from . import eval_pipeline, io

logger = logging.getLogger(__name__)


class MegaDepth1500Pipeline(eval_pipeline.RelativePosePipeline):
    default_conf = {
        "data": {
            "name": "posed_images",
            "root": "",
            "image_dir": "{scene}/images",
            "depth_dir": "{scene}/depths",
            "views": "{scene}/views.txt",
            "view_groups": "{scene}/pairs.txt",
            "depth_format": "h5",
            "scene_list": ["megadepth1500"],
            "preprocessing": {
                "side": "long",
            },
            "num_workers": 4,
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": eval_pipeline.RelativePosePipeline.default_conf["eval"],
    }

    def _init(self, conf):
        if not (settings.DATA_PATH / "megadepth1500").exists():
            logger.info("Downloading the MegaDepth-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"
            zip_path = settings.DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(settings.DATA_PATH)
            zip_path.unlink()


if __name__ == "__main__":
    io.run_cli(MegaDepth1500Pipeline, name=Path(__file__).stem)
