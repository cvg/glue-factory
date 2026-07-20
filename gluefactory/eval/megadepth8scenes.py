"""Evaluation on the Megadepth-8-Scenes split.
Proposed by Edstedt et al, CVPR 2023 (DKM)."""

import logging
import zipfile
from pathlib import Path

import torch

from gluefactory import settings

from . import io
from .eval_pipeline import RelativePosePipeline

logger = logging.getLogger(__name__)


class Megadepth8ScenesPipeline(RelativePosePipeline):
    default_conf = {
        "data": {
            "name": "posed_images",
            "root": "",
            "image_dir": "{scene}/images",
            "depth_dir": "{scene}/depths",
            "views": "{scene}/views.txt",
            "view_groups": "{scene}/pairs.txt",
            "depth_format": "h5",
            "scene_list": ["megadepth8scenes"],
            "preprocessing": {"side": "long"},
            "num_workers": None,
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": RelativePosePipeline.default_conf["eval"],
    }

    def _init(self, conf):
        if not (settings.DATA_PATH / "megadepth8scenes").exists():
            logger.info("Downloading the MegaDepth-8-Scenes dataset.")
            url = "https://cvg-data.inf.ethz.ch/megadepth/megadepth8scenes.zip"
            zip_path = settings.DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(settings.DATA_PATH)
            zip_path.unlink()


if __name__ == "__main__":
    io.run_cli(Megadepth8ScenesPipeline, name=Path(__file__).stem)
