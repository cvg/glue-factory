"""
Wrapper for the IMCUI matcher API.
This allows running and evaluating  many different image matching models
from the IMCUI matcher zoo (or custom):
https://github.com/Vincentqyw/image-matching-webui/blob/main/config/config.yaml
"""

import logging
import pprint
import urllib.request
from pathlib import Path

import numpy as np
import torch
from imcui.api import ImageMatchingAPI
from imcui.ui.utils import get_matcher_zoo, load_config
from omegaconf import OmegaConf

from gluefactory import settings
from gluefactory.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class IMCUIMatcher(BaseModel):
    default_conf = {
        "zoo_name": "none",  # Name of the matcher to use from the IMCUI matcher zoo.
        "custom": {},  # Custom configuration (overwrites zoo config).
        "config_path": None,  # Path to the config file where the matcher zoo is stored.
        "overwrite_download": False,  # If True, always download the config file.
        "device": "cuda",  # Use 'cuda' for GPU or 'cpu' for CPU inference.
        "detection_threshold": 0.015,  # Detection threshold for keypoints.
        "max_num_keypoints": 100000,  # Maximum number of keypoints to detect.
        "match_threshold": 0.0,  # Matching threshold. By default accept all.
        "max_num_matches": None,  # Maximum number of matches to return.
    }
    required_data_keys = ["view0", "view1"]

    def resolve_imcui_config_path(self, config_path: str | None) -> Path:
        if config_path is not None:
            return config_path
        # First try if it exists locally, i.e. the IMCUI is installed from source.
        from imcui.ui import app_class

        local_config_path = Path(app_class.__file__).parent / "config.yaml"
        if local_config_path.exists():
            return local_config_path
        else:
            # Otherwise, download file from github.
            target_path = settings.DATA_PATH / "imcui_config.yaml"
            if not target_path.exists() or self.conf.overwrite_download:
                url = "https://raw.githubusercontent.com/Vincentqyw/image-matching-webui/refs/heads/main/config/config.yaml"  # noqa: E501
                logger.info("Downloading IMCUI config from GitHub.")
                urllib.request.urlretrieve(url, target_path)
                logger.info(f"Downloaded IMCUI config to {target_path}.")
            return target_path

    def _init(self, conf):
        if conf.zoo_name is not None:
            imc_config = load_config(self.resolve_imcui_config_path(conf.config_path))
            matcher_zoo = get_matcher_zoo(imc_config["matcher_zoo"])
            if conf.zoo_name not in matcher_zoo:
                raise ValueError(
                    f"Matcher {conf.zoo_name} not found in the IMCUI matcher zoo."
                    f" Available matchers: {list(matcher_zoo.keys())}"
                )
            self.model_conf = matcher_zoo[conf.zoo_name]
        else:
            self.model_conf = {}
        self.model_conf = OmegaConf.to_container(
            OmegaConf.merge(
                OmegaConf.create(self.model_conf),
                OmegaConf.create(conf.custom),
            ),
            resolve=True,
        )
        logger.info("IMCUI Matcher configuration:")
        pprint.pprint(self.model_conf)
        self._api = ImageMatchingAPI(
            conf={
                **self.model_conf,
                **{"ransac": {"enable": False}},  # Never run RANSAC in matcher.
            },
            device=self.conf.device,  # Use CPU for inference
            detect_threshold=conf.detection_threshold,
            max_keypoints=conf.max_num_keypoints,
            match_threshold=conf.match_threshold,
        )
        self.set_initialized()

    def _forward(self, data):
        img0, img1 = data["view0"]["image"], data["view1"]["image"]
        assert img0.shape[0] == 1 and img0.ndim == 4
        imcui_pred = self._api(
            img0[0].cpu().permute(1, 2, 0).numpy() * 255,  # Convert to HWC format
            img1[0].cpu().permute(1, 2, 0).numpy() * 255,
        )

        num_matches = imcui_pred["mkeypoints0"].shape[0]
        if self.conf.max_num_matches is not None:
            idxs = np.argpartition(imcui_pred["mconf"], -self.conf.max_num_matches)[
                -self.conf.max_num_matches :
            ]
            num_matches = idxs.shape[0]
            imcui_pred = {
                **imcui_pred,
                "mkeypoints0_orig": imcui_pred["mkeypoints0_orig"][idxs],
                "mkeypoints1_orig": imcui_pred["mkeypoints1_orig"][idxs],
                "mconf": imcui_pred["mconf"][idxs],
            }
        np_pred = {
            "keypoints0": imcui_pred["mkeypoints0_orig"],
            "keypoints1": imcui_pred["mkeypoints1_orig"],
            "keypoint_scores0": imcui_pred["mconf"],
            "keypoint_scores1": imcui_pred["mconf"],
            "matches0": np.arange(0, num_matches),
            "matches1": np.arange(0, num_matches),
            "matching_scores0": imcui_pred["mconf"],
            "matching_scores1": imcui_pred["mconf"],
        }
        return {
            k: torch.tensor(v, device=img0.device)[None] for k, v in np_pred.items()
        }

    def loss(self, pred, data):
        raise NotImplementedError("IMCUIMatcher does not implement loss computation.")
