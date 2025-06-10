import numpy as np
import torch

import gluefactory.models.deeplsd_inference as deeplsd_inference

from ...settings import DATA_PATH
from ..base_model import BaseModel


class DeepLSD(BaseModel):
    default_conf = {
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
        "model_conf": {
            "detect_lines": False,
            "line_detection_params": {
                "use_img_grad_angle": False,
                "merge": False,
                "grad_nfa": True,
                "filtering": "normal",
                "grad_thresh": 3,
                "faster_lsd": False,
            },
        },
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert (
                self.conf.max_num_lines is not None
            ), "Missing max_num_lines parameter"
        ckpt = DATA_PATH / "weights/deeplsd_md.tar"
        if not ckpt.is_file():
            self.download_model(ckpt)
        ckpt = torch.load(ckpt, map_location="cpu")
        self.net = deeplsd_inference.DeepLSD(conf.model_conf).eval()
        self.net.load_state_dict(ckpt["model"])
        self.set_initialized()

    def download_model(self, path):
        import subprocess

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        link = "https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar"
        cmd = ["wget", link, "-O", path]
        print("Downloading DeepLSD model...")
        subprocess.run(cmd, check=True)

    def _forward(self, data):
        image = data["image"]
        lines, line_scores, valid_lines = [], [], []
        if image.shape[1] == 3:
            # Convert to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        # Forward pass
        with torch.no_grad():
            segs = self.net({"image": image})["lines"]

        # Line scores are the sqrt of the length
        for seg in segs:
            lengths = np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1)
            segs = seg[lengths >= self.conf.min_length]
            scores = np.sqrt(lengths[lengths >= self.conf.min_length])

            # Keep the best lines
            indices = np.argsort(-scores)
            if self.conf.max_num_lines is not None:
                indices = indices[: self.conf.max_num_lines]
                segs = segs[indices]
                scores = scores[indices]

            # Pad if necessary
            n = len(segs)
            valid_mask = np.ones(n, dtype=bool)
            if self.conf.force_num_lines:
                pad = self.conf.max_num_lines - n
                segs = np.concatenate(
                    [segs, np.zeros((pad, 2, 2), dtype=np.float32)], axis=0
                )
                scores = np.concatenate(
                    [scores, np.zeros(pad, dtype=np.float32)], axis=0
                )
                valid_mask = np.concatenate(
                    [valid_mask, np.zeros(pad, dtype=bool)], axis=0
                )

            lines.append(segs)
            line_scores.append(scores)
            valid_lines.append(valid_mask)

        # Batch if possible
        if len(image) == 1 or self.conf.force_num_lines:
            lines = torch.from_numpy(np.stack(lines, axis=0)).to(image.device).float()
            line_scores = torch.from_numpy(np.stack(line_scores, axis=0)).to(image.device).float()
            valid_lines = torch.from_numpy(np.stack(valid_lines, axis=0)).to(image.device).bool()

        return {"lines": lines, "line_scores": line_scores, "valid_lines": valid_lines}

    def forward_ha(self, data):
        image = data["image"]
        if image.shape[1] == 3:
            # Convert to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        # Forward pass
        with torch.no_grad():
            outputs = self.net({"image": image})
        return outputs

    def line_detection_single_image(
        self,
        image: torch.Tensor,
        angle_field: torch.Tensor,
        distance_field: torch.Tensor,
    ) -> dict:
        """
        Standalone line detection based on angle and distance field for use on HA produced af/df.
        Only works for a single image!!
        """
        print("Line Detection...")
        np_img = (image.cpu().numpy()[:, 0] * 255).astype(np.uint8).squeeze(0)
        np_df = distance_field.cpu().numpy()
        np_ll = angle_field.cpu().numpy()
        lines = self.net.detect_afm_lines(
            np_img, np_df, np_ll, **self.net.conf.line_detection_params
        )
        # Filter detected lines
        lengths = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
        segs = lines[lengths >= self.conf.min_length]
        scores = np.sqrt(lengths[lengths >= self.conf.min_length])

        # Keep the best lines
        indices = np.argsort(-scores)
        if self.conf.max_num_lines is not None:
            indices = indices[: self.conf.max_num_lines]
            segs = segs[indices]
            scores = scores[indices]

        # Pad if necessary
        n = len(segs)
        valid_mask = np.ones(n, dtype=bool)
        if self.conf.force_num_lines:
            pad = self.conf.max_num_lines - n
            segs = np.concatenate(
                [segs, np.zeros((pad, 2, 2), dtype=np.float32)], axis=0
            )
            scores = np.concatenate([scores, np.zeros(pad, dtype=np.float32)], axis=0)
            valid_mask = np.concatenate([valid_mask, np.zeros(pad, dtype=bool)], axis=0)

        return {"lines": segs, "line_scores": scores, "valid_lines": valid_mask}

    def loss(self, pred, data):
        raise NotImplementedError
