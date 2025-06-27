import numpy as np
import torch

from ...settings import DATA_PATH
from ..base_model import BaseModel

import gluefactory.models.scalelsd_inference as scalelsd_inference
class ScaleLSD(BaseModel):
    required_data_keys = ["image"]

    def load_scalelsd_model(self, ckpt_path, device='cuda'):
        """load model"""
        use_layer_scale = False if 'v1' in str(ckpt_path) else True

        model = scalelsd_inference.ScaleLSD(self.conf, gray_scale=True, use_layer_scale=use_layer_scale)
        model = model.eval().to(device)

        model.junction_threshold_hm = self.junction_threshold_hm
        model.num_junctions_inference = self.num_junctions_inference

        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        try:
            model.load_state_dict(state_dict['model_state'])
        except:
            model.load_state_dict(state_dict)

        return model

    def _init(self, conf):
        self.model_name = 'scalelsd-vitbase-v2-train-sa1b.pt'
        # TODO: Make it parametrizable with the base configuration
        self.threshold = 10
        self.junction_threshold_hm = 0.008
        self.num_junctions_inference = 512
        self.width = 512
        self.height = 512
        self.line_width = 2
        self.juncs_size = 4
        self.whitebg = 0.0
        self.draw_junctions_only = False
        self.use_lsd = False
        self.use_nms = False

        if self.conf.force_num_lines:
            assert (
                self.conf.max_num_lines is not None
            ), "Missing max_num_lines parameter"
        ckpt = DATA_PATH / "weights" / self.model_name
        if not ckpt.is_file():
            self.download_model(ckpt)


        # TODO: Replace this part
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.load_scalelsd_model(ckpt, device)


    def download_model(self, path):
        import subprocess

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        link = "https://huggingface.co/cherubicxn/scalelsd/resolve/main/scalelsd-vitbase-v2-train-sa1b.pt?download=true"
        cmd = ["wget", link, "-O", path]
        print("Downloading ScaleLSD model...")
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
            segs = self.net(image)

        # Line scores are the sqrt of the length
        for seg in segs:
            line_pred = seg['lines_pred'].reshape(-1, 2,2)
            scores = seg['lines_score'].reshape(-1)
            line_pred = line_pred[scores >= self.threshold]
            scores = scores[scores >= self.threshold]

            # Pad if necessary
            n = len(line_pred)
            valid_mask = np.ones(n, dtype=bool)
            if self.conf.force_num_lines:
                pad = self.conf.max_num_lines - n
                segs = np.concatenate(
                    [line_pred, np.zeros((pad, 2, 2), dtype=np.float32)], axis=0
                )
                scores = np.concatenate(
                    [scores, np.zeros(pad, dtype=np.float32)], axis=0
                )
                valid_mask = np.concatenate(
                    [valid_mask, np.zeros(pad, dtype=bool)], axis=0
                )

            lines.append(line_pred)
            line_scores.append(scores)
            valid_lines.append(valid_mask)

        # Batch if possible
        if len(image) == 1 or self.conf.force_num_lines:
            lines = torch.from_numpy(np.stack(lines, axis=0).astype(np.float32)).to(image.device).float()
            line_scores = torch.from_numpy(np.stack(line_scores, axis=0).astype(np.float32)).to(image.device).float()
            valid_lines = torch.from_numpy(np.stack(valid_lines, axis=0).astype(np.uint8)).to(image.device).bool()

        return {"lines": lines, "line_scores": line_scores, "valid_lines": valid_lines}

    def loss(self, pred, data):
        raise NotImplementedError
