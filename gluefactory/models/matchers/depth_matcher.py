import torch

from ...geometry.gt_generation import (
    gt_line_matches_from_pose_depth,
    gt_matches_from_pose_depth,
)
from ..base_model import BaseModel

# Hacky workaround for torch.amp.custom_fwd to support older versions of PyTorch.
AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


class DepthMatcher(BaseModel):
    default_conf = {
        # GT parameters for points
        "use_points": True,
        "th_positive": 3.0,
        "th_negative": 5.0,
        "th_epi": None,  # add some more epi outliers
        "th_consistency": None,  # check for projection consistency in px
        # GT parameters for lines
        "use_lines": False,
        "n_line_sampled_pts": 50,
        "line_perp_dist_th": 5,
        "overlap_th": 0.2,
        "min_visibility_th": 0.5,
    }

    required_data_keys = ["view0", "view1", "T_0to1"]

    def _init(self, conf):
        # TODO (iago): Is this just boilerplate code?
        if self.conf.use_points:
            self.required_data_keys += ["keypoints0", "keypoints1"]
        if self.conf.use_lines:
            self.required_data_keys += [
                "lines0",
                "lines1",
                "valid_lines0",
                "valid_lines1",
            ]

    @AMP_CUSTOM_FWD_F32
    def _forward(self, data):
        result = {}
        if self.conf.use_points:
            if "depth_keypoints0" in data:
                keys = [
                    "depth_keypoints0",
                    "valid_depth_keypoints0",
                    "depth_keypoints1",
                    "valid_depth_keypoints1",
                ]
                kw = {k: data[k] for k in keys}
            else:
                kw = {}
            result = gt_matches_from_pose_depth(
                data["keypoints0"],
                data["keypoints1"],
                data,
                pos_th=self.conf.th_positive,
                neg_th=self.conf.th_negative,
                epi_th=self.conf.th_epi,
                cc_th=self.conf.th_consistency,
                **kw,
            )
        if self.conf.use_lines:
            line_assignment, line_m0, line_m1 = gt_line_matches_from_pose_depth(
                data["lines0"],
                data["lines1"],
                data["valid_lines0"],
                data["valid_lines1"],
                data,
                self.conf.n_line_sampled_pts,
                self.conf.line_perp_dist_th,
                self.conf.overlap_th,
                self.conf.min_visibility_th,
            )
            result["line_matches0"] = line_m0
            result["line_matches1"] = line_m1
            result["line_assignment"] = line_assignment
        return result

    def loss(self, pred, data):
        raise NotImplementedError
