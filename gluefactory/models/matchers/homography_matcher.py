from ...geometry.gt_generation import (
    gt_line_matches_from_homography,
    gt_matches_from_homography,
)
from ..base_model import BaseModel


class HomographyMatcher(BaseModel):
    default_conf = {
        # GT parameters for points
        "use_points": True,
        "th_positive": 3.0,
        "th_negative": 3.0,
        # GT parameters for lines
        "use_lines": False,
        "n_line_sampled_pts": 50,
        "line_perp_dist_th": 5,
        "overlap_th": 0.2,
        "min_visibility_th": 0.5,
    }

    required_data_keys = ["H_0to1"]

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

    def _forward(self, data):
        result = {}
        if self.conf.use_points:
            result = gt_matches_from_homography(
                data["keypoints0"],
                data["keypoints1"],
                data["H_0to1"],
                pos_th=self.conf.th_positive,
                neg_th=self.conf.th_negative,
            )
        if self.conf.use_lines:
            line_assignment, line_m0, line_m1 = gt_line_matches_from_homography(
                data["lines0"],
                data["lines1"],
                data["valid_lines0"],
                data["valid_lines1"],
                data["view0"]["image"].shape,
                data["view1"]["image"].shape,
                data["H_0to1"],
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
