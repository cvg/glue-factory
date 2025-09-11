import unittest
from collections import namedtuple
from os.path import splitext

import cv2
import hydra
import matplotlib.pyplot as plt
import torch.cuda
from kornia import image_to_tensor
from parameterized import parameterized
from torch import Tensor

from gluefactory import logger, settings
from gluefactory.eval.utils import (
    eval_homography_dlt,
    eval_homography_robust,
    eval_matches_homography,
)
from gluefactory.models.two_view_pipeline import TwoViewPipeline
from gluefactory.utils import misc
from gluefactory.utils.preprocess import ImagePreprocessor
from gluefactory.utils.tools import set_seed
from gluefactory.visualization import viz2d


def create_input_data(cv_img0, cv_img1, device):
    img0 = image_to_tensor(cv_img0).float() / 255
    img1 = image_to_tensor(cv_img1).float() / 255
    ip = ImagePreprocessor({})
    data = {"view0": ip(img0), "view1": ip(img1)}
    data = misc.map_tensor(
        data,
        lambda t: (
            t[None].to(device)
            if isinstance(t, Tensor)
            else torch.from_numpy(t)[None].to(device)
        ),
    )
    return data


ExpectedResults = namedtuple("ExpectedResults", ("num_matches", "prec3px", "h_error"))


class TestIntegration(unittest.TestCase):
    methods_to_test = [
        ("superpoint+NN.yaml", "poselib", ExpectedResults(1300, 0.8, 1.0)),
        ("superpoint-open+NN.yaml", "poselib", ExpectedResults(1300, 0.8, 1.0)),
        (
            "superpoint+lsd+gluestick.yaml",
            "homography_est",
            ExpectedResults(1300, 0.8, 1.0),
        ),
        (
            "superpoint+lightglue-official.yaml",
            "poselib",
            ExpectedResults(1300, 0.8, 1.0),
        ),
    ]

    visualize = False

    @parameterized.expand(methods_to_test)
    @torch.no_grad()
    def test_real_homography(self, conf_file, estimator, exp_results):
        set_seed(0)
        img_path0 = settings.root / "assets" / "boat1.png"
        img_path1 = settings.root / "assets" / "boat2.png"
        h_gt = torch.tensor(
            [
                [0.85799, 0.21669, 9.4839],
                [-0.21177, 0.85855, 130.48],
                [1.5015e-06, 9.2033e-07, 1],
            ]
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with hydra.initialize(config_path="../gluefactory/configs", version_base=None):
            conf = hydra.compose(config_name=conf_file)
        gs = TwoViewPipeline(conf.model).to(device).eval()

        cv_img0, cv_img1 = cv2.imread(str(img_path0)), cv2.imread(str(img_path1))
        data = create_input_data(cv_img0, cv_img1, device)
        pred = gs(data)
        pred = misc.map_tensor(
            pred, lambda t: torch.squeeze(t, dim=0) if isinstance(t, Tensor) else t
        )
        data["H_0to1"] = h_gt.to(device)
        data["H_1to0"] = torch.linalg.inv(h_gt).to(device)

        results = eval_matches_homography(data, pred)
        results = {**results, **eval_homography_dlt(data, pred)}

        results = {
            **results,
            **eval_homography_robust(
                data,
                pred,
                {"estimator": estimator},
            ),
        }

        logger.info(results)
        self.assertGreater(results["num_matches"], exp_results.num_matches)
        self.assertGreater(results["prec@3px"], exp_results.prec3px)
        self.assertLess(results["H_error_ransac"], exp_results.h_error)

        if self.visualize:
            pred = misc.map_tensor(
                pred, lambda t: t.cpu().numpy() if isinstance(t, Tensor) else t
            )
            kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
            m0 = pred["matches0"]
            valid0 = m0 != -1
            kpm0, kpm1 = kp0[valid0], kp1[m0[valid0]]

            viz2d.plot_images([cv_img0, cv_img1])
            viz2d.plot_matches(kpm0, kpm1, a=0.0)
            plt.savefig(f"{splitext(conf_file)[0]}_point_matches.svg")

            if "lines0" in pred and "lines1" in pred:
                lines0, lines1 = pred["lines0"], pred["lines1"]
                lm0 = pred["line_matches0"]
                lvalid0 = lm0 != -1
                linem0, linem1 = lines0[lvalid0], lines1[lm0[lvalid0]]

                viz2d.plot_images([cv_img0, cv_img1])
                viz2d.plot_color_line_matches([linem0, linem1])
                plt.savefig(f"{splitext(conf_file)[0]}_line_matches.svg")
            plt.show()
