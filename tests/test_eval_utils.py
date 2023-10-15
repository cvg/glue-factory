import unittest

import torch

from gluefactory.eval.utils import eval_matches_homography
from gluefactory.geometry.homography import warp_points_torch


class TestEvalUtils(unittest.TestCase):
    @staticmethod
    def default_pts():
        return torch.tensor(
            [
                [10.0, 10.0],
                [10.0, 20.0],
                [20.0, 20.0],
                [20.0, 10.0],
            ]
        )

    @staticmethod
    def default_pred(kps0, kps1):
        return {
            "keypoints0": kps0,
            "keypoints1": kps1,
            "matches0": torch.arange(len(kps0)),
            "matching_scores0": torch.ones(len(kps1)),
        }

    def test_eval_matches_homography_trivial(self):
        data = {"H_0to1": torch.eye(3)}
        kps = self.default_pts()
        pred = self.default_pred(kps, kps)

        results = eval_matches_homography(data, pred)

        self.assertEqual(results["prec@1px"], 1)
        self.assertEqual(results["prec@3px"], 1)
        self.assertEqual(results["num_matches"], 4)
        self.assertEqual(results["num_keypoints"], 4)

    def test_eval_matches_homography_real(self):
        data = {"H_0to1": torch.tensor([[1.5, 0.2, 21], [-0.3, 1.6, 33], [0, 0, 1.0]])}
        kps0 = self.default_pts()
        kps1 = warp_points_torch(kps0, data["H_0to1"], inverse=False)
        pred = self.default_pred(kps0, kps1)

        results = eval_matches_homography(data, pred)

        self.assertEqual(results["prec@1px"], 1)
        self.assertEqual(results["prec@3px"], 1)

    def test_eval_matches_homography_real_outliers(self):
        data = {"H_0to1": torch.tensor([[1.5, 0.2, 21], [-0.3, 1.6, 33], [0, 0, 1.0]])}
        kps0 = self.default_pts()
        kps0 = torch.cat([kps0, torch.tensor([[5.0, 5.0]])])
        kps1 = warp_points_torch(kps0, data["H_0to1"], inverse=False)
        # Move one keypoint 1.5 pixels away in x and y
        kps1[-1] += 1.5
        pred = self.default_pred(kps0, kps1)

        results = eval_matches_homography(data, pred)
        self.assertAlmostEqual(results["prec@1px"], 0.8)
        self.assertAlmostEqual(results["prec@3px"], 1.0)

    def test_eval_matches_homography_batched(self):
        H0 = torch.tensor([[1.5, 0.2, 21], [-0.3, 1.6, 33], [0, 0, 1.0]])
        H1 = torch.tensor([[0.7, 0.1, -5], [-0.1, 0.65, 13], [0, 0, 1.0]])
        data = {"H_0to1": torch.stack([H0, H1])}
        kps0 = torch.stack([self.default_pts(), self.default_pts().flip(0)])
        kps1 = warp_points_torch(kps0, data["H_0to1"], inverse=False)
        # In the first element of the batch there is one outlier
        kps1[0, -1] += 5
        matches0 = torch.stack([torch.arange(4), torch.arange(4)])
        # In the second element of the batch there is only 2 matches
        matches0[1, :2] = -1
        pred = {
            "keypoints0": kps0,
            "keypoints1": kps1,
            "matches0": matches0,
            "matching_scores0": torch.ones_like(matches0),
        }

        results = eval_matches_homography(data, pred)
        self.assertAlmostEqual(results["prec@1px"][0], 0.75)
        self.assertAlmostEqual(results["prec@1px"][1], 1.0)
        self.assertAlmostEqual(results["num_matches"][0], 4)
        self.assertAlmostEqual(results["num_matches"][1], 2)
