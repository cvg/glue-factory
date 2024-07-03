"""
Inference model of SuperPoint, a feature detector and descriptor.

Described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork
"""

import numpy as np
import torch
from torch import nn
from pathlib import Path
from kornia.geometry.transform import warp_perspective
from kornia.morphology import erosion

from gluefactory.models.base_model import BaseModel
from gluefactory.datasets.homographies import sample_homography_corners
from ...settings import DATA_PATH


def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        size: an interger scalar, the radius of the NMS window.
    """
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius*2+1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, b, h, w):
    mask_h = (keypoints[1] >= b) & (keypoints[1] < (h - b))
    mask_w = (keypoints[2] >= b) & (keypoints[2] < (w - b))
    mask = mask_h & mask_w
    return (keypoints[0][mask], keypoints[1][mask], keypoints[2][mask])


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2*radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
        scores[:, None], width, 1, radius, divisor_override=1)
    sum_ = torch.clamp(sum_, min=1e-6)
    ar = torch.arange(-radius, radius+1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(
        scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
        scores[:, None], kernel_x.transpose(2, 3), padding=radius)
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints


# Legacy (broken) sampling of the descriptors
def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors.permute(0, 2, 1)


# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8,
                                    normalize=True):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2),
        mode='bilinear', align_corners=False).reshape(b, c, -1)
    if normalize:
        descriptors = torch.nn.functional.normalize(
            descriptors, p=2, dim=1)
    return descriptors.permute(0, 2, 1)


class SuperPoint(BaseModel):
    default_conf = {
        'has_detector': True,
        'has_descriptor': True,
        'descriptor_dim': 256,

        # Inference
        'return_all': False,
        'sparse_outputs': True,
        'nms_radius': 4,
        'refinement_radius': 0,
        'detection_threshold': 0.005,
        'max_num_keypoints': -1,
        'force_num_keypoints': False,
        'remove_borders': 4,
        'legacy_sampling': False,  # True to use the old broken sampling
        
        # Homography adaptation
        'ha': {
            'enable': False,
            'num_H': 10,
            'mini_bs': 5,
            'aggregation': 'mean',
            'H_params': {
                'difficulty': 0.8,
                'translation': 1.0,
                'max_angle': 60,
                'n_angles': 10,
                'min_convexity': 0.05
            }
        }
    }
    required_data_keys = ['image']

    def _init(self, conf):
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        if conf.has_detector:
            self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        if conf.has_descriptor:
            self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convDb = nn.Conv2d(
                c5, conf.descriptor_dim, kernel_size=1, stride=1, padding=0)

        path = Path(DATA_PATH, 'weights/superpoint_v1.pth')
        self.load_state_dict(torch.load(str(path)), strict=False)

        self.erosion_kernel = torch.tensor(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=torch.float
        )

    def backbone(self, image):
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        return x

    def kp_head(self, feat):
        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(feat))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        return scores

    def _forward(self, data):
        image = data['image']
        b_size = len(image)
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image*scale).sum(1, keepdim=True)

        # Shared Encoder
        x = self.backbone(image)

        pred = {}
        if self.conf.has_detector and self.conf.max_num_keypoints != 0:
            # Heatmap prediction
            if self.conf.ha.enable:
                scores = self.homography_adaptation(image)
            else:
                scores = self.kp_head(x)
            pred['keypoint_scores'] = dense_scores = scores
            h, w = scores.shape[1] // 8, scores.shape[2] // 8
        if self.conf.has_descriptor:
            # Compute the dense descriptors
            cDa = self.relu(self.convDa(x))
            all_desc = self.convDb(cDa)
            all_desc = torch.nn.functional.normalize(all_desc, p=2, dim=1)
            pred['descriptors'] = all_desc

            if self.conf.max_num_keypoints == 0:  # Predict dense descriptors only
                device = image.device
                return {
                    'keypoints': torch.empty(b_size, 0, 2, device=device),
                    'keypoint_scores': torch.empty(b_size, 0, device=device),
                    'descriptors': torch.empty(b_size, self.conf.descriptor_dim, 0, device=device),
                    'all_descriptors': all_desc
                }

        if self.conf.sparse_outputs:
            assert self.conf.has_detector and self.conf.has_descriptor

            scores = simple_nms(scores, self.conf.nms_radius)

            # Extract keypoints
            best_kp = torch.where(scores > self.conf.detection_threshold)

            # Discard keypoints near the image borders
            best_kp = remove_borders(best_kp, self.conf.remove_borders,
                                     h * 8, w * 8)
            scores = scores[best_kp]

            # Separate into batches
            keypoints = [torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i]
                         for i in range(b_size)]
            scores = [scores[best_kp[0] == i] for i in range(b_size)]

            # Keep the k keypoints with highest score
            if self.conf.max_num_keypoints > 0:
                keypoints, scores = list(zip(*[
                    top_k_keypoints(k, s, self.conf.max_num_keypoints)
                    for k, s in zip(keypoints, scores)]))
                keypoints, scores = list(keypoints), list(scores)

            if self.conf['refinement_radius'] > 0:
                keypoints = soft_argmax_refinement(
                    keypoints, dense_scores, self.conf['refinement_radius'])

            # Convert (h, w) to (x, y)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]

            if self.conf.force_num_keypoints:
                _, _, h, w = data['image'].shape
                assert self.conf.max_num_keypoints > 0
                scores = list(scores)
                for i in range(len(keypoints)):
                    k, s = keypoints[i], scores[i]
                    missing = self.conf.max_num_keypoints - len(k)
                    if missing > 0:
                        new_k = torch.rand(missing, 2).to(k)
                        new_k = new_k * k.new_tensor([[w-1, h-1]])
                        new_s = torch.zeros(missing).to(s)
                        keypoints[i] = torch.cat([k, new_k], 0)
                        scores[i] = torch.cat([s, new_s], 0)

            # Batch the data
            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                keypoints = torch.stack(keypoints, 0)
                scores = torch.stack(scores, 0)

            # Extract descriptors
            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                # Batch sampling of the descriptors
                if self.conf.legacy_sampling:
                    desc = sample_descriptors(keypoints, all_desc, 8)
                else:
                    desc = sample_descriptors_fix_sampling(
                        keypoints, all_desc, 8)
            else:
                if self.conf.legacy_sampling:
                    desc = [sample_descriptors(k[None], d[None], 8)[0]
                            for k, d in zip(keypoints, all_desc)]
                else:
                    desc = [sample_descriptors_fix_sampling(
                        k[None], d[None], 8)[0]
                            for k, d in zip(keypoints, all_desc)]

            pred = {
                'keypoints': keypoints,
                'keypoint_scores': scores,
                'descriptors': desc,
            }

            if self.conf.return_all:
                pred['all_descriptors'] = all_desc
                pred['dense_score'] = dense_scores
            else:
                del all_desc
                torch.cuda.empty_cache()

        return pred

    def extract_kp_only(self, image):
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image*scale).sum(1, keepdim=True)
        b_size = len(image)

        pred = {}
        # Heatmap prediction
        if self.conf.ha.enable:
            scores = self.homography_adaptation(image)
        else:
            x = self.backbone(image)
            scores = self.kp_head(x)
        pred['keypoint_scores'] = dense_scores = scores
        h, w = scores.shape[1] // 8, scores.shape[2] // 8

        if self.conf.sparse_outputs:
            scores = simple_nms(scores, self.conf.nms_radius)

            # Extract keypoints
            best_kp = torch.where(scores > self.conf.detection_threshold)

            # Discard keypoints near the image borders
            best_kp = remove_borders(best_kp, self.conf.remove_borders,
                                     h * 8, w * 8)
            scores = scores[best_kp]

            # Separate into batches
            keypoints = [torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i]
                         for i in range(b_size)]
            scores = [scores[best_kp[0] == i] for i in range(b_size)]

            # Keep the k keypoints with highest score
            if self.conf.max_num_keypoints > 0:
                keypoints, scores = list(zip(*[
                    top_k_keypoints(k, s, self.conf.max_num_keypoints)
                    for k, s in zip(keypoints, scores)]))
                keypoints, scores = list(keypoints), list(scores)

            if self.conf['refinement_radius'] > 0:
                keypoints = soft_argmax_refinement(
                    keypoints, dense_scores, self.conf['refinement_radius'])

            # Convert (h, w) to (x, y)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]

            if self.conf.force_num_keypoints:
                _, _, h, w = image.shape
                assert self.conf.max_num_keypoints > 0
                scores = list(scores)
                for i in range(len(keypoints)):
                    k, s = keypoints[i], scores[i]
                    missing = self.conf.max_num_keypoints - len(k)
                    if missing > 0:
                        new_k = torch.rand(missing, 2).to(k)
                        new_k = new_k * k.new_tensor([[w-1, h-1]])
                        new_s = torch.zeros(missing).to(s)
                        keypoints[i] = torch.cat([k, new_k], 0)
                        scores[i] = torch.cat([s, new_s], 0)

            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                keypoints = torch.stack(keypoints, 0)
                scores = torch.stack(scores, 0)

            pred = {
                'keypoints': keypoints,
                'keypoint_scores': scores,
            }

            if self.conf.return_all:
                pred['dense_score'] = dense_scores

        return pred

    def homography_adaptation(self, img):
        """ Perform homography adaptation on the score heatmap. """
        bs = self.conf.ha.mini_bs
        num_H = self.conf.ha.num_H
        device = img.device
        self.erosion_kernel = self.erosion_kernel.to(device)
        B, _, h, w = img.shape

        # Generate homographies
        Hs = []
        for i in range(num_H):
            if i == 0:
                # Always include at least the identity
                Hs.append(torch.eye(3, dtype=torch.float, device=device))
            else:
                Hs.append(torch.tensor(
                    sample_homography_corners(
                        (w, h), patch_shape=(w, h),
                        **self.conf.ha.H_params)[0],
                    dtype=torch.float, device=device))
        Hs = torch.stack(Hs, dim=0)

        # Loop through all mini batches
        n_mini_batch = int(np.ceil(num_H / bs))
        scores = torch.empty((B, 0, h, w), dtype=torch.float, device=device)
        counts = torch.empty((B, 0, h, w), dtype=torch.float, device=device)
        for i in range(n_mini_batch):
            H = Hs[i*bs:(i+1)*bs]
            nh = len(H)
            H = H.repeat(B, 1, 1)

            # Warp the image
            warped_imgs = warp_perspective(
                torch.repeat_interleave(img, nh, dim=0),
                H, (h, w), mode='bilinear')

            # Forward pass
            with torch.no_grad():
                score = self.kp_head(self.backbone(warped_imgs))

                # Compute valid pixels
                H_inv = torch.inverse(H)
                count = warp_perspective(
                    torch.ones_like(score).unsqueeze(1),
                    H, (h, w), mode='nearest')
                count = erosion(count, self.erosion_kernel)
                count = warp_perspective(count, H_inv, (h, w),
                                         mode='nearest')[:, 0]

                # Warp back the scores
                score = warp_perspective(score[:, None], H_inv, (h, w),
                                         mode='bilinear')[:, 0]

            # Aggregate the results
            scores = torch.cat([scores, score.reshape(B, nh, h, w)], dim=1)
            counts = torch.cat([counts, count.reshape(B, nh, h, w)], dim=1)

        # Aggregate the results
        if self.conf.ha.aggregation == 'mean':
            score = (scores * counts).sum(dim=1) / counts.sum(dim=1)
        elif self.conf.ha.aggregation == 'median':
            scores[counts == 0] = np.nan
            score = torch.nanmedian(scores, dim=1)[0]
        elif self.conf.ha.aggregation == 'max':
            scores[counts == 0] = 0
            score = scores.max(dim=1)[0]
        else:
            raise ValueError("Unknown aggregation method: " + self.conf.ha.aggregation)

        return score

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
