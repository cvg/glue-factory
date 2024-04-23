"""
Retrained and improved version of SuperPoint, a feature detector and descriptor.

Originally described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from kornia.geometry.transform import warp_perspective
from kornia.morphology import erosion

from gluefactory.models.base_model import BaseModel
from gluefactory.datasets.homographies import sample_homography_corners
from .backbones.resnet50 import ResNet50
from .backbones.resnet_custom import ResNetCustom
from .backbones.lcnn_hourglass_net import LCNNHourglassBackbone
from .backbones.vgg import VGG
from .superpoint import (simple_nms, remove_borders, top_k_keypoints,
                         soft_argmax_refinement,
                         sample_descriptors_fix_sampling)
from ..geometry.kp_losses import kp_ce_loss, soft_argmax_only_loss
from ..geometry.desc_losses import triplet_loss, nll_loss, caps_window_loss
from ..geometry.metrics import get_repeatability_and_loc_error, matching_score


class SuperPointV2(BaseModel):
    default_conf = {
        'backbone': {
            'name': 'resnet_custom',  # 'resnet', 'lcnn_hourglass', 'resnet_custom', or 'vgg'
            'blocks': [2, 2, 2],
            'params': {
                'input_channel': 3,
                'depth': 4,
                'num_stacks': 2,
                'num_blocks': 1,
                'num_classes': 5,
            }
        },
        'has_detector': True,
        'has_descriptor': False,

        # Inference
        'sparse_outputs': False,
        'nms_radius': 4,
        'refinement_radius': 0,
        'detection_threshold': 0.005,
        'max_num_keypoints': -1,
        'force_num_keypoints': False,
        'remove_borders': 4,

        # Descriptor
        'descriptor_dim': 128,
        'desc_loss': 'triplet',  # 'triplet', 'nll', or 'caps'
        'margin': 1.,
        'dist_thresh': 16,
        'temperature': 50.,  # if = 'learned', learn it as a parameter

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
            },
        },

        # Loss weights
        'loss_weights': {
            'type': 'static',  # 'static' or 'dynamic'
            'kp': 1.,
            'loc': 1.,
            'desc': 1.,
        }
    }
    required_data_keys = ['image']

    def _init(self, conf):
        # Backbone network
        if self.conf.backbone.name == 'lcnn_hourglass':
            self.backbone = LCNNHourglassBackbone(**self.conf.backbone.params)
            dense_feat_dim = 256
        elif self.conf.backbone.name == 'resnet':
            self.backbone = ResNet50()
            dense_feat_dim = 512
        elif self.conf.backbone.name == 'resnet_custom':
            self.backbone = ResNetCustom(self.conf.backbone.blocks,
                                         last_activation=None)
            dense_feat_dim = 256
        elif self.conf.backbone.name == 'vgg':
            self.backbone = VGG()
            dense_feat_dim = 128
        else:
            raise ValueError("Unknown backbone: " + self.conf.backbone.name)

        if conf.has_detector:
            self.convPa = nn.Conv2d(dense_feat_dim, 256, kernel_size=3,
                                    stride=1, padding=1)
            self.convPb = nn.Conv2d(256, 65, kernel_size=1,
                                    stride=1, padding=0)

        if conf.has_descriptor:
            self.convDa = nn.Conv2d(dense_feat_dim, 256, kernel_size=3,
                                    stride=1, padding=1)
            self.convDb = nn.Conv2d(256, conf.descriptor_dim, kernel_size=1,
                                    stride=1, padding=0)

        if self.conf.loss_weights.type == 'dynamic':
            self.kp_w = nn.Parameter(torch.tensor(self.conf.loss_weights.kp,
                                     dtype=torch.float32), requires_grad=True)
            self.loc_w = nn.Parameter(torch.tensor(self.conf.loss_weights.loc,
                                     dtype=torch.float32), requires_grad=True)
            if self.conf.loss_weights.desc > 0:
                self.desc_w = nn.Parameter(
                    torch.tensor(self.conf.loss_weights.desc,
                                 dtype=torch.float32), requires_grad=True)

        self.erosion_kernel = torch.tensor(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=torch.float
        )

        if self.conf.temperature == 'learned':
            temperature = nn.Parameter(torch.tensor(50.))
            self.register_parameter('temperature', temperature)

    def kp_head(self, feat):
        # Compute the dense keypoint scores
        cPa = F.relu(self.convPa(feat))
        logits = self.convPb(cPa)
        scores = torch.nn.functional.softmax(logits, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        return scores, logits

    def _forward(self, data):
        image = data['image']

        # Shared Encoder
        x = self.backbone(image)

        pred = {}
        if self.conf.has_detector and self.conf.max_num_keypoints != 0:
            # Heatmap prediction
            if self.conf.ha.enable:
                dense_scores = self.homography_adaptation(image)
            else:
                dense_scores, logits = self.kp_head(x)
                pred['logits'] = logits
            pred['dense_score'] = dense_scores
        if self.conf.has_descriptor:
            # Compute the dense descriptors
            dense_desc = F.relu(self.convDa(x))
            dense_desc = self.convDb(dense_desc)
            dense_desc = torch.nn.functional.normalize(dense_desc, p=2, dim=1)
            pred['dense_desc'] = dense_desc

        if self.conf.sparse_outputs and self.conf.has_detector:
            sparse_pred = self.get_sparse_outputs(
                image, dense_scores,
                dense_desc=(dense_desc if self.conf.has_descriptor else None),
                orig_size=([*zip(data["height"], data["width"])]
                           if "height" in data else None))
            pred = {**pred, **sparse_pred}

        return pred

    def get_sparse_outputs(self, image, dense_scores, dense_desc=None,
                           orig_size=None):
        """ Extract sparse feature points from dense scores and descriptors. """
        b_size, _, h, w = image.shape
        device = image.device
        pred = {}

        if self.conf.max_num_keypoints == 0:
            pred['keypoints'] = torch.empty(b_size, 0, 2, device=device)
            pred['keypoint_scores'] = torch.empty(b_size, 0, device=device)
            pred['descriptors'] = torch.empty(
                b_size, self.conf.descriptor_dim, 0, device=device)
            return pred

        scores = simple_nms(dense_scores, self.conf.nms_radius)

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

        # Label keypoints outside of the original image size as invalid
        if orig_size is not None:
            for i in range(b_size):
                scores[i][(keypoints[i][:, 0] >= orig_size[i][0])
                           | (keypoints[i][:, 1] >= orig_size[i][1])] = 0

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

        pred['keypoints'] = keypoints
        pred['keypoint_scores'] = scores

        if self.conf.has_descriptor:
            # Extract descriptors
            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                # Batch sampling of the descriptors
                pred['descriptors'] = sample_descriptors_fix_sampling(
                    keypoints, dense_desc, 8, normalize=True)
            else:
                pred['descriptors'] = [sample_descriptors_fix_sampling(
                    k[None], d[None], 8)[0]
                                       for k, d in zip(keypoints, dense_desc)]

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
                score = self.kp_head(self.backbone(warped_imgs))[0]

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
            raise ValueError("Unknown aggregation method: "
                             + self.conf.ha.aggregation)

        return score

    def loss(self, pred, data):
        losses = {}
        loss = 0
        H = data['H_0to1']
        valid_kp0 = data["keypoint_scores0"] > 0
        valid_kp1 = data["keypoint_scores1"] > 0

        # Cross entropy loss for keypoints
        kp_loss0 = kp_ce_loss(
            pred['logits0'], data['keypoints0'], valid_kp0,
            valid_mask=(None if 'valid_mask0' not in data
                        else data['valid_mask0']))
        kp_loss1 = kp_ce_loss(
            pred['logits1'], data['keypoints1'], valid_kp1,
            valid_mask=(None if 'valid_mask1' not in data
                        else data['valid_mask1']))
        kp_loss = (kp_loss0 + kp_loss1) / 2
        losses['kp_loss'] = kp_loss
        if self.conf.loss_weights.type == 'static':
            loss += losses['kp_loss'] * self.conf.loss_weights.kp
        else:
            loss += losses['kp_loss'] * torch.exp(-self.kp_w) + self.kp_w

        # Soft argmax loss
        if self.conf.refinement_radius > 0 and self.conf.loss_weights.loc > 0:
            loc_loss = soft_argmax_only_loss(
                pred['dense_score0'], pred['dense_score1'],
                data['keypoints0'], valid_kp0,
                H, self.conf.refinement_radius)
            losses['loc_loss'] = loc_loss
            if self.conf.loss_weights.type == 'static':
                loss += losses['loc_loss'] * self.conf.loss_weights.loc
            else:
                loss += losses['loc_loss'] * torch.exp(-self.loc_w) + self.loc_w

        # Descriptor loss
        if self.conf.has_descriptor and self.conf.loss_weights.desc > 0:
            if self.conf.desc_loss == 'triplet':
                desc_loss = triplet_loss(
                    data['keypoints0'][:, :, [1, 0]], valid_kp0,
                    H, pred['dense_desc0'], pred['dense_desc1'],
                    self.conf.margin, self.conf.dist_thresh)
            elif self.conf.desc_loss == 'nll':
                desc_loss = nll_loss(
                    data['keypoints0'][:, :, [1, 0]], valid_kp0,
                    H, pred['dense_desc0'], pred['dense_desc1'],
                    self.conf.temperature)
            elif self.conf.desc_loss == 'caps':
                desc_loss0 = caps_window_loss(
                    pred['keypoints0'][:, :, [1, 0]],
                    pred['keypoint_scores0'],
                    pred['descriptors0'], H, pred['dense_desc1'],
                    temperature=(self.temperature
                                 if self.conf.temperature == 'learned'
                                 else self.conf.temperature))
                desc_loss1 = caps_window_loss(
                    pred['keypoints1'][:, :, [1, 0]],
                    pred['keypoint_scores1'],
                    pred['descriptors1'], torch.inverse(H),
                    pred['dense_desc0'],
                    temperature=(self.temperature
                                 if self.conf.temperature == 'learned'
                                 else self.conf.temperature))
                desc_loss = (desc_loss0 + desc_loss1) / 2
            else:
                raise ValueError("Unknown descriptor loss: " + self.conf.desc_loss)
            losses['desc_loss'] = desc_loss
            if self.conf.loss_weights.type == 'static':
                loss += losses['desc_loss'] * self.conf.loss_weights.desc
            else:
                loss += losses['desc_loss'] * torch.exp(-self.desc_w) + self.desc_w

        losses['total'] = loss
        
        if not self.training:
            # add metrics
            metrics = self.metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics

    def get_pr(self, pred_kp, gt_kp, tol=3):
        """ Compute the precision and recall, based on GT KP. """
        if len(gt_kp) == 0:
            precision = float(len(pred_kp) == 0)
            recall = 1.
        elif len(pred_kp) == 0:
            precision = 1.
            recall = float(len(gt_kp) == 0)
        else:
            dist = torch.norm(pred_kp[:, None] - gt_kp[None], dim=2)
            close = (dist < tol).float()
            precision = close.max(dim=1)[0].mean()
            recall = close.max(dim=0)[0].mean()
        return precision, recall

    def metrics(self, pred, data):
        device = pred['dense_score0'].device
        valid_kp0 = data["keypoint_scores0"] > 0
        valid_kp1 = data["keypoint_scores1"] > 0

        # Get sparse output
        if 'keypoints0' not in pred:
            sparse_pred0 = self.get_sparse_outputs(
                data['image0'], pred['dense_score0'],
                dense_desc=(pred['dense_desc0'] if self.conf.has_descriptor
                            else None),
                orig_size=([*zip(data["height"], data["width"])]
                           if "height" in data else None))
            sparse_pred1 = self.get_sparse_outputs(
                data['image1'], pred['dense_score1'],
                dense_desc=(pred['dense_desc1'] if self.conf.has_descriptor
                            else None),
                orig_size=([*zip(data["height"], data["width"])]
                           if "height" in data else None))
            pred = {**pred,
                    **{f'{k}0': v for k, v in sparse_pred0.items()},
                    **{f'{k}1': v for k, v in sparse_pred1.items()}}

        # Compute the precision and recall
        precision, recall = [], []
        for i in range(len(data['keypoints0'])):
            valid_gt_kp0 = data['keypoints0'][i][valid_kp0[i]]
            prec0, rec0 = self.get_pr(pred['keypoints0'][i], valid_gt_kp0)

            valid_gt_kp1 = data['keypoints1'][i][valid_kp1[i]]
            prec1, rec1 = self.get_pr(pred['keypoints1'][i], valid_gt_kp1)

            precision.append((prec0 + prec1) / 2)
            recall.append((rec0 + rec1) / 2)

        # Compute the KP repeatability and localization error
        rep, loc_error = get_repeatability_and_loc_error(
            pred['keypoints0'], pred['keypoints1'], pred['keypoint_scores0'],
            pred['keypoint_scores1'], data['H_0to1'])

        out = {
            'precision': torch.tensor(precision, dtype=torch.float, device=device),
            'recall': torch.tensor(recall, dtype=torch.float, device=device),
            'repeatability': rep, 'loc_error': loc_error
        }

        if self.conf.has_descriptor:
            # Matching score
            out['matching_score'] = matching_score(
                data['keypoints0'][:, :, [1, 0]], valid_kp0,
                data['H_0to1'], pred['dense_desc0'], pred['dense_desc1'])

        return out
