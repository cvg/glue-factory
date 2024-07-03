from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet
import numpy as np
import matplotlib.pyplot as plt

from gluefactory.models.base_model import BaseModel
from ..lines.pold2_extractor import LineExtractor
from ...models.extractors.aliked import ALIKED
from ...geometry.kp_losses import kp_ce_loss, soft_argmax_only_loss, kp_bce_loss
from ...geometry.desc_losses import triplet_loss, nll_loss, caps_window_loss
from ...geometry.metrics import get_repeatability_and_loc_error, matching_score
from .superpoint import simple_nms, remove_borders, top_k_keypoints, soft_argmax_refinement, sample_descriptors_fix_sampling
from functools import cmp_to_key

# coordinates system
#  ------------------------------>  [ x: range=-1.0~1.0; w: range=0~W ]
#  | -----------------------------
#  | |                           |
#  | |                           |
#  | |                           |
#  | |         image             |
#  | |                           |
#  | |                           |
#  | |                           |
#  | |---------------------------|
#  v
# [ y: range=-1.0~1.0; h: range=0~H ]


class ALIKED_backbone(ALIKED):

    def __init__(self, conf):
        super().__init__(conf)
    
    def forward(self, data):
        image = data["image"]
        feature_map, score_map = self.extract_dense_map(image)
        return {
            "score_map": score_map.squeeze(1) if score_map is not None else None,
            "feature_map": feature_map
        }


class JointPointLine(BaseModel):

    default_conf = {
        'backbone': {
            "model_name": "aliked-n16",
            "max_num_keypoints": -1,
            "detection_threshold": 0.2,
            "force_num_keypoints": False,
            "get_scores": True,
            "has_detector": False,
            "has_descriptor": False,
            "pretrained": True,
            "trainable": True,
            "nms_radius": 2,
        },
        'has_detector': True,
        'has_descriptor': False,
        'has_8x8_detection': False,

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
        'desc_loss': 'caps',  # 'triplet', 'nll', or 'caps'
        'temperature': 50.,  # if = 'learned', learn it as a parameter

        # Loss weights
        'loss_weights': {
            'type': 'static',  # 'static' or 'dynamic'
            'kp': 1.,
            'loc': 1.,
            'desc': 1.,
            'df': 1.,
            'angle': 1.,
        },

        # line detection
        'has_line_detection': True,
        'sharpen': True,
        'line_neighborhood': 5,
        'is_eval': False
    }
    
    def _init(self, conf):
        if conf.has_8x8_detection:
            if conf.backbone.get_scores == True:
                raise ValueError(
                    "8x8 detection is enabled. Set get_scores in Backbone to False."
                )
        else:
            if conf.backbone.get_scores == False:
                raise ValueError(
                    "8x8 detection is disabled. Set get_scores in Backbone to True."
                )

        self.backbone = ALIKED_backbone(conf.backbone)
        
        dense_feat_dim = self.backbone.cfgs[self.backbone.conf.model_name]["dim"]

        # 8x8 Patch Score head
        if conf.has_8x8_detection:
            # Features to score map
            self.score_map_head = nn.Sequential(
                nn.Conv2d(dense_feat_dim, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
            )
            
            # score map to 8x8 patch scores
            self.patch_scores_head = nn.Sequential(
                nn.Conv2d(64, 65, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(65, 65, kernel_size=1, stride=1, padding=0),
            )

        # descriptor head
        if conf.has_descriptor:
            self.desc_head = nn.Sequential(
                nn.Conv2d(dense_feat_dim, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, conf.descriptor_dim, kernel_size=1, stride=1, padding=0),
            )
        
        # line detection head
        if conf.has_line_detection:
            # line distance field head
            self.df_head = nn.Sequential(
                nn.Conv2d(dense_feat_dim, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU()
            )

            # line angle field head
            self.angle_head = nn.Sequential(
                nn.Conv2d(dense_feat_dim, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0),
                nn.Tanh()
            )

            # Loss
            self.l1_loss_fn = nn.L1Loss(reduction='none')
            self.l2_loss_fn = nn.MSELoss(reduction='none')

        # dynamic weighting for different losses
        if self.conf.loss_weights.type == 'dynamic':
            self.kp_w = nn.Parameter(torch.tensor(self.conf.loss_weights.kp,
                                     dtype=torch.float32), requires_grad=True)
            self.loc_w = nn.Parameter(torch.tensor(self.conf.loss_weights.loc,
                                     dtype=torch.float32), requires_grad=True)
            if self.conf.loss_weights.desc > 0:
                self.desc_w = nn.Parameter(
                    torch.tensor(self.conf.loss_weights.desc,
                                 dtype=torch.float32), requires_grad=True)
            if self.conf.loss_weights.df > 0:
                self.df_w = nn.Parameter(
                    torch.tensor(self.conf.loss_weights.df,
                                 dtype=torch.float32), requires_grad=True)
            if self.conf.loss_weights.angle > 0:
                self.angle_w = nn.Parameter(
                    torch.tensor(self.conf.loss_weights.angle,
                                 dtype=torch.float32), requires_grad=True)

        # line extractor
        if self.conf['is_eval']:
            self.line_extractor = LineExtractor(8, 150, "cuda" if torch.cuda.is_available() else "cpu")

        self.set_initialized()

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
        best_kp = remove_borders(best_kp, self.conf.remove_borders, h, w)
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
            """We have dense descriptors (HxWxD) irrespective of H/8xW/8x64 or HxWx1 keypoint detection.
            """
            # Extract descriptors
            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                # Batch sampling of the descriptors
                pred['descriptors'] = sample_descriptors_fix_sampling(
                    keypoints, dense_desc, 1, normalize=True)
            else:
                pred['descriptors'] = [sample_descriptors_fix_sampling(
                    k[None], d[None], 1)[0]
                                       for k, d in zip(keypoints, dense_desc)]

        return pred
    
    def kp_head(self, feat):
        # Compute the dense keypoint scores
        score_map = self.score_map_head(feat)
        b, _, h, w = score_map.shape

        score_map = score_map.permute(0,2,3,1).contiguous().reshape(b, h, w//8, 8)
        score_map = score_map.permute(0,2,3,1).contiguous().reshape(b,w//8,8,h//8,8)
        score_map = score_map.permute(0,3,1,4,2).contiguous().reshape(b, h//8, w//8, 64)
        score_map = score_map.permute(0,3,1,2).contiguous()
        
        logits = self.patch_scores_head(score_map)
        scores = torch.nn.functional.softmax(logits, 1)[:, :-1]

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).contiguous().reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).contiguous().reshape(b, h*8, w*8)

        return scores, logits

    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data):
        outputs = {}

        pred = self.backbone(data)
        dense_feat = pred["feature_map"]
        outputs['score_map'] = pred["score_map"]

        if self.conf.has_8x8_detection:
            # Patch keypoint
            scores, logits = self.kp_head(dense_feat)
            outputs['logits'] = logits
            outputs['score_map'] = scores
        
        if self.conf.has_descriptor:
            # descriptor head
            desc = self.desc_head(dense_feat)
            desc = F.normalize(desc, p=2, dim=1)
            outputs['dense_desc'] = desc

        if self.conf.has_line_detection:

            # DF prediction
            if self.conf.sharpen:
                outputs['df_norm'] = self.df_head(dense_feat).squeeze(1)
                outputs['df'] = self.denormalize_df(outputs['df_norm'])
            else:
                outputs['df'] = self.df_head(dense_feat).squeeze(1)

            # Closest line direction prediction
            # outputs['line_level'] = self.angle_head(dense_feat).squeeze(1) * np.pi
            outputs['line_level'] = self.angle_head(dense_feat)

        if self.conf.sparse_outputs:
            pred_sparse = self.get_sparse_outputs(
                data['image'],
                scores if self.conf.has_8x8_detection else pred["score_map"],
                dense_desc=desc if self.conf.has_descriptor else None,
                orig_size=data.get('image_size'))

            # Adds the keys 'keypoints', 'keypoint_scores', 'descriptors'
            outputs = {**outputs, **pred_sparse}

        if self.conf['is_eval']:
            # Post processing step
            outputs['lines'] = []
            outputs['valid_lines'] = []
            bs = outputs['keypoints'].shape[0]
            for i in range(bs):
                outputs['lines'].append(
                    self.line_extractor.post_processing_step(
                        outputs['keypoints'][i],
                        data['image'][i],
                        outputs['df'][i],
                        outputs['line_level'][i],
                    )
                )
                # outputs['lines'].append(
                #     self.line_extractor.post_processing_step(
                #         outputs['keypoints'][i],
                #         outputs['df'][i],
                #         outputs['line_level'][i]
                #     )
                # )

                if len(outputs['lines'][-1]) == 0:
                    print("NO LINES DETECTED")
                    outputs['lines'][-1] = torch.arange(30).reshape(-1, 2).to(outputs["df"][-1].device)

                outputs['valid_lines'].append(torch.ones(len(outputs['lines'][-1])).to(outputs["df"][-1].device))

        # TODO : Uncomment if eval needs numpy
        # outputs['lines'] = np.array(outputs['lines'])

        return outputs
    
    def df_angle_loss(self, valid_mask, pred_df_norm, pred_df, data_df, pred_af, data_af):

        valid_norm = valid_mask.sum(dim=[1, 2])
        valid_norm[valid_norm == 0] = 1

        # Retrieve the mask of pixels close to GT lines
        line_mask = (valid_mask
                     * (data_df < self.conf.line_neighborhood).float())
        line_norm = line_mask.sum(dim=[1, 2])
        line_norm[line_norm == 0] = 1

        # DF loss, with supervision only on the lines neighborhood
        if self.conf.sharpen:
            df_loss = self.l1_loss_fn(pred_df_norm,
                                      self.normalize_df(data_df))
        else:
            df_loss = self.l1_loss_fn(pred_df, data_df)
            df_loss /= self.conf.line_neighborhood
        df_loss = (df_loss * line_mask).sum(dim=[1, 2]) / line_norm
        
        # Angle loss, with supervision only on the lines neighborhood
        """
        # L2 loss
        angle_loss = torch.minimum(
            (pred_af - data_af) ** 2,
            (np.pi - (pred_af - data_af).abs()) ** 2)
        """

        # Continuous angle loss - 1 - cos(pred, gt)^2
        norm_pred = F.normalize(pred_af, dim=1)
        norm_gt = F.normalize(data_af, dim=1)
        angle_loss = 1 - (norm_pred * norm_gt).sum(dim=1)**2

        angle_loss = (angle_loss * line_mask).sum(dim=[1, 2]) / line_norm

        return df_loss, angle_loss
 
    def loss(self, pred, data):

        # setup data keys
        for k in data['view0']['cache'].keys():
            data[k + "0"] = data['view0']['cache'][k]
            data[k + "1"] = data['view1']['cache'][k]

        losses = {}
        loss = 0
        H = data['H_0to1']
        valid_kp0 = data["keypoint_scores0"] > 0
        valid_kp1 = data["keypoint_scores1"] > 0


        if self.conf.has_detector:
            if self.conf.has_8x8_detection:    
                # Cross entropy loss for keypoints
                kp_loss0 = kp_ce_loss(
                    pred['logits0'], data['keypoints0'], valid_kp0,
                    valid_mask=(None if 'valid_mask0' not in data
                                else data['valid_mask0']))
                kp_loss1 = kp_ce_loss(
                    pred['logits1'], data['keypoints1'], valid_kp1,
                    valid_mask=(None if 'valid_mask1' not in data
                                else data['valid_mask1']))
            else:
                # Binary Cross entropy loss for keypoints
                kp_loss0 = kp_bce_loss(
                    pred['score_map0'], data['keypoints0'], valid_kp0,
                    valid_mask=(None if 'valid_mask0' not in data
                                else data['valid_mask0']))
                kp_loss1 = kp_bce_loss(
                    pred['score_map1'], data['keypoints1'], valid_kp1,
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
                pred['score_map0'], pred['score_map1'],
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
                raise NotImplementedError()
            elif self.conf.desc_loss == 'nll':
                raise NotImplementedError()
            elif self.conf.desc_loss == 'caps':
                """We have dense descriptors (HxWxD) irrespective of H/8xW/8x64 or HxWx1 keypoint detection.
                """
                desc_loss0 = caps_window_loss(
                    pred['keypoints0'][:, :, [1, 0]],
                    pred['keypoint_scores0'],
                    pred['descriptors0'], H, pred['dense_desc1'],
                    temperature=(self.temperature
                                 if self.conf.temperature == 'learned'
                                 else self.conf.temperature),
                    s=1)
                desc_loss1 = caps_window_loss(
                    pred['keypoints1'][:, :, [1, 0]],
                    pred['keypoint_scores1'],
                    pred['descriptors1'], torch.inverse(H),
                    pred['dense_desc0'],
                    temperature=(self.temperature
                                 if self.conf.temperature == 'learned'
                                 else self.conf.temperature),
                    s=1)
                desc_loss = (desc_loss0 + desc_loss1) / 2
            else:
                raise ValueError("Unknown descriptor loss: " + self.conf.desc_loss)
            losses['desc_loss'] = desc_loss
            if self.conf.loss_weights.type == 'static':
                loss += losses['desc_loss'] * self.conf.loss_weights.desc
            else:
                loss += losses['desc_loss'] * torch.exp(-self.desc_w) + self.desc_w

        
        if self.conf.has_line_detection and (self.conf.loss_weights.df > 0 or self.conf.loss_weights.angle > 0):
            df_loss0, angle_loss0 = self.df_angle_loss(
                valid_mask=data['ref_valid_mask0'],
                pred_df_norm=pred['df_norm0'] if self.conf.sharpen else None,
                pred_df=pred['df0'],
                data_df=data['df0'],
                pred_af=pred['line_level0'],
                data_af=data['line_level0']
            )
            df_loss1, angle_loss1 = self.df_angle_loss(
                valid_mask=data['ref_valid_mask1'],
                pred_df_norm=pred['df_norm1'] if self.conf.sharpen else None,
                pred_df=pred['df1'],
                data_df=data['df1'],
                pred_af=pred['line_level1'],
                data_af=data['line_level1']
            )

            df_loss = (df_loss0 + df_loss1) / 2
            angle_loss = (angle_loss0 + angle_loss1) / 2

            if self.conf.loss_weights.df > 0:
                losses['df_loss'] = df_loss
                if self.conf.loss_weights.type == 'static':
                    loss += losses['df_loss'] * self.conf.loss_weights.df
                else:
                    loss += losses['df_loss'] * torch.exp(-self.df_w) + self.df_w
            
            if self.conf.loss_weights.angle > 0:
                losses['angle_loss'] = angle_loss
                if self.conf.loss_weights.type == 'static':
                    loss += losses['angle_loss'] * self.conf.loss_weights.angle
                else:
                    loss += losses['angle_loss'] * torch.exp(-self.angle_w) + self.angle_w

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

    def compute_point_metrics(self, pred, data):
        device = pred['keypoints0'].device
        valid_kp0 = data["keypoint_scores0"] > 0
        valid_kp1 = data["keypoint_scores1"] > 0

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
            'repeatability': rep, 
            'loc_error': loc_error
        }

        if self.conf.has_descriptor:
            # Matching score
            out['matching_score'] = matching_score(
                data['keypoints0'][:, :, [1, 0]], valid_kp0,
                data['H_0to1'], pred['dense_desc0'], pred['dense_desc1'])
            
        return out
    
    def compute_line_metrics(self, pred, data):
        return 0

    def metrics(self, pred, data):
        # TODO: Merge with compute_line_metrics
        return self.compute_point_metrics(pred,data)
    
__main_model__ = JointPointLine
