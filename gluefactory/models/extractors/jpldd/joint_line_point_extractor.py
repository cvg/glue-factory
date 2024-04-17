import torch

from omegaconf import OmegaConf

from gluefactory.models.base_model import BaseModel
from gluefactory.models.extractors.jpldd.backbone_encoder import AlikedEncoder, aliked_cfgs
from gluefactory.models.extractors.jpldd.descriptor_head import SDDH
from gluefactory.models.extractors.jpldd.keypoint_decoder import SMH
from gluefactory.models.extractors.jpldd.keypoint_detection import DKD
from gluefactory.models.extractors.jpldd.utils import InputPadder
from gluefactory.models.extractors.jpldd.line_heatmap_decoder import PixelShuffleDecoder
from gluefactory.models.extractors.jpldd.line_detection import LineSegmentDetectionModule
from gluefactory.models.extractors.jpldd.utils import line_map_to_segments


to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class JointPointLineDetectorDescriptor(BaseModel):
    aliked_conf = {  # ToDo: create default conf once everything is running
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "force_num_keypoints": False,
        "pretrained": True,
        "nms_radius": 2,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        # get configurations
        # c1-c4 -> output dimensions of encoder blocks, dim -> dimension of hidden feature map
        # K=Kernel-Size, M=num sampling pos
        c1, c2, c3, c4, dim, K, M = [v for _, v in aliked_cfgs[conf.model_name].items()]
        # Load Network Components
        self.encoder = AlikedEncoder(c1, c2, c3, c4, dim)
        self.keypoint_and_junction_branch = SMH(dim)  # using SMH from ALIKE here
        self.dkd = DKD(radius=conf.nms_radius,
                       top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
                       scores_th=conf.detection_threshold,
                       n_limit=(
                           conf.max_num_keypoints
                           if conf.max_num_keypoints > 0
                           else self.n_limit_max
                       ), )  # Differentiable Keypoint Detection from ALIKE
        conv2D = False
        mask = False
        self.descriptor_branch = SDDH(dim, K, M, gate=self.gate, conv2D=conv2D, mask=mask)
        self.line_heatmap_branch = PixelShuffleDecoder(input_feat_dim=dim)  # Use SOLD2 branch
        self.line_extractor = LineSegmentDetectionModule()  # USe SOLD2 one
        self.line_descriptor = torch.lerp  # we take the endpoints of lines and interpolate to get the descriptor

    def _forward(self, data):
        # load image and padder
        image = data["image"]
        div_by = 2 ** 5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)

        # Get Hidden Feature Map and Keypoint/junction scoring
        feature_map_padded = self.encoder(padder.pad(image))
        score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)
        feature_map_padded_normalized = torch.nn.functional.normalize(feature_map_padded, p=2, dim=1)
        feature_map = padder.unpad(feature_map_padded_normalized)
        keypoint_and_junction_score_map = padder.unpad(score_map_padded)

        line_heatmap = self.line_heatmap_branch.forward(feature_map)

        keypoints, kptscores, scoredispersitys = self.dkd(
            keypoint_and_junction_score_map, image_size=data.get("image_size")
        )

        # ToDo: Does it work well to use keypoints for juctions?? -> Design decision
        # ToDo: need preprocessing like in SOLD2 repo before passing it to line extractor?
        line_map, junctions, heatmap = self.line_extractor.detect(keypoints, line_heatmap)
        line_segments = line_map_to_segments(junctions, line_map)

        descriptors, offsets = self.desc_head(feature_map, keypoints)
        # TODO: can we make sure endpoints are always keypoints?! + Fix Input to this function
        line_descriptors = self.line_descriptor(line_segments[0], line_segments[1], 0.5)  # TODO: Interpolate line-endpoint descriptors

        _, _, h, w = image.shape
        wh = torch.tensor([w, h], device=image.device)
        # no padding required,
        # we can set detection_threshold=-1 and conf.max_num_keypoints
        return {
            "keypoints": wh * (torch.stack(keypoints) + 1) / 2.0,  # B N 2
            "keypoint_descriptors": torch.stack(descriptors),  # B N D
            "keypoint_scores": torch.stack(kptscores),  # B N
            "score_dispersity": torch.stack(scoredispersitys),
            "score_map": keypoint_and_junction_score_map,  # Bx1xHxW
            "line_heatmap": heatmap,
            "line_endpoints": line_segments,  # as tuples
            "line_descriptors": line_descriptors  # as vectors
        }

    def loss(self, pred, data):
        raise NotImplementedError
