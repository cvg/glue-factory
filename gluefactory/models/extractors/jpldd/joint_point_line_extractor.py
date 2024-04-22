import torch
import torch.nn as nn

from omegaconf import OmegaConf

from gluefactory.models.base_model import BaseModel
from gluefactory.models.extractors.jpldd.backbone_encoder import AlikedEncoder, aliked_cfgs
from gluefactory.models.extractors.jpldd.descriptor_head import SDDH
from gluefactory.models.extractors.jpldd.keypoint_decoder import SMH
from gluefactory.models.extractors.jpldd.keypoint_detection import DKD
from gluefactory.models.extractors.jpldd.utils import InputPadder, change_dict_key

to_ctr = OmegaConf.to_container  # convert DictConfig to dict
checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"


class JointPointLineDetectorDescriptor(BaseModel):
    # currently contains only ALIKED
    default_conf = {
        # ToDo: create default conf once everything is running -> default conf is merged with input conf to the init method!
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "force_num_keypoints": False,
        "pretrained": True,
        "nms_radius": 2,
    }

    n_limit_max = 20000  # taken from ALIKED which gives max num keypoints to detect! ToDo

    line_extractor_cfg = {
        "detect_thresh": 1 / 65,
        "grid_size": 8,
        "junc_detect_thresh": 1 / 65,
        "max_num_junctions": 300
    }

    # other needed conf values:
    # line_neighborhood -> parameter r, determining the radius for normalization

    required_data_keys = ["image"]

    def _init(self, conf):
        print(f"final config dict(type={type(conf)}): {conf}")
        # c1-c4 -> output dimensions of encoder blocks, dim -> dimension of hidden feature map
        # K=Kernel-Size, M=num sampling pos
        self.conf = conf
        aliked_model_cfg = aliked_cfgs[conf.model_name]
        dim = aliked_model_cfg["dim"]
        K = aliked_model_cfg["K"]
        M = aliked_model_cfg["M"]
        # Load Network Components
        print(f"aliked cfg(type={type(aliked_model_cfg)}): {aliked_model_cfg}")
        self.encoder_backbone = AlikedEncoder(aliked_model_cfg)
        self.keypoint_and_junction_branch = SMH(dim)  # using SMH from ALIKE here
        self.dkd = DKD(radius=conf.nms_radius,
                       top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
                       scores_th=conf.detection_threshold,
                       n_limit=(
                           conf.max_num_keypoints
                           if conf.max_num_keypoints > 0
                           else self.n_limit_max
                       ), )  # Differentiable Keypoint Detection from ALIKE
        # Keypoint and line descriptors
        self.descriptor_branch = SDDH(dim, K, M, gate=nn.SELU(inplace=True), conv2D=False, mask=False)
        self.line_descriptor = torch.lerp  # we take the endpoints of lines and interpolate to get the descriptor
        # Line Attraction Field information (Line Distance Field and Angle Field)
        self.distance_field_branch = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
        )
        self.angle_field_branch = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # ToDo Figure out heuristics
        # self.line_extractor = LineExtractor(torch.device("cpu"), self.line_extractor_cfg)

        # loss
        self.l1_loss_fn = nn.L1Loss(reduction='none')
        self.l2_loss_fn = nn.MSELoss(reduction='none')

        # load pretrained_elements if wanted (for now that only the ALIKED parts of the network)
        if conf.pretrained:
            old_test_val = self.encoder_backbone.conv1.weight.data.clone()
            self.load_pretrained_elements()
            assert not torch.all(torch.eq(self.encoder_backbone.conv1.weight.data.clone(), old_test_val)).item() # test if weights really loaded!

    # Utility methods for line df and af with deepLSD
    def normalize_df(self, df):
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm):
        return torch.exp(-df_norm) * self.conf.line_neighborhood

    def _forward(self, data):
        # load image and padder
        image = data["image"]
        div_by = 2 ** 5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)

        # Get Hidden Feature Map and Keypoint/junction scoring
        feature_map_padded = self.encoder_backbone(padder.pad(image))
        score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)
        feature_map_padded_normalized = torch.nn.functional.normalize(feature_map_padded, p=2, dim=1)
        feature_map = padder.unpad(feature_map_padded_normalized)
        print(
            f"Image size: {image.shape}\nFeatureMap-unpadded: {feature_map.shape}\nFeatureMap-padded: {feature_map_padded.shape}")
        assert (feature_map.shape[2], feature_map.shape[3]) == (image.shape[2], image.shape[3])
        keypoint_and_junction_score_map = padder.unpad(score_map_padded)

        # Line Elements
        line_angle_field = self.angle_field_branch(feature_map)
        line_distance_field = self.distance_field_branch(feature_map)

        keypoints, kptscores, scoredispersitys = self.dkd(
            keypoint_and_junction_score_map, image_size=data.get("image_size")
            # ToDo: image_size in data, not enough to get from image itself?? (maybe before transformation)
        )

        # DKD gives list with Tensor containing Keypoints, convert to tensor
        junct = torch.stack(keypoints, dim=0)

        # ToDo: Implement/Find good line extraction based on junctions, af, df
        # line_map, junctions, heatmap = self.line_extractor(junct, line_heatmap)
        # line_segments = line_map_to_segments(junctions, line_map)
        line_segments = None

        keypoint_descriptors, offsets = self.descriptor_branch(feature_map, keypoints)
        # TODO: can we make sure endpoints are always keypoints?! + Fix Input to this function
        # line_descriptors = self.line_descriptor(line_segments[0], line_segments[1], 0.5)
        line_descriptors = None

        _, _, h, w = image.shape
        wh = torch.tensor([w, h], device=image.device)
        # no padding required,
        # we can set detection_threshold=-1 and conf.max_num_keypoints
        return {
            "keypoints": wh * (torch.stack(keypoints) + 1) / 2.0,  # B N 2
            "keypoint_descriptors": torch.stack(keypoint_descriptors),  # B N D
            "keypoint_scores": torch.stack(kptscores),  # B N
            "score_dispersity": torch.stack(scoredispersitys),
            "keypoint_and_junction_score_map": keypoint_and_junction_score_map,  # B x 1 x H x W
            "line_anglefield": line_angle_field,
            "line_distancefield": line_distance_field,
            "line_endpoints": line_segments,  # as tuples
            "line_descriptors": line_descriptors,  # as vectors
        }

    def loss(self, pred, data):
        raise NotImplementedError

    def load_pretrained_elements(self):
        # Load state-dict of wanted aliked-model
        aliked_state_url = checkpoint_url.format(self.conf.model_name)
        aliked_state_dict = torch.hub.load_state_dict_from_url(aliked_state_url, map_location="cpu")
        # change keys
        for k, v in list(aliked_state_dict.items()):
            if k.startswith("block") or k.startswith("conv"):
                change_dict_key(aliked_state_dict, k, f"encoder_backbone.{k}")
            elif k.startswith("score_head"):
                change_dict_key(aliked_state_dict, k, f"keypoint_and_junction_branch.{k[11:]}")
            elif k.startswith("desc_head"):
                change_dict_key(aliked_state_dict, k, f"descriptor_branch.{k[10:]}")
            else:
                continue
        # load values
        self.load_state_dict(aliked_state_dict, strict=False)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
