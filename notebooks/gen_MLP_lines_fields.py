"""
Save DeepLsd and JPLDD outputs in a h5 file, for each image save the following:
    - lines
    - distance field
    - angle field
Usage:
    python -m notebooks.gen_MLP_lines_fields.py
"""

from gluefactory.models.deeplsd_inference import DeepLSD
from gluefactory.models import get_model
from gluefactory.datasets import get_dataset
from gluefactory.settings import DATA_PATH
import torch
import numpy as np
import random
from tqdm import tqdm
from pprint import pprint
import cv2
import glob
from omegaconf import OmegaConf
import h5py as h5

# Configs and Constants
deeplsd_conf = {
    "detect_lines": True,
    "line_detection_params": {
        "merge": False,
        "filtering": True,
        "grad_thresh": 3,
        "grad_nfa": True,
    },
    "weights": "DeepLSD/weights/deeplsd_md.tar",  # path to the weights of the DeepLSD model (relative to DATA_PATH)
}

jpldd_conf = {
    "name": "joint_point_line_extractor",
    "max_num_keypoints": 1000,  # setting for training, for eval: -1
    "timeit": True,  # override timeit: False from BaseModel
    "line_df_decoder_channels": 32,
    "line_af_decoder_channels": 32,
    "line_detection": {
        "do": False,
    },
    "checkpoint": "/local/home/Point-Line/outputs/training/focal_loss_experiments/rk_focal_threshDF_focal/checkpoint_best.tar"
    #"checkpoint": "/local/home/rkreft/shared_team_folder/outputs/training/rk_oxparis_focal_hard_gt/checkpoint_best.tar"
    #"checkpoint": "/local/home/rkreft/shared_team_folder/outputs/training/rk_pold2gt_oxparis_base_hard_gt/checkpoint_best.tar"
}

dset_conf = {
    "reshape": [800, 800], # ex [800, 800]
    "load_features": {
        "do": False,
    },
    "debug": False
}

SAVE_PATH = DATA_PATH / 'DeepLSD-Outputs-OXPA' / 'DeepLSD-Outputs-OXPA.h5'
IMG_GLOB = DATA_PATH / "revisitop1m_POLD2/jpg/**/base_image.jpg"
NUM_IMGS = -1  # Number of images to process, -1 for all

if not SAVE_PATH.parent.exists():
    SAVE_PATH.parent.mkdir(parents=True)

# Set seed
seed = 42
np.random.seed(seed)
random.seed(seed)

# Set device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_built():
    device = 'mps'
else:
    device = 'cpu'
print(f"Device Used: {device}")

## DeepLSD Model
deeplsd_conf = OmegaConf.create(deeplsd_conf)

ckpt_path = DATA_PATH / deeplsd_conf.weights
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
deeplsd_net = DeepLSD(deeplsd_conf)
deeplsd_net.load_state_dict(ckpt["model"])
deeplsd_net = deeplsd_net.to(device).eval()

## JPLDD Model
jpldd_model = get_model("joint_point_line_extractor")(jpldd_conf).to(device)
jpldd_model.eval()

## Dataset
oxpa_2 = get_dataset("oxford_paris_mini_1view_jpldd")(dset_conf)
ds = oxpa_2.get_dataset(split="train")

"""
fps = glob.glob(str(IMG_GLOB), recursive=True)
print(f"Found {len(fps)} images in the OXPA dataset, using {NUM_IMGS} images")
if NUM_IMGS > 0:
    fps = np.random.choice(fps, NUM_IMGS, replace=False)
fps = list(fps)
"""

NUM_IMGS = len(ds) if NUM_IMGS == -1 else NUM_IMGS
print(f"Found {len(ds)} images in the OXPA dataset, using {NUM_IMGS} images")
idx = np.arange(NUM_IMGS)

# Save DeepLSD outputs
for i in tqdm(idx):

    img_torch = ds[i]["image"].to(device).unsqueeze(0)
    img = (img_torch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    c_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

    with torch.no_grad():
        jpldd_output = jpldd_model({"image": img_torch})

    with torch.no_grad():
        inputs = {
            "image": torch.tensor(gray_img, dtype=torch.float, device=device)[
                None, None
            ]
            / 255.0
        }
        deeplsd_output = deeplsd_net(inputs)

    with h5.File(SAVE_PATH, 'a') as f:
        if f"img_{i}" in f:
            del f[f"img_{i}"]
        f.create_group(f"img_{i}")
        f[f"img_{i}"].create_dataset("img", data=img)
        f[f"img_{i}"].create_dataset("deeplsd_lines", data=deeplsd_output["lines"][0])
        f[f"img_{i}"].create_dataset("deeplsd_df", data=deeplsd_output["df"][0].cpu().numpy())
        f[f"img_{i}"].create_dataset("deeplsd_af", data=deeplsd_output["line_level"][0].cpu().numpy())

        f[f"img_{i}"].create_dataset("jpldd_df", data=jpldd_output["line_distancefield"][0].cpu().numpy())
        f[f"img_{i}"].create_dataset("jpldd_af", data=jpldd_output["line_anglefield"][0].cpu().numpy())

print(f"DeepLSD outputs saved at {SAVE_PATH}")