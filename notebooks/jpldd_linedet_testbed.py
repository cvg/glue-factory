"""
Testbed for Joint Point-Line Detection Model (JPLDD) with Line Detection.
Replication of notebooks/refine_line_detection.ipynb as a python script instead of a notebook.
Usage:
    python -m notebooks.jpldd_linedet_testbed.py
"""


from gluefactory.models import get_model
from gluefactory.datasets import get_dataset
from gluefactory.models.deeplsd_inference import DeepLSD
from gluefactory.settings import DATA_PATH
import torch
import numpy as np
import random
from tqdm import tqdm
from pprint import pprint
import cv2
import os
import matplotlib.pyplot as plt
import flow_vis
from omegaconf import OmegaConf

# Configs and Constants
DEBUG_DIR = "tmp_testbed"

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
    "max_num_keypoints": 500,  # setting for training, for eval: -1
    "timeit": True,  # override timeit: False from BaseModel
    "line_df_decoder_channels": 32,
    "line_af_decoder_channels": 32,
    "line_detection": {
        "do": True,
        "conf": {
            "max_point_size": 1500,
            "min_line_length": 60,
            "max_line_length": None,
            "samples": [8, 16, 32],

            "distance_map": {
                "max_value": 5,
                "threshold": 0.5,
                "smooth_threshold": 0.85,
                "avg_filter_size": 13,
                "avg_filter_padding": 6,
                "avg_filter_stride": 1,
                "inlier_ratio": 0.9,
                "max_accepted_mean_value": 0.3,
            },

            "mlp_conf": {
                "has_angle_field": True,
                "has_distance_field": True, 
                "num_bands": 3,
                "band_width": 1,
                "num_line_samples": 150,
                "mlp_hidden_dims": [256, 128, 128, 64, 32],
                "pred_threshold": 0.95,
                "weights": "/local/home/Point-Line/outputs/training/pold2_mlp_gen+train_Bands-3-1_1kimg_50pimg/checkpoint_best.tar",
            },
            "nms": True,
            "debug": True,
            "debug_dir": DEBUG_DIR,
        }
    },
    "checkpoint": "/local/home/Point-Line/outputs/training/focal_loss_experiments/rk_focal_threshDF_focal/checkpoint_best.tar"
    #"checkpoint": "/local/home/rkreft/shared_team_folder/outputs/training/rk_oxparis_focal_hard_gt/checkpoint_best.tar"
    #"checkpoint": "/local/home/rkreft/shared_team_folder/outputs/training/rk_pold2gt_oxparis_base_hard_gt/checkpoint_best.tar"
}

dset_conf = {
            "reshape": [400, 400], # ex [800, 800]
            "load_features": {
                "do": True,
                "check_exists": True,
                "point_gt": {
                    "data_keys": ["superpoint_heatmap"],
                    "use_score_heatmap": False,
                },
                "line_gt": {
                    "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
                },
            },
            "debug": True
        }

# Plotting functions
def show_points(image, points):
    for point in points:
        cv2.circle(image, (point[0], point[1]), 4, (191, 69, 17), -1)

    return image

def show_lines(image, lines, color='green'):
    if color == 'green':
        cval = (0, 255, 0)
    elif color == 'red':
        cval = (0, 0, 255)
    elif color == 'yellow':
        cval = (0, 255, 255)
    elif color == 'blue':
        cval = (255, 0, 0)
    else:
        cval = (0, 255, 0)
    for pair_line in lines:
        cv2.line(image, pair_line[0], pair_line[1], cval, 3)

    return image

def get_flow_vis(df, ang, line_neighborhood=5):
    norm = line_neighborhood + 1 - np.clip(df, 0, line_neighborhood)
    flow_uv = np.stack([norm * np.cos(ang), norm * np.sin(ang)], axis=-1)
    flow_img = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_img

def visualize_img_and_pred(keypoints, heatmap, distance_field, angle_field, img):
    _, ax = plt.subplots(1, 4, figsize=(20, 20))
    ax[0].axis('off')
    ax[0].set_title('Heatmap')
    ax[0].imshow(heatmap)

    ax[1].axis('off')
    ax[1].set_title('Distance Field')
    ax[1].imshow(distance_field)

    ax[2].axis('off')
    ax[2].set_title('Angle Field')
    ax[2].imshow(get_flow_vis(distance_field, angle_field))

    ax[3].axis('off')
    ax[3].set_title('Original')
    ax[3].imshow(img.permute(1,2,0))
    ax[3].scatter(keypoints[:,0],keypoints[:,1], marker="o", color="red", s=3)

# Set Output Directory
if os.path.exists(f"{DEBUG_DIR}"):
    os.system(f"rm -r {DEBUG_DIR}")
os.makedirs(f"{DEBUG_DIR}", exist_ok=True)


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

## Model
jpldd_model = get_model("joint_point_line_extractor")(jpldd_conf).to(device)
jpldd_model.eval()


## Dataset
oxpa_2 = get_dataset("oxford_paris_mini_1view_jpldd")(dset_conf)
ds = oxpa_2.get_dataset(split="train")

# load one test element
elem = ds[0]
print(f"Keys: {elem.keys()}")

# print example shapes
af = elem["deeplsd_angle_field"]
df = elem["deeplsd_distance_field"]
hmap = elem["superpoint_heatmap"]
orig_pt = elem["orig_points"]

print(f"AF: type: {type(af)}, shape: {af.shape}, min: {torch.min(af)}, max: {torch.max(af)}")
print(f"DF: type: {type(df)}, shape: {df.shape}, min: {torch.min(df)}, max: {torch.max(df)}")
print(f"KP-HMAP: type: {type(hmap)}, shape: {hmap.shape}, min: {torch.min(hmap)}, max: {torch.max(hmap)}, sum: {torch.sum(hmap)}")

## Inference - Random 300 samples [Get FPS]
# Comment while inspecting binary_distance_field
rand_idx = random.sample(range(0, len(ds)), 300) 
"""
for i in rand_idx:
    img_torch = ds[i]["image"].to(device).unsqueeze(0)
    with torch.no_grad():
        output_model = jpldd_model({"image": img_torch})

timings=jpldd_model.get_current_timings(reset=True)
pprint(timings)
print(f"~FPS: {1 / (timings['total-makespan'])} using device {device}")
"""
# Save images
IDX = 0
for i in tqdm(rand_idx):
    img_torch = ds[i]["image"].to(device).unsqueeze(0)
    with torch.no_grad():
        output_model = jpldd_model({"image": img_torch})

    # Save image with keypoints and lines
    img = (img_torch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    c_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    lines = output_model["lines"][0].cpu().numpy().astype(int)
    points = output_model["keypoints"][0].cpu().numpy().astype(int)

    img = c_img.copy()
    img = show_points(img, points)
    img = show_lines(img, lines)
    img = cv2.putText(img, "JPLDD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Calculate DeepLSD Lines
    with torch.no_grad():
        gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
        inputs = {
            "image": torch.tensor(gray_img, dtype=torch.float, device=device)[
                None, None
            ]
            / 255.0
        }
        deeplsd_output = deeplsd_net(inputs)
        deeplsd_lines = np.array(deeplsd_output["lines"][0]).astype(int)
        print(f"Num DeepLSD Lines: {len(deeplsd_lines)}")

    c_img = show_points(c_img, points)
    c_img = show_lines(c_img, deeplsd_lines, color='red')
    c_img = cv2.putText(c_img, "DeepLSD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Concatenate both images
    img = np.concatenate([img, c_img], axis=1)

    cv2.imwrite(f'{DEBUG_DIR}/{IDX}_lines.png', img)

    # Save the distance field and angle field
    keypoints = output_model["keypoints"][0].cpu().numpy()
    heatmap = output_model["keypoint_and_junction_score_map"][0].cpu().numpy()
    distance_field = output_model["line_distancefield"][0].cpu().numpy()#
    angle_field = output_model["line_anglefield"][0].cpu().numpy()

    visualize_img_and_pred(
        keypoints,
        heatmap,
        distance_field,
        angle_field,
        img_torch[0].cpu()
    )
    plt.savefig(f'{DEBUG_DIR}/{IDX}_fields.png')
    plt.close()

    IDX += 1