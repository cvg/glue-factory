"""
Testbed for Joint Point-Line Detection Model (JPLDD) with Line Detection.
Replication of notebooks/refine_line_detection.ipynb as a python script instead of a notebook.
Usage:
    python -m notebooks.jpldd_linedet_testbed.py
"""


from gluefactory.models import get_model
from gluefactory.datasets import get_dataset
from gluefactory.models.deeplsd_inference import DeepLSD
from gluefactory.models.lines.pold2_extractor import LineExtractor
from gluefactory.settings import DATA_PATH
from gluefactory.visualization.viz2d import show_distance_field
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
    "max_num_keypoints": 1000,  # setting for training, for eval: -1
    "timeit": True,  # override timeit: False from BaseModel
    "line_df_decoder_channels": 32,
    "line_af_decoder_channels": 32,
    "line_detection": {
        "do": True,
        "conf": {
            "max_point_size": 1500,
            "min_line_length": 6,
            "max_line_length": None,
            "samples": [24],

            "distance_map": {
                "max_value": 5,
                "threshold": 0.45,
                "smooth_threshold": 0.8,
                "avg_filter_size": 13,
                "avg_filter_padding": 6,
                "avg_filter_stride": 1,
                "inlier_ratio": 0.7,
                "max_accepted_mean_value": 0.5,
            },

            "brute_force_df": {
                "use": True,                       # Use brute force sampling for distance field in the second stage
                "image_size": 800,                  # Image size for which the coefficients are generated
                "inlier_ratio": 0.8,                # Ratio of inliers
                "max_accepted_mean_value": 0.3,     # Maximum accepted DF mean value along the line
            },

            "angle_map": {
                "threshold": 0.1,                   # Threshold for deciding if a line angle is correct
                "inlier_ratio": 1.0,                # Ratio of inliers
                "max_accepted_mean_value": 0.1,     # Maximum difference in AF mean value with line angle
            },

            "mlp_conf": {
                "has_angle_field": True,
                "has_distance_field": True, 
                
                "num_line_samples": 30,    # number of sampled points between line endpoints
                "brute_force_samples": True,  # sample all points between line endpoints
                "image_size": 800,         # size of the input image, relevant only if brute_force_samples is True

                "num_bands": 5,            # number of bands to sample along the line
                "band_width": 1,           # width of the band to sample along the line

                "mlp_hidden_dims": [512, 256, 128, 64, 32], # hidden dimensions of the MLP

                "cnn_1d": {              # 1D CNN to extract features from the input
                    "mode": "shared",  # separate CNNs for angle and distance fields, disjoint or shared
                    "merge_mode": "concat",  # how to merge the features from angle and distance fields
                    "kernel_size": 11,
                    "stride": 1,
                    "padding": "same",
                    "channels": [16, 8, 4, 1],  # number of channels in each layer
                },           

                "pred_threshold": 0.8,            
                "weights": None,
            },

            "filters": {
                "distance_field": True,
                "angle_field": False,
                "brute_force_df": True,
                "mlp": False,
            },

            "nms": False,
            "debug": True,
            "debug_dir": DEBUG_DIR,
            # "device": "cpu"
        }
    },
    "checkpoint": "/local/home/Point-Line/outputs/training/oxparis_800_focal/checkpoint_best.tar"
}

dset_conf = {
    "reshape": 800,  # ex. 800
    "multiscale_learning": {
        "do": False,
        "scales_list": [800, 600, 400],
        "scale_selection": 'round-robin' # random or round-robin
    },
    "load_features": {
        "do": True,
        "check_exists": True,
        "point_gt": {
            "data_keys": ["superpoint_heatmap", "gt_keypoints", "gt_keypoints_scores"],
            "use_score_heatmap": True,
        },
        "line_gt": {
            "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
            "enforce_threshold": 5.0,  # Enforce values in distance field to be no greater than this value
        },
    },
    #"debug": True
}

harris_conf = {
    "blockSize": 5,     # neighborhood size
    "ksize": 5,         # aperture parameter for the Sobel operator
    "k": 0.04,          # Harris detector free parameter
    "thresh": 0.01,     # threshold for corner detection on harris corner response
    "zeroZone": -1,     # 0 means that no extra zero pixels are used
    "winSize": 5,       # window size for cornerSubPix
    "criteria": {       # termination criteria for cornerSubPix
        "type": "cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER",
        "maxCount": 100,
        "epsilon": 0.001
    }
}
harris_conf = OmegaConf.create(harris_conf)

visualizations = {
    "jpldd_fields": False,
    "jpldd_lines": True,
    "deeplsd": True,
    "pold2+deeplsd": False,
    "pold2+harris": False,
    "jpldd+lsd": False
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

def detect_harris_corners(image, conf):
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, conf.blockSize, conf.ksize, conf.k)
    dst = cv2.dilate(dst, None)
    _, dst = cv2.threshold(dst, conf.thresh * dst.max(), 255, 0)
    dst = np.uint8(dst)
    _, _, _, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (eval(conf.criteria.type), conf.criteria.maxCount, conf.criteria.epsilon)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (conf.winSize, conf.winSize), (conf.zeroZone, conf.zeroZone), criteria)

    return corners

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

## Line Extraction
line_extractor = LineExtractor(jpldd_conf["line_detection"]["conf"])

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

print(f"AF: type: {type(af)}, shape: {af.shape}, min: {torch.min(af)}, max: {torch.max(af)}")
print(f"DF: type: {type(df)}, shape: {df.shape}, min: {torch.min(df)}, max: {torch.max(df)}")
print(f"KP-HMAP: type: {type(hmap)}, shape: {hmap.shape}, min: {torch.min(hmap)}, max: {torch.max(hmap)}, sum: {torch.sum(hmap)}")

## Inference - Random 300 samples [Get FPS]
# Comment while inspecting binary_distance_field
random.seed(42)
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
    gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

    lines = output_model["lines"][0].cpu().numpy().astype(int)
    points = output_model["keypoints"][0].cpu().numpy().astype(int)

    viz_img = np.zeros((c_img.shape[0]+50, 0, 3), dtype=np.uint8)

    if visualizations["jpldd_lines"]:
        img = c_img.copy()
        img = show_points(img, points)
        img = show_lines(img, lines)
        img = cv2.copyMakeBorder(img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        img = cv2.putText(img, "Pold2 + JPLDD Points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        viz_img = np.concatenate([viz_img, img], axis=1)

        df = output_model["line_distancefield"][0].cpu().numpy()
        df = (df / np.max(df) * 255).astype(np.uint8)
        df = cv2.applyColorMap(df, cv2.COLORMAP_JET)
        df = cv2.copyMakeBorder(df, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        df = cv2.putText(df, "JPLDD Distance Field", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        viz_img = np.concatenate([df, viz_img], axis=1)
        print("--------------JPLDD OVER--------------")

    # Calculate DeepLSD Lines
    if visualizations["deeplsd"]:
        with torch.no_grad():
            inputs = {
                "image": torch.tensor(gray_img, dtype=torch.float, device=device)[
                    None, None
                ]
                / 255.0
            }
            deeplsd_output = deeplsd_net(inputs)
            deeplsd_lines = np.array(deeplsd_output["lines"][0]).astype(int)
            print(f"Num DeepLSD Lines: {len(deeplsd_lines)}")

        dpoints = deeplsd_lines.reshape(-1, 2).astype(int)
        dpoints[:, 0] = np.clip(dpoints[:, 0], 0, c_img.shape[1] - 1)
        dpoints[:, 1] = np.clip(dpoints[:, 1], 0, c_img.shape[0] - 1)

        d_img = c_img.copy()
        d_img = show_points(d_img, points)
        d_img = show_lines(d_img, deeplsd_lines, color='red')
        d_img = cv2.copyMakeBorder(d_img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        d_img = cv2.putText(d_img, "DeepLSD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        viz_img = np.concatenate([viz_img, d_img], axis=1)
        print("--------------DEELSD OVER--------------")

    # Use Pold2 Line Extractor on DeepLSD endpoints
    if visualizations["pold2+deeplsd"]:
        line_extractor_input = {
            "points": torch.from_numpy(dpoints).float().to(device),
            "distance_map": output_model["line_distancefield"][0].clone(),
            "angle_map": output_model["line_anglefield"][0].clone(),
            "descriptors": torch.zeros(dpoints.shape[0], 128).to(device),
        }
        pold2_lines = line_extractor(line_extractor_input)["lines"].cpu()
        pold2_lines = np.array(pold2_lines).astype(int)

        p_img = c_img.copy()
        p_img = show_points(p_img, dpoints)
        p_img = show_lines(p_img, pold2_lines)
        p_img = cv2.copyMakeBorder(p_img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        p_img = cv2.putText(p_img, "Pold2 + DeepLSD Points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        viz_img = np.concatenate([viz_img, p_img], axis=1)
        print("--------------POLD2+DEELSD OVER--------------")

    # Use Pold2 Line Extractor on HARRIS corners
    if visualizations["pold2+harris"]:
        harris_points = detect_harris_corners(gray_img, harris_conf)
        hpoints = torch.from_numpy(harris_points).float().to(device)
        line_extractor_input = {
            # "points": torch.cat(
            #     [hpoints, torch.from_numpy(dpoints).float().to(device)], dim=0
            # ),
            "points": torch.cat(
                [hpoints, torch.from_numpy(points).float().to(device)], dim=0
            ),
            # "points": hpoints,
            "distance_map": output_model["line_distancefield"][0].clone(),
            "angle_map": output_model["line_anglefield"][0].clone(),
            # "descriptors": torch.zeros(hpoints.shape[0] + dpoints.shape[0], 128).to(device),
            "descriptors": torch.zeros(hpoints.shape[0] + points.shape[0], 128).to(device),
            # "descriptors": torch.zeros(hpoints.shape[0], 128).to(device),
        }
        pold2_lines = line_extractor(line_extractor_input)["lines"].cpu()
        pold2_lines = np.array(pold2_lines).astype(int)
        print(f"Num Harris Lines: {len(pold2_lines)}")

        h_img = c_img.copy()
        h_img = show_points(h_img, harris_points.astype(int))
        h_img = show_lines(h_img, pold2_lines)
        h_img = cv2.copyMakeBorder(h_img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        h_img = cv2.putText(h_img, "Pold2 + Harris + JPLDD Points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        viz_img = np.concatenate([viz_img, h_img], axis=1)
        print("--------------POLD2+HARRIS OVER--------------")

    # Use LSD on top of JPLDD DF and AF
    if visualizations["jpldd+lsd"]:
        lsd_lines = []
        np_img = (inputs["image"].cpu().numpy()[:, 0] * 255).astype(np.uint8)
        np_df = output_model["line_distancefield"].cpu().numpy()
        np_ll = output_model["line_anglefield"].cpu().numpy()
        for im, df, ll in zip(np_img, np_df, np_ll):
            line = deeplsd_net.detect_afm_lines(
                im, df, ll, **deeplsd_conf.line_detection_params
            )
            lsd_lines.append(line.astype(int))
        l_img = c_img.copy()
        l_img = show_points(l_img, points)
        l_img = show_lines(l_img, lsd_lines[0], color='yellow')
        l_img = cv2.copyMakeBorder(l_img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        l_img = cv2.putText(l_img, "JPLDD + LSD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        viz_img = np.concatenate([viz_img, l_img], axis=1)
        print("--------------JPLDD+LSD OVER--------------")

    cv2.imwrite(f'{DEBUG_DIR}/{IDX}_lines.png', viz_img)

    # Save the distance field and angle field
    if visualizations["jpldd_fields"]:
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