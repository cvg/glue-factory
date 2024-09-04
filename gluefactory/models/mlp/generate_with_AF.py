import glob
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gluefactory.models.deeplsd_inference import DeepLSD
from gluefactory.settings import DATA_PATH

# Deep LSD Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = {
    "detect_lines": True,  # Whether to detect lines or only DF/AF
    "line_detection_params": {
        "merge": False,  # Whether to merge close-by lines
        "filtering": True,
        # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
        "grad_thresh": 3,
        "grad_nfa": True,
        # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
    },
}

# Load the model
ckpt = DATA_PATH / "DeepLSD/weights/deeplsd_md.tar"
ckpt = torch.load(str(ckpt), map_location=device)
net = DeepLSD(conf)
net.load_state_dict(ckpt["model"])
net = net.to(device).eval()


def get_line_from_image(path):
    img = cv2.imread(path)[:, :, ::-1]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    inputs = {
        "image": torch.tensor(gray_img, dtype=torch.float, device=device)[None, None]
        / 255.0
    }

    with torch.no_grad():
        out = net(inputs)

    # distance field
    distances = out["df"][0]
    distances /= net.conf.line_neighborhood
    distances = distances.cpu().numpy()

    # angle field
    angles = out["line_level"][0]
    angles = angles.cpu().numpy() / np.pi

    lines = np.array(out["lines"][0])

    return distances, angles, lines


def datasetEntryFromPoints(p1, p2, distance_map, angle_map, blend):
    v1 = blend * p1
    v2 = (1 - blend) * p2

    points = (v1 + v2).round().astype(int)

    points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)
    points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)

    df_val = distance_map[points[:, 1], points[:, 0]].reshape(-1)
    af_val = angle_map[points[:, 1], points[:, 0]].reshape(-1)

    return np.hstack([df_val, af_val])

num_train_images = 150
num_val_images = 25
num_interp = 150
blend = np.linspace(0, 1, num_interp).reshape(-1, 1)

positives = []
negatives = []

positives_test = []
negatives_test = []
cnt = 0

for file_path in glob.glob(str(DATA_PATH / "revisitop1m/jpg/**/base_image.jpeg"), recursive=True):
    print(f"{cnt} == {file_path}")

    # Load base image
    image = np.array(Image.open(file_path))

    # Load lines - inference from deep lsd
    distance_map, angle_map, lines = get_line_from_image(file_path)

    # Set lines as integer for interpolation
    lines = lines.astype(int)

    for line in lines:
        # Add positives lines
        if cnt < num_train_images:
            positives.append(datasetEntryFromPoints(line[0], line[1], distance_map, angle_map, blend))
        else:
            positives_test.append(datasetEntryFromPoints(line[0], line[1], distance_map, angle_map, blend))

        for i in range(1):
            # Now we can random sample the image
            p1 = np.array(
                [random.randint(0, image.shape[1]), random.randint(0, image.shape[0])]
            )
            p2 = np.array(
                [random.randint(0, image.shape[1]), random.randint(0, image.shape[0])]
            )

            # Add negative lines
            if cnt < num_train_images:
                negatives.append(datasetEntryFromPoints(p1, p2, distance_map, angle_map, blend))
            else:
                negatives_test.append(datasetEntryFromPoints(p1, p2, distance_map, angle_map, blend))

    cnt += 1

    if cnt == num_train_images + num_val_images:
        break

np.save(DATA_PATH / "mlp_data/positives.npy", positives)
np.save(DATA_PATH / "mlp_data/negatives.npy", negatives)

np.save(DATA_PATH / "mlp_data/positives_test.npy", positives_test)
np.save(DATA_PATH / "mlp_data/negatives_test.npy", negatives_test)
