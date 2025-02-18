"""
MLP classifier for candidate line endpoints using DF and AF values sampled along the line.
Use the following command to train the MLP:
    python -m gluefactory.train pold2_mlp_test --conf gluefactory/configs/pold2_mlp_train.yaml
Use the following command to plot the confusion matrix):
    python -m gluefactory.models.lines.pold2_mlp \
        --conf gluefactory/configs/pold2_mlp_train.yaml \
        --weights outputs/training/pold2_mlp_test/checkpoint_best.tar
Use the following command to test the dataloader:
    python -m gluefactory.datasets.pold2_mlp_dataset --conf gluefactory/configs/pold2_mlp_dataloader_test.yaml
"""

import argparse
import enum
import logging

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from gluefactory.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class CNNMode(enum.Enum):
    DISJOINT = "disjoint"
    SHARED = "shared"


class MERGE_MODE(enum.Enum):
    CONCAT = "concat"
    ADD = "add"


class POLD2_MLP(BaseModel):

    default_conf = {
        "name": "lines.pold2_mlp",
        "has_angle_field": True,
        "has_distance_field": True,
        "num_line_samples": 30,  # number of sampled points between line endpoints
        "brute_force_samples": False,  # sample all points between line endpoints
        "image_size": 800,  # size of the input image, relevant only if brute_force_samples is True
        "num_bands": 1,  # number of bands to sample along the line
        "band_width": 1,  # width of the band to sample along the line
        "mlp_hidden_dims": [256, 128, 128, 64, 32],  # hidden dimensions of the MLP
        "cnn_1d": {  # 1D CNN to extract features from the input
            "use": True,
            "mode": "disjoint",  # separate CNNs for angle and distance fields, disjoint or shared
            "merge_mode": "concat",  # how to merge the features from angle and distance fields
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
            "channels": [3, 3, 1],  # number of channels in each layer
        },
        "cnn_2d": {  # 2D CNN to extract features from the input
            "use": False,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
            "channels": [4, 8, 16],  # number of channels in each layer
        },
        "pred_threshold": 0.9,
        "weights": None,
        "device": None,
    }

    def _init(self, conf):
        if conf.device is not None:
            self.device = conf.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        input_dim = 0
        self.num_line_samples = conf.num_line_samples
        if conf.brute_force_samples:
            assert (
                conf.image_size is not None
            ), "image_size must be provided for brute force sampling"
            self.num_line_samples = np.sqrt(2 * conf.image_size**2).astype(int)
        if conf.has_angle_field:
            input_dim += self.num_line_samples
        if conf.has_distance_field:
            input_dim += self.num_line_samples
        input_dim *= conf.num_bands
        if input_dim == 0:
            raise ValueError(
                "No input features selected for MLP, please set has_angle_field or has_distance_field to True"
            )

        self.cnn1d_active = conf.cnn_1d is not None and conf.cnn_1d["use"]
        self.cnn2d_active = conf.cnn_2d is not None and conf.cnn_2d["use"]
        assert not (
            self.cnn1d_active and self.cnn2d_active
        ), "Only one of 1D and 2D CNNs can be active"

        # 1D CNN layers
        if self.cnn1d_active:
            assert conf.cnn_1d["mode"] in [m.value for m in CNNMode]
            assert conf.cnn_1d["merge_mode"] in [m.value for m in MERGE_MODE]
            if conf.cnn_1d["mode"] == CNNMode.DISJOINT.value:
                assert conf.has_angle_field and conf.has_distance_field

            cnn_channels = [c for c in conf.cnn_1d["channels"]]
            if conf.cnn_1d["mode"] == CNNMode.DISJOINT.value:

                af_cnn = []
                in_c = conf.num_bands
                self.conv_channels = [in_c] + cnn_channels

                for i in range(len(self.conv_channels) - 1):
                    af_cnn.append(
                        nn.Conv1d(
                            in_channels=self.conv_channels[i],
                            out_channels=self.conv_channels[i + 1],
                            kernel_size=conf.cnn_1d["kernel_size"],
                            stride=conf.cnn_1d["stride"],
                            padding=conf.cnn_1d["padding"],
                        )
                    )
                    af_cnn.append(nn.ReLU())
                af_cnn.append(nn.Flatten())

                self.af_cnn = nn.Sequential(*af_cnn)
                self.af_cnn.to(self.device)

                df_cnn = []
                for i in range(len(self.conv_channels) - 1):
                    df_cnn.append(
                        nn.Conv1d(
                            in_channels=self.conv_channels[i],
                            out_channels=self.conv_channels[i + 1],
                            kernel_size=conf.cnn_1d["kernel_size"],
                            stride=conf.cnn_1d["stride"],
                            padding=conf.cnn_1d["padding"],
                        )
                    )
                    df_cnn.append(nn.ReLU())
                df_cnn.append(nn.Flatten())

                self.df_cnn = nn.Sequential(*df_cnn)
                self.df_cnn.to(self.device)

                input_dim = self.conv_channels[-1] * self.num_line_samples
                input_dim *= 2 if conf.cnn_1d["merge_mode"] == "concat" else 1

            elif conf.cnn_1d["mode"] == CNNMode.SHARED.value:
                cnn_layers = []
                in_c = 0
                if conf.has_angle_field:
                    in_c += conf.num_bands
                if conf.has_distance_field:
                    in_c += conf.num_bands
                self.conv_channels = [in_c] + cnn_channels

                for i in range(len(self.conv_channels) - 1):
                    cnn_layers.append(
                        nn.Conv1d(
                            in_channels=self.conv_channels[i],
                            out_channels=self.conv_channels[i + 1],
                            kernel_size=conf.cnn_1d["kernel_size"],
                            stride=conf.cnn_1d["stride"],
                            padding=conf.cnn_1d["padding"],
                        )
                    )
                    cnn_layers.append(nn.ReLU())
                cnn_layers.append(nn.Flatten())

                self.cnn = nn.Sequential(*cnn_layers)
                self.cnn.to(self.device)
                input_dim = self.conv_channels[-1] * self.num_line_samples

        # 2D CNN layers
        if self.cnn2d_active:
            cnn_layers = []
            in_c = 0
            if conf.has_angle_field:
                in_c += conf.num_bands
            if conf.has_distance_field:
                in_c += conf.num_bands

            cnn_channels = [c for c in conf.cnn_2d["channels"]]
            self.conv_channels = [1] + cnn_channels

            for i in range(len(self.conv_channels) - 1):
                cnn_layers.append(
                    nn.Conv2d(
                        in_channels=self.conv_channels[i],
                        out_channels=self.conv_channels[i + 1],
                        kernel_size=conf.cnn_2d["kernel_size"],
                        stride=conf.cnn_2d["stride"],
                        padding=conf.cnn_2d["padding"],
                    )
                )
                cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Flatten())

            self.cnn = nn.Sequential(*cnn_layers)
            self.cnn.to(self.device)
            input_dim *= self.conv_channels[-1]

        # MLP layers
        mlp_layers = []
        mlp_layers.append(nn.Linear(input_dim, conf.mlp_hidden_dims[0]))
        mlp_layers.append(nn.ReLU())
        for i in range(1, len(conf.mlp_hidden_dims)):
            mlp_layers.append(
                nn.Linear(conf.mlp_hidden_dims[i - 1], conf.mlp_hidden_dims[i])
            )
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(conf.mlp_hidden_dims[-1], 1))
        self.mlp = nn.Sequential(*mlp_layers)

        if self.conf.weights is not None:
            ckpt = torch.load(
                str(self.conf.weights), map_location=self.device, weights_only=True
            )
            self.load_state_dict(ckpt["model"], strict=True)
            logger.info(f"Successfully loaded model weights from {self.conf.weights}")
        self.mlp.to(self.device)

        self.set_initialized()

    def _forward(self, data):
        x = data["input"]  # shape: (num_lines, num_samples * num_bands * (2 or 1))

        if not self.cnn1d_active and not self.cnn2d_active:
            return {"line_probs": torch.sigmoid(self.mlp(x))}

        if self.cnn1d_active:
            if self.conf.cnn_1d["mode"] == CNNMode.DISJOINT.value:
                # CNN for distance field
                df = x[:, : self.num_line_samples * self.conf.num_bands]
                df = df.view(-1, self.conf.num_bands, self.num_line_samples)
                df = self.df_cnn(df)

                # CNN for angle field
                af = x[:, self.num_line_samples * self.conf.num_bands :]
                af = af.view(-1, self.conf.num_bands, self.num_line_samples)
                af = self.af_cnn(af)

                # Merge the features
                if self.conf.cnn_1d["merge_mode"] == MERGE_MODE.CONCAT.value:
                    x = torch.cat([df, af], dim=1)
                elif self.conf.cnn_1d["merge_mode"] == MERGE_MODE.ADD.value:
                    x = df + af

            elif self.conf.cnn_1d["mode"] == CNNMode.SHARED.value:
                in_c = self.conv_channels[0]
                x = x.view(-1, in_c, self.num_line_samples)
                x = self.cnn(x)

        elif self.cnn2d_active:
            if self.conf.has_angle_field and self.conf.has_distance_field:
                df = x[:, : self.num_line_samples * self.conf.num_bands].view(
                    -1, 1, self.conf.num_bands, self.num_line_samples
                )
                af = x[:, self.num_line_samples * self.conf.num_bands :].view(
                    -1, 1, self.conf.num_bands, self.num_line_samples
                )
                x = torch.cat([df, af], dim=3)

            elif self.conf.has_angle_field or self.conf.has_distance_field:
                x = x.view(-1, 1, self.conf.num_bands, self.num_line_samples)

            x = self.cnn(x)

        logits = self.mlp(x)
        return {
            "line_probs": torch.sigmoid(logits),
            "logits": logits,
        }

    def loss(self, pred, data):

        losses = {}

        # Compute the loss (BCE loss between predictions and labels)
        labels = data["label"]
        x_pred = pred["line_probs"]

        loss = nn.BCELoss()(x_pred.reshape(-1), labels.float())
        losses["total"] = loss.unsqueeze(0)

        metrics = self.metrics(pred, data)
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                metrics[k] = v.unsqueeze(0)

        return losses, metrics

    def metrics(self, pred, data, eps=1e-7):
        labels = data["label"].flatten()
        x_pred = pred["line_probs"].flatten()

        device = labels.device

        x_pred_th = (x_pred > self.conf.pred_threshold).float()

        tp = (x_pred_th * labels).sum().item()
        tn = ((1 - x_pred_th) * (1 - labels)).sum().item()
        fp = (x_pred_th * (1 - labels)).sum().item()
        fn = ((1 - x_pred_th) * labels).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)

        return {
            "accuracy": torch.tensor(accuracy, dtype=torch.float, device=device),
            "precision": torch.tensor(precision, dtype=torch.float, device=device),
            "recall": torch.tensor(recall, dtype=torch.float, device=device),
            "f1": torch.tensor(f1, dtype=torch.float, device=device),
        }


__main_model__ = POLD2_MLP

# Run the model and plot the confusion matrix
if __name__ == "__main__":
    from ... import logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    conf = (
        OmegaConf.load(args.conf) if args.conf is not None else POLD2_MLP.default_conf
    )
    conf.weights = args.weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = POLD2_MLP(conf).to(device)
    model.eval()

    # Load the data
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import metrics

    from gluefactory.datasets.pold2_mlp_dataset import POLD2_MLP_Dataset

    dataset = POLD2_MLP_Dataset(conf.data)
    dataloader = dataset.get_data_loader("val")

    actual = []
    predicted = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == len(dataloader) - 1:
            break

        y = batch["label"]
        batch["input"] = batch["input"].to(device)
        batch["label"] = batch["label"].to(device)

        with torch.no_grad():
            x_pred = model(batch)
            x_pred = x_pred["line_probs"]

            actual.append(y.cpu().numpy())
            predicted.append(x_pred.cpu().numpy())

    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    predicted = predicted > conf.model.pred_threshold

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=[0, 1]
    )

    cm_display.plot()
    output_path = Path(args.weights).parent / "confusion_matrix.png"
    plt.savefig(output_path)
    logger.info(f"Confusion matrix saved to {output_path}")
