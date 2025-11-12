"""This module implements the writer class for logging to tensorboard or wandb."""

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TFSummaryWriter

from .. import __module_name__, settings
from ..utils import misc

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    logger.debug("Could not import wandb.")
    wandb = None


def latest_wandb_run_id(log_dir: Path) -> str | None:
    """Get the latest wandb run id from a log directory."""
    if not (log_dir / "wandb").exists():
        return None
    file = list((log_dir / "wandb" / "latest-run").glob("run-*.wandb"))
    if not file:
        return None
    if len(file) > 1:
        raise ValueError(f"Multiple ({len(file)}) latest runs found in {log_dir}.")
    return file[0].stem.replace("run-", "")


class SummaryWriter:
    """Writer class for logging to tensorboard or wandb."""

    def __init__(
        self,
        log_dir: Path,
        conf: DictConfig | None = None,
        writer: str | Sequence = "tensorboard",
        name: str | None = None,
        project: str = __module_name__,
        run_id: str | None = None,
        name_as_run_id: bool = True,
        reload_run_id: bool = True,
        **wandb_kwargs,
    ):
        """Initialize the writer."""
        self.log_dir = log_dir

        if not writer:
            self.use_wandb = False
            self.use_tensorboard = False
            return

        self.use_wandb = "wandb" in writer
        self.use_tensorboard = "tensorboard" in writer

        if self.use_wandb and not wandb:
            raise ImportError("wandb not installed.")

        if self.use_tensorboard:
            self.writer = TFSummaryWriter(log_dir=log_dir)

        if self.use_wandb:
            os.environ["WANDB__SERVICE_WAIT"] = "300"
            name = (
                name
                if name is not None
                else str(log_dir.relative_to(settings.TRAINING_PATH))
            )
            if conf is not None:
                wandb_conf = OmegaConf.to_container(conf, resolve=True)
                wandb_conf = misc.flatten_dict(wandb_conf)
            else:
                wandb_conf = None
            if name_as_run_id and run_id is None:
                run_id = name.replace("/", "_")
            elif reload_run_id and run_id is None:
                run_id = latest_wandb_run_id(log_dir)
                if run_id is not None:
                    logger.info(f"Reloaded wandb run {run_id} from {log_dir}.")
                    wandb_kwargs["resume"] = "must"  # always resume if run_id is given
            wandb.init(
                project=project,
                name=name,
                dir=log_dir,
                config=wandb_conf,
                tags=name.split("/"),
                id=run_id,
                **wandb_kwargs,
            )

    def add_scalar(self, tag: str, value: float, step: int | None = None):
        """Log a scalar value to tensorboard or wandb."""
        if self.use_wandb:
            step = 1 if step == 0 else step
            wandb.log({tag: value}, step=step)

        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)

    def add_figure(self, tag: str, figure: Any, step: int | None = None):
        """Log a figure to tensorboard or wandb."""
        if self.use_wandb:
            step = 1 if step == 0 else step
            wandb.log({tag: wandb.Image(figure)}, step=step)
        if self.use_tensorboard:
            self.writer.add_figure(tag, figure, step, close=True)

    def add_histogram(self, tag: str, values, step: int | None = None):
        """Log a histogram to tensorboard or wandb."""
        if self.use_tensorboard:
            self.writer.add_histogram(tag, values, step)

    def add_mesh(
        self,
        tag: str,
        vertices,
        faces=None,
        colors=None,
        step: int | None = None,
        point_size: float | None = None,
        config: dict | None = None,
    ):
        """Log a 3D mesh to tensorboard or wandb."""
        if self.use_wandb:
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            wandb.log(
                {tag: wandb.Object3D.from_numpy(vertices)},
                step=step,
            )
        if self.use_tensorboard:
            config = config if config is not None else {}
            if point_size is not None:
                config["material"] = {"cls": "PointsMaterial", "size": point_size}

            def abd(x):
                if x is None:
                    return None
                else:
                    return x if x.ndim == 3 else x.unsqueeze(0)

            self.writer.add_mesh(
                tag,
                abd(vertices),
                faces=abd(faces),
                colors=abd(colors),
                global_step=step,
                config_dict=config,
            )

    def add_text(self, tag: str, text: str, step: int | None = None):
        """Log text to tensorboard or wandb."""
        if self.use_tensorboard:
            self.writer.add_text(tag, text, step)

    def add_pr_curve(self, tag: str, *labels_preds, step: int | None = None):
        """Log a precision-recall curve to tensorboard or wandb."""
        if self.use_wandb:
            # The wandb PR curve interface is a bit clunky, so we compute the metrics ourselves
            step = 1 if step == 0 else step
            # @TODO: check if this works
            labels, preds = labels_preds
            try:
                precision, recall, _ = precision_recall_curve(labels, preds)
            except ValueError as e:
                logger.warning(f"Could not compute {tag}/pr_curve at step {step}.")
                return
            f1score = 2 * (precision * recall) / (precision + recall + 1e-8)
            best = np.argmax(f1score)
            self.add_scalar(f"{tag}/f1", f1score[best], step)
            self.add_scalar(f"{tag}/precision", precision[best], step)
            self.add_scalar(f"{tag}/recall", recall[best], step)
            auprc = auc(recall, precision)
            self.add_scalar(f"{tag}/auprc", auprc, step)
            try:
                auroc = roc_auc_score(labels, preds)
                self.add_scalar(f"{tag}/auroc", auroc, step)
            except ValueError as e:
                logger.warning(f"Could not compute {tag}/auroc at step {step}.")

            ap = average_precision_score(labels, preds)
            self.add_scalar(f"{tag}/ap", ap, step)

        if self.use_tensorboard:
            try:
                self.writer.add_pr_curve(tag, *labels_preds, global_step=step)
            except Exception as e:
                logger.warning(f"Could not compute {tag}/pr_curve at step {step}.")
                return

    def define_metric(self, name: str, summary: str | None = None):
        """Define a custom metric for wandb."""
        if self.use_wandb:
            wandb.define_metric(name, summary=summary)

    def watch(self, model: nn.Module, log_freq: int = 1000):
        """Watch a model for gradient updates."""
        if self.use_wandb:
            wandb.watch(
                model,
                log="gradients",
                log_freq=log_freq,
            )

    def close(self):
        """Close the writer."""
        if self.use_wandb:
            wandb.finish()

        if self.use_tensorboard:
            self.writer.close()


if __name__ == "__main__":

    path = Path("outputs/training/megadepth/sp+lg/xyz/xyz_rmse/L5/0")
