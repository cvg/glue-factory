"""
A set of utilities to manage and load checkpoints of training experiments.

Author: Paul-Edouard Sarlin (skydes)
"""

import logging
import os
import re
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models import get_model
from ..settings import TRAINING_PATH

logger = logging.getLogger(__name__)


def list_checkpoints(dir_):
    """List all valid checkpoints in a given directory."""
    checkpoints = []
    for p in dir_.glob("checkpoint_*.tar"):
        numbers = re.findall(r"(\d+)", p.name)
        assert len(numbers) <= 2
        if len(numbers) == 0:
            continue
        if len(numbers) == 1:
            checkpoints.append((int(numbers[0]), p))
        else:
            checkpoints.append((int(numbers[1]), p))
    return checkpoints


def get_last_checkpoint(exper, allow_interrupted=True):
    """Get the last saved checkpoint for a given experiment name."""
    ckpts = list_checkpoints(Path(TRAINING_PATH, exper))
    if not allow_interrupted:
        ckpts = [(n, p) for (n, p) in ckpts if "_interrupted" not in p.name]
    assert len(ckpts) > 0
    return sorted(ckpts)[-1][1]


def get_best_checkpoint(exper):
    """Get the checkpoint with the best loss, for a given experiment name."""
    p = Path(TRAINING_PATH, exper, "checkpoint_best.tar")
    return p


def delete_old_checkpoints(dir_, num_keep):
    """Delete all but the num_keep last saved checkpoints."""
    ckpts = list_checkpoints(dir_)
    ckpts = sorted(ckpts)[::-1]
    kept = 0
    for ckpt in ckpts:
        if ("_interrupted" in str(ckpt[1]) and kept > 0) or kept >= num_keep:
            logger.info(f"Deleting checkpoint {ckpt[1].name}")
            ckpt[1].unlink()
        else:
            kept += 1


def load_experiment(exper, conf={}, get_last=False, ckpt=None):
    """Load and return the model of a given experiment."""
    exper = Path(exper)
    if exper.suffix != ".tar":
        if get_last:
            ckpt = get_last_checkpoint(exper)
        else:
            ckpt = get_best_checkpoint(exper)
    else:
        ckpt = exper
    logger.info(f"Loading checkpoint {ckpt.name}")
    ckpt = torch.load(str(ckpt), map_location="cpu")

    loaded_conf = OmegaConf.create(ckpt["conf"])
    OmegaConf.set_struct(loaded_conf, False)
    conf = OmegaConf.merge(loaded_conf.model, OmegaConf.create(conf))
    model = get_model(conf.name)(conf).eval()

    state_dict = ckpt["model"]
    dict_params = set(state_dict.keys())
    model_params = set(map(lambda n: n[0], model.named_parameters()))
    diff = model_params - dict_params
    if len(diff) > 0:
        subs = os.path.commonprefix(list(diff)).rstrip(".")
        logger.warning(f"Missing {len(diff)} parameters in {subs}")
    model.load_state_dict(state_dict, strict=False)
    return model


# @TODO: also copy the respective module scripts (i.e. the code)
def save_experiment(
    model,
    optimizer,
    lr_scheduler,
    conf,
    losses,
    results,
    best_eval,
    epoch,
    iter_i,
    output_dir,
    stop=False,
    distributed=False,
    cp_name=None,
):
    """Save the current model to a checkpoint
    and return the best result so far."""
    state = (model.module if distributed else model).state_dict()
    checkpoint = {
        "model": state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "conf": OmegaConf.to_container(conf, resolve=True),
        "epoch": epoch,
        "losses": losses,
        "eval": results,
    }
    if cp_name is None:
        cp_name = (
            f"checkpoint_{epoch}_{iter_i}" + ("_interrupted" if stop else "") + ".tar"
        )
    logger.info(f"Saving checkpoint {cp_name}")
    cp_path = str(output_dir / cp_name)
    torch.save(checkpoint, cp_path)
    if cp_name != "checkpoint_best.tar" and results[conf.train.best_key] < best_eval:
        best_eval = results[conf.train.best_key]
        logger.info(f"New best val: {conf.train.best_key}={best_eval}")
        shutil.copy(cp_path, str(output_dir / "checkpoint_best.tar"))
    delete_old_checkpoints(output_dir, conf.train.keep_last_checkpoints)
    return best_eval
