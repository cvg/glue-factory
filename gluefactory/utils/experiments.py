"""
A set of utilities to manage and load checkpoints of training experiments.

Author: Paul-Edouard Sarlin (skydes)
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import hydra
import pkg_resources
import torch
from omegaconf import OmegaConf

from .. import settings
from ..models import get_model

logger = logging.getLogger(__name__)


def list_configs(configs_path: Path) -> list[str]:
    """List all available configs in a given directory."""
    return list(sorted([x.stem for x in Path(configs_path).glob("*.yaml")]))


def parse_config_path(
    name_or_path: Optional[str], default_config_dir: str = "configs/"
) -> Path:
    default_configs = {}
    for c in pkg_resources.resource_listdir("gluefactory", str(default_config_dir)):
        if c.endswith(".yaml"):
            default_configs[Path(c).stem] = Path(
                pkg_resources.resource_filename("gluefactory", default_config_dir + c)
            )
    if name_or_path is None:
        return None
    if name_or_path in default_configs:
        return default_configs[name_or_path]
    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find the config file: {name_or_path}. "
            f"Not in the default configs {list(default_configs.keys())} "
            "and not an existing path."
        )
    return Path(path)


def compose_config(
    name_or_path: Optional[str],
    default_config_dir: str = "configs/",
    overrides: Optional[list[str]] = None,
) -> tuple[Path, OmegaConf]:

    conf_path = parse_config_path(name_or_path, default_config_dir)
    logger.info(f"Hydra config directory: {conf_path.parent}.")

    # pathlib does not support walk_up with python < 3.12, so use os.path.relpath
    rel_conf_dir = Path(os.path.relpath(conf_path.parent, Path(__file__).parent))
    hydra.initialize(version_base=None, config_path=str(rel_conf_dir))
    custom_conf = hydra.compose(config_name=conf_path.stem, overrides=overrides)
    return conf_path, custom_conf


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
    if Path(exper).exists():
        experiment_dir = Path(exper)
    else:
        experiment_dir = Path(settings.TRAINING_PATH, exper)
    ckpts = list_checkpoints(experiment_dir)
    if not allow_interrupted:
        ckpts = [(n, p) for (n, p) in ckpts if "_interrupted" not in p.name]
    assert len(ckpts) > 0
    return sorted(ckpts)[-1][1]


def get_best_checkpoint(exper):
    """Get the checkpoint with the best loss, for a given experiment name."""
    if Path(exper).exists():
        experiment_dir = Path(exper)
    else:
        experiment_dir = Path(settings.TRAINING_PATH, exper)
    p = experiment_dir / "checkpoint_best.tar"
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


def load_experiment(
    exper, conf={}, get_last=False, ckpt=None, weights_only=settings.ALLOW_PICKLE
):
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
    ckpt = torch.load(str(ckpt), map_location="cpu", weights_only=weights_only)

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
