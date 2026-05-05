"""
Base class for trainable models.
"""

import logging
from abc import ABCMeta, abstractmethod
from copy import copy

import omegaconf
import torch
from omegaconf import OmegaConf
from torch import nn

from gluefactory.utils import misc

logger = logging.getLogger(__name__)


class MetaModel(ABCMeta):
    def __prepare__(name, bases, **kwds):
        total_conf = OmegaConf.create()
        for base in bases:
            for key in ("base_default_conf", "default_conf"):
                update = getattr(base, key, {})
                if isinstance(update, dict):
                    update = OmegaConf.create(update)
                total_conf = OmegaConf.merge(total_conf, update)
        return dict(base_default_conf=total_conf)


class BaseModel(nn.Module, metaclass=MetaModel):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It recursively updates the default_conf of all parent classes, and
        it is updated by the user-provided configuration passed to __init__.
        Configurations can be nested.

        required_data_keys: list of expected keys in the input data dictionary.

        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unknown configuration entries will raise an error.

        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.

        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.

        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """

    default_conf = {
        "name": None,
        "trainable": True,  # if false: do not optimize this model parameters
        "freeze_batch_normalization": False,  # use test-time statistics
        "timeit": False,  # time forward pass
        "visualize": True,  # visualize model predictions
        "compile": True,  # compile the model for faster inference
        "compile_loss": True,  # compile losses for faster inference
        "run_loss_in_forward": False,  # compute losses inside forward
        "force_f32": False,
    }
    required_data_keys = []
    strict_conf = False

    are_weights_initialized = False

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.input_conf = conf
        default_conf = OmegaConf.merge(
            self.base_default_conf, OmegaConf.create(self.default_conf)
        )
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        # fixme: backward compatibility
        if "pad" in conf and "pad" not in default_conf:  # backward compat.
            with omegaconf.read_write(conf):
                with omegaconf.open_dict(conf):
                    conf["interpolation"] = {"pad": conf.pop("pad")}

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        if self.conf.freeze_batch_normalization:
            self.apply(freeze_bn)

        return self

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""

        def recursive_key_check(expected, given):
            for key in expected:
                assert key in given, f"Missing key {key} in data"
                if isinstance(expected, dict):
                    recursive_key_check(expected[key], given[key])

        recursive_key_check(self.required_data_keys, data)
        if self.conf.force_f32:
            pred = misc.force_f32(self._forward)(data)
        else:
            pred = self._forward(data)

        if self.conf.run_loss_in_forward:
            pred["loss"] = self.loss(pred, data)
        return pred

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def loss_metrics(self, pred, data):
        """Wrapper around loss and metrics computation."""

        if self.conf.run_loss_in_forward:
            return pred.pop("loss")
        else:
            return self.loss(pred, data)

    def visualize(self, pred, data, **kwargs):
        """To be implemented by the child class."""
        return {}

    def pr_metrics(self, pred, data):
        """To be implemented by the child class."""
        return {}

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Load the state dict of the model, and set the model to initialized.
        If strict=False, parameters with mismatched shapes are skipped
        (default-initialized) and logged."""
        if not strict:
            model_state = self.state_dict()
            mismatched = []
            for key in list(state_dict.keys()):
                if key not in model_state:
                    continue
                try:
                    shapes_match = state_dict[key].shape == model_state[key].shape
                except RuntimeError:
                    continue  # UninitializedParameter (LazyLinear etc.)
                if not shapes_match:
                    mismatched.append(
                        f"{key}: checkpoint {list(state_dict[key].shape)}"
                        f" vs model {list(model_state[key].shape)}"
                    )
                    del state_dict[key]
            if mismatched:
                logger.warning(
                    "Skipped %d parameters with mismatched shapes:\n  %s",
                    len(mismatched),
                    "\n  ".join(mismatched),
                )
        ret = super().load_state_dict(state_dict, strict=strict, **kwargs)
        self.set_initialized()
        return ret

    def is_initialized(self):
        """Recursively check if the model is initialized, i.e. weights are loaded"""
        is_initialized = True  # initialize to true and perform recursive and
        for _, w in self.named_children():
            if isinstance(w, BaseModel):
                # if children is BaseModel, we perform recursive check
                is_initialized = is_initialized and w.is_initialized()
            else:
                # else, we check if self is initialized or the children has no params
                n_params = len(list(w.parameters()))
                is_initialized = is_initialized and (
                    n_params == 0 or self.are_weights_initialized
                )
        return is_initialized

    def set_initialized(self, to: bool = True):
        """Recursively set the initialization state."""
        self.are_weights_initialized = to
        for _, w in self.named_children():
            if isinstance(w, BaseModel):
                w.set_initialized(to)

    def make_ddp(self, *args, **kwargs) -> nn.parallel.DistributedDataParallel:
        """Make the model DDP compatible."""
        model = nn.SyncBatchNorm.convert_sync_batchnorm(self)
        model = nn.parallel.DistributedDataParallel(model, *args, **kwargs)
        # Add key methods to the DDP model
        model.loss = self.loss
        model.loss_metrics = self.loss_metrics
        model.visualize = self.visualize
        model.pr_metrics = self.pr_metrics
        model.load_state_dict = self.load_state_dict
        return model

    def compile(self, *args, **kwargs) -> "BaseModel":
        """Compile the model for faster inference."""
        if self.conf.compile:
            logger.info("Compiling %s", str(type(self)))
            self._compile(*args, **kwargs)
            if self.conf.compile_loss:
                self._compile_loss(*args, **kwargs)
        return self

    def _compile(self, *args, **kwargs) -> None:
        """Compile the model for faster inference."""
        super().compile(*args, **kwargs)

    def _compile_loss(self, *args, **kwargs) -> None:
        self.loss = torch.compile(self.loss, *args, **kwargs)
