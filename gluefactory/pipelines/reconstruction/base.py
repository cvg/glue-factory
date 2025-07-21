"""
Base class for reconstruction pipelines.
"""

from pathlib import Path

from omegaconf import OmegaConf

from ...geometry import reconstruction
from ...models import base_model
from ...utils import types


class ReconstructionPipeline(object, metaclass=base_model.MetaModel):
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
    """

    default_conf = {
        "name": None,
        "timeit": False,  # time forward pass
    }
    required_data_keys = []
    strict_conf = False

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        default_conf = OmegaConf.merge(
            self.base_default_conf, OmegaConf.create(self.default_conf)
        )
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self._init(conf)

    def _init(self, conf):
        """Initialize the model with the provided configuration."""
        # Optionally overridden by child classes.
        pass

    def export_priors(
        self,
        output_dir: Path,
        model: base_model.BaseModel,
        data: types.ReconstructionData,
    ) -> None:
        """Export priors from the reconstruction data."""
        # E.g. export features, matches, ...
        raise NotImplementedError("TBD")

    def run_reconstruction(
        self,
        output_dir: Path,
        model: base_model.BaseModel,
        data: types.ReconstructionData,
    ) -> tuple[reconstruction.Reconstruction, dict]:
        """Run the reconstruction process and collect statistics."""
        raise NotImplementedError("TBD")
