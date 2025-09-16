"""Type definitions and global variables."""

import dataclasses
from pathlib import Path
from typing import Any, TypeAlias

from omegaconf import OmegaConf

STRING_CLASSES = (str, bytes)

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1
Key: TypeAlias = str | tuple[str, ...]
Value: TypeAlias = Any
Tree: TypeAlias = dict[Key, Value]


@dataclasses.dataclass
class ReconstructionData:
    image_list: list[str]
    image_dir: Path
    reference_sfm: Path | None = None  # Can be used for calibrated reconstruction.
    pairs_file: Path | None = None  # Optional list of pairs.

    def image_loader(self, data_conf: dict):
        """Load images from the image directory."""
        from ..datasets.image_folder import ImageFolder

        default_conf = {
            "root_folder": self.image_dir,
            "images": self.image_list,
        }
        conf = OmegaConf.merge(default_conf, data_conf)
        return ImageFolder(conf).get_data_loader("train")

    def pair_loader(
        self, data_conf: dict, pairs_file: Path, features_file: Path | None = None
    ):
        """Load pairs from a file."""

        from ..datasets.image_pairs import ImagePairs

        default_conf = {
            "pairs": pairs_file,
            "root": self.image_dir,
            "load_features": {
                "do": features_file is not None,
                "path": str(features_file) if features_file else "",
                "collate": False,
            },
        }
        conf = OmegaConf.merge(default_conf, data_conf)
        return ImagePairs(conf).get_data_loader("train")
