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
    image_dir: Path
    image_list: list[str] | None = None  # If None, load all images in directory.
    reference_sfm: Path | None = None  # Can be used for calibrated reconstruction.
    pairs_file: Path | None = None  # Optional list of pairs.

    def __post_init__(self):
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist.")
        if self.image_list is None:
            self.image_list = []
            for suffix in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
                self.image_list.extend(
                    [
                        str(p.relative_to(self.image_dir))
                        for p in sorted(self.image_dir.glob("**/" + suffix))
                    ]
                )
        if len(self.image_list) == 0:
            raise ValueError(f"No images found in directory {self.image_dir}.")
        if self.reference_sfm is not None and not self.reference_sfm.exists():
            raise FileNotFoundError(
                f"Reference sfm path {self.reference_sfm} does not exist."
            )
        if self.pairs_file is not None and not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file {self.pairs_file} does not exist.")

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
