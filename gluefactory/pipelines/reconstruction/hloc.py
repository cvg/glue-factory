"""Simple hloc-based reconstruction pipeline."""

import dataclasses
import gc
import os
import shutil
from pathlib import Path

from hloc import extract_features as hloc_extract_features
from hloc import pairs_from_exhaustive, pairs_from_retrieval
from hloc import reconstruction as hloc_reconstruction
from omegaconf import OmegaConf

from ...geometry import reconstruction
from ...models import base_model
from ...utils import export, types
from . import base


class HlocPipeline(base.ReconstructionPipeline):
    default_conf = {
        "name": "hloc_reconstruction",
        "data": {},
        "retrieval": {
            "conf": None,  # hloc global extractor: "netvlad", "dir", "openibl", "megaloc"
            "num_matched": 20,
        },
        "reconstruction": {
            "mapper_options": {},
            "camera_mode": "AUTO",
        },
        "export_half": False,
        "export_keys": None,
    }

    @dataclasses.dataclass
    class PathConfig:
        cache_dir: Path
        feature_file: Path | None = None
        matches_file: Path | None = None
        pairs_file: Path | None = None
        feature_file_name: str = "features.h5"
        matches_file_name: str = "matches.h5"
        pairs_file_name: str = "pairs.txt"
        global_features_file: Path | None = None
        global_features_file_name: str = "global_features.h5"

        def __post_init__(self):
            self.feature_file = self.feature_file or (
                self.cache_dir / self.feature_file_name
            )
            self.matches_file = self.matches_file or (
                self.cache_dir / self.matches_file_name
            )
            self.pairs_file = self.pairs_file or (self.cache_dir / self.pairs_file_name)
            self.global_features_file = self.global_features_file or (
                self.cache_dir / self.global_features_file_name
            )

    def extract_pairs(self, output_dir: Path, data: types.ReconstructionData) -> Path:
        """Extract pairs of images for reconstruction.

        Priority: custom pairs file > retrieval > exhaustive.
        """
        hloc_output = self.PathConfig(output_dir)
        if data.pairs_file:
            shutil.copy(data.pairs_file, hloc_output.pairs_file)
        elif self.conf.retrieval.conf is not None:
            self._extract_pairs_retrieval(output_dir, data)
        elif not hloc_output.pairs_file.exists():
            pairs_from_exhaustive.main(
                hloc_output.pairs_file,
                image_list=data.image_list,
            )
        return hloc_output.pairs_file

    def _extract_pairs_retrieval(
        self, output_dir: Path, data: types.ReconstructionData
    ) -> Path:
        """Extract global descriptors with hloc and generate pairs via retrieval."""
        hloc_output = self.PathConfig(output_dir)
        retrieval_conf = hloc_extract_features.confs[self.conf.retrieval.conf]

        hloc_extract_features.main(
            retrieval_conf,
            data.image_dir,
            feature_path=hloc_output.global_features_file,
            image_list=data.image_list,
        )

        pairs_from_retrieval.main(
            hloc_output.global_features_file,
            hloc_output.pairs_file,
            num_matched=self.conf.retrieval.num_matched,
        )
        return hloc_output.pairs_file

    def extract_features(
        self,
        output_dir: Path,
        model: base_model.BaseModel,
        data: types.ReconstructionData,
    ):
        """Extract features from images."""
        image_loader = data.image_loader(self.conf.data)
        export.export_predictions(
            image_loader,
            model.extractor,
            self.PathConfig(output_dir).feature_file,
            as_half=self.conf.export_half,
        )
        del image_loader
        gc.collect()
        return self.PathConfig(output_dir).feature_file

    def match_features(
        self,
        output_dir: Path,
        model: base_model.BaseModel,
        data: types.ReconstructionData,
        optional_keys=None,
    ) -> Path:
        """Match features between images."""
        hloc_output = self.PathConfig(output_dir)
        _ = self.extract_pairs(output_dir, data)
        pair_loader = data.pair_loader(
            {
                "num_workers": self.conf.data.get("num_workers", 8),
                "preprocessing": self.conf.data.get("preprocessing", {}),
                "draft_size": self.conf.data.get("draft_size", None),
            },
            pairs_file=hloc_output.pairs_file,
            features_file=hloc_output.feature_file,
        )

        export.export_predictions(
            pair_loader,
            model,
            hloc_output.matches_file,
            as_half=self.conf.export_half,
            keys=["matches0", "matches1", "matching_scores0", "matching_scores1"],
            optional_keys=optional_keys or [],
            mixed_precision=self.conf.export_half,
        )
        # Shut down DataLoader workers so they release h5 file handles
        # before hloc reconstruction tries to open the same files.
        del pair_loader
        gc.collect()
        return hloc_output.matches_file

    def export_priors(
        self,
        output_dir: Path,
        model: base_model.BaseModel,
        data: types.ReconstructionData,
    ):
        self.extract_features(output_dir, model, data)
        self.match_features(output_dir, model, data)

    def colmap_reconstruction(self, output_dir: Path, data: types.ReconstructionData):
        hloc_output = self.PathConfig(output_dir)
        return hloc_reconstruction.main(
            output_dir,
            data.image_dir,
            hloc_output.pairs_file,
            hloc_output.feature_file,
            hloc_output.matches_file,
            image_list=data.image_list,
            **OmegaConf.to_container(self.conf.reconstruction, resolve=True),
        )

    def run_reconstruction(
        self,
        output_dir: Path,
        model: base_model.BaseModel,
        data: types.ReconstructionData,
    ) -> tuple[reconstruction.Reconstruction, dict]:
        """Run the reconstruction pipeline."""
        pycolmap_rec = self.colmap_reconstruction(output_dir, data)
        rec = reconstruction.Reconstruction.from_colmap(pycolmap_rec)
        return rec, {}
