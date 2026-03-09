"""Raw COLMAP automatic reconstruction pipeline (SIFT + exhaustive matching)."""

import logging
import shutil
from pathlib import Path

import pycolmap

from ...geometry import reconstruction
from ...models import base_model
from ...utils import types
from . import base

logger = logging.getLogger(__name__)


class ColmapPipeline(base.ReconstructionPipeline):
    """Runs COLMAP's full automatic pipeline: SIFT extraction, exhaustive
    matching, and incremental mapping. No learned features involved."""

    default_conf = {
        "name": "colmap",
        "sift_extraction": {},
        "sift_matching": {},
        "mapper_options": {},
    }

    def export_priors(self, output_dir, model, data):
        pass

    def run_reconstruction(
        self,
        output_dir: Path,
        model: base_model.BaseModel,
        data: types.ReconstructionData,
    ) -> tuple[reconstruction.Reconstruction, dict]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = str(data.image_dir)

        # Create COLMAP database
        db_path = str(output_dir / "colmap.db")

        # Extract SIFT features
        logger.info("Extracting SIFT features")
        sift_opts = pycolmap.SiftExtractionOptions(
            dict(self.conf.sift_extraction) if self.conf.sift_extraction else {}
        )
        pycolmap.extract_features(
            database_path=db_path,
            image_path=image_dir,
            image_names=list(data.image_list),
            sift_options=sift_opts,
        )

        # Exhaustive matching
        logger.info("Running exhaustive matching")
        match_opts = pycolmap.SiftMatchingOptions(
            dict(self.conf.sift_matching) if self.conf.sift_matching else {}
        )
        pycolmap.match_exhaustive(
            database_path=db_path,
            sift_options=match_opts,
        )

        # Incremental mapping
        recon_path = output_dir / "reconstruction"
        if recon_path.is_dir():
            shutil.rmtree(recon_path)
        recon_path.mkdir(parents=True, exist_ok=True)

        mapper_opts = dict(self.conf.mapper_options) if self.conf.mapper_options else {}
        mapper_opts.setdefault("multiple_models", False)

        logger.info("Running COLMAP incremental mapping")
        pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=image_dir,
            output_path=str(recon_path),
            options=pycolmap.IncrementalPipelineOptions(mapper_opts),
        )

        # Load reconstruction
        model_path = recon_path / "0"
        if not model_path.exists():
            subdirs = sorted(recon_path.iterdir())
            if subdirs:
                model_path = subdirs[0]
            else:
                raise RuntimeError("COLMAP mapper produced no reconstruction")

        shutil.copytree(model_path, output_dir, dirs_exist_ok=True)
        rec = reconstruction.Reconstruction.from_colmap(model_path)
        logger.info("Reconstruction: %s", rec)

        stats = {}
        return rec, stats
