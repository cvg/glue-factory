from pathlib import Path

from . import io
from .doppelgangers import DoppelgangersPipeline


class DoppelgangersVisymPipeline(DoppelgangersPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "doppelgangers",
            "root": "doppelgangerspp",
            "num_workers": 16,
            "preprocessing": {
                "resize": 1024,  # we also resize during eval to have comparable metrics
                "side": "long",
                "crop_if_short_side": True,
                "square_pad": True,
                "center_pad": True,
            },
            "seed": 42,
            "visym": False,
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {"score_key": "overlaps", "subset_idxs": None},
    }


if __name__ == "__main__":
    io.run_cli(DoppelgangersVisymPipeline, name=Path(__file__).stem)
