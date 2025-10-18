from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = ""
DATA_PATH_HOMO = Path("/gemini/data-1")  # datasets and pretrained weights
DATA_PATH_MEGA = Path("/gemini/data-2")  # datasets and pretrained weights
DATA_PATH_HPAT = Path("/gemini/data-3")  # datasets and pretrained weights
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = root / "outputs/results/"  # evaluation results
THIRD_PARTY_PATH = root / "third_party/"  # third-party libraries

ALLOW_PICKLE = False  # allow pickle (e.g. in torch.load)
