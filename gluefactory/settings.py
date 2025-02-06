from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
p = Path("/local/home/Point-Line/")  # cluster team folder for 3dv
DATA_PATH = p / "data"  # datasets and pretrained weights
TRAINING_PATH = p / "outputs/training/"  # training checkpoints
EVAL_PATH = p / "outputs/results/"  # evaluation results
