from pathlib import Path

cluster_team_folder = Path(
    "/local/home/rkreft/shared_team_folder"
)  # cluster team folder for 3dv
root = Path(__file__).parent.parent  # top-level directory
p = Path("/local/home/Point-Line/")
DATA_PATH = p / "data"  # datasets and pretrained weights
TRAINING_PATH = p / "outputs/training/"  # training checkpoints
EVAL_PATH = p / "outputs/results/"  # evaluation results
