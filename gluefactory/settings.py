from pathlib import Path

cluster_team_folder = Path(
    "/local/home/Point-Line/"
)  # cluster team folder for 3dv
root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = cluster_team_folder / "data"  # datasets and pretrained weights
TRAINING_PATH = cluster_team_folder / "outputs/training/"  # training checkpoints
EVAL_PATH = cluster_team_folder / "outputs/results/"  # evaluation results
