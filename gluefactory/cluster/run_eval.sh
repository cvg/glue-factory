#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=eval_lines.out
#SBATCH --gpus=1
#SBATCH --time=1440

# above timelimit is given in minutes (24h=1440min), STDOUT of the job goes to training.out

. /etc/profile.d/modules.sh
module add cuda/12.1 # add cuda 12.1 as this is what pytorch is compiled with for gluefactory

# Assume that data is in team folder(/cluster/courses/3dv/data/team-2/minidepth) and DATA_PATH in gluefactory is set to the team-folder
# assume 3dv_venv python venv is created and all packages has been installed (via pip install -e .)
# SET OUTPUT FOLDER CORRECTLY (EVAL_PATH, TRAIN_PATH in glue factory)

# ADAPT TO YOUR REPO AND VENV
source /home/fmoeller/3dv_venv/bin/activate
cd /home/fmoeller/glue-factory

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python -m gluefactory.eval.hpatches_lines --conf=jpldd+LM --overwrite
echo "Finished training!"
