#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=training.out
#SBATCH --gpus=4
#SBATCH --time=5000

# above timelimit is given in minutes (24h=1440min), STDOUT of the job goes to training.out

. /etc/profile.d/modules.sh
module add cuda/12.1 # add cuda 12.1 as this is what pytorch is compiled with for gluefactory

# Assume that data is in team folder(/cluster/courses/3dv/data/team-2/minidepth) and DATA_PATH in gluefactory is set to the team-folder
# assume 3dv_venv python venv is created and all packages has been installed (via pip install -e .)
# SET OUTPUT FOLDER CORRECTLY (EVAL_PATH, TRAIN_PATH in glue factory)

# ADAPT TO YOUR REPO AND VENV
source /home/fmoeller/merged_3dv_venv/bin/activate
cd /home/fmoeller/merged_gluefactory

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python -m gluefactory.train rk_jpldd_test --conf=gluefactory/configs/rk_jpldd_test.yaml --distributed --restore
echo "Finished training!"
