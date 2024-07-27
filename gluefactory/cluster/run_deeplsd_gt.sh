#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=deeplsd_gt.out
#SBATCH --gpus=4
#SBATCH --time=1440
#SBATCH --mem=120000MB

# above timelimit is given in minutes (24h=1440min), STDOUT of the job goes to superpoint_gt.out

. /etc/profile.d/modules.sh
module add cuda/12.1 # add cuda 12.1 as this is what pytorch is compiled with for gluefactory

# Assume that data is in team folder(/cluster/courses/3dv/data/team-2/minidepth) and DATA_PATH in gluefactory is set to the team-folder
# assume 3dv_venv python venv is created and all packages has been installed (via pip install -e .)
# SET OUTPUT FOLDER CORRECTLY (EVAL_PATH in glue factory)

# ADAPT TO YOUR REPO AND VENV
source /home/fmoeller/merged_3dv_venv/bin/activate
cd /home/fmoeller/merged_gluefactory

# Run script
python -m gluefactory.ground_truth_generation.deeplsd_gt_multiple_files --num_H=100 --output_folder=deeplsd_gt/oxford_paris_mini --n_gpus=4 --n_jobs_dataloader=2 --image_name_list=oxp_img.txt
echo "Size of generated groundtruth: "
du -sh /cluster/courses/3dv/data/team-2/outputs/results/deeplsd_gt/oxford_paris_mini
