#!/bin/bash
# Cmd params 'run_training.sh [exp_name] [path to conf]'

#SBATCH --time=2000
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3072
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --output=train.out
#SBATCH --mail-type=END
#SBATCH --mail-user=r.kreft@stud.ethz.ch
#SBATCH --job-name="jpl_training_$1"

module load eth_proxy


echo "Exp-Name: $1"
echo "Conf-Path: $2"

source /cluster/home/rkreft/jpl_venv/bin/activate
cd /cluster/home/rkreft/glue-factory

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python -m gluefactory.train "$1" --conf="$2"
echo "Finished training!"
