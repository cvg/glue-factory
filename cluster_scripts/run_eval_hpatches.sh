#!/bin/bash
#activates venv and runs training. Dont forget to activate tmux before
# Cmd params 'run_training.sh [exp_name] [path to conf/conf name]'

echo "Conf-Path: $1"

source /local/home/rkreft/shared_team_folder/jpl_venv/bin/activate
cd /local/home/rkreft/glue-factory

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python -m gluefactory.eval.hpatches_extended --conf="$1" --overwrite
echo "Finished Benchmark!"
