#!/bin/bash
# RUN BENCHMARK ON DLAB
# activates venv and runs training. Dont forget to activate tmux before
# Cmd params 'run_eval_megadepth.sh [exp_name] [path to conf/conf name]'

echo "Conf-Path: $1"

source /local/home/rkreft/shared_team_folder/jpl_venv/bin/activate
cd ~/glue-factory || exit

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python -m gluefactory.eval.megadepth1500_extended --conf="$1" --overwrite
echo "Finished eval!"
