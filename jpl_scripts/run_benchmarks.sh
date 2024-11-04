#!/bin/bash

# SCRIPT THAT RUNS SPECIFIED BENCHMARKS AND IS DESIGNED TO BE CONFIGURABLE
# please adapt parameters below before you start!


# script params
conf="0"
out_folder_path="0"
use_extended_bm=0

# List benchmarks to run
LIST_OF_NORMAL_BM_FILES=("hpathes" "megadepth1500")
LIST_OF_EXTENDED_BM_FILES=("hpatches_extended" "megadepth1500_extended")  # extended bm usually also contain line metrics besides the default point metrics
LIST_OF_COMMON_BM_FILES=()

# user specific settings
path_to_repo="/local/home/rkreft/glue-factory"
environment="/local/home/rkreft/shared_team_folder/jpl_venv/bin/activate"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -wl|--withlines)
      echo "Option withlines is given, using extended benchmarks where possible"
      use_extended_bm=1
      ;;
    -c|--conf)
      if [[ -n "$2" ]]; then
        conf="$2"
        echo "Using Configuration: $conf."
        shift
      else
        echo "Error: Conf requires a value."
        exit 1
      fi
      ;;
    -o|--out)
      if [[ -n "$2" ]]; then
        out_folder_path="$2"
        echo "Using Outputfolder: $out_folder_path."
        shift
      else
        echo "Error: name requires a value."
        exit 1
      fi
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -c|--conf <config_file> -n|--name <name_used_as_outfolder_name> [-wl|--withlines]"
      exit 1
      ;;
  esac
  shift
done

# check parameters
if [ "$conf" = "0" ]; then
    echo "Path to config is necessary!"
    echo "Usage: $0 -c|--conf <config_file> -n|--name <name_used_as_outfolder_name> [-wl|--withlines]"
    exit 1
fi
if [ "$out_folder_path" = "0" ]; then
    echo "Path to outputfolder is necessary!"
    echo "Usage: $0 -c|--conf <config_file> -n|--name <name_used_as_outfolder_name> [-wl|--withlines]"
    exit 1
fi

# create directory if not existing
if [ ! -d "$DIRECTORY" ]; then
  mkdir "$out_folder_path"
fi


run_benchmark() {
  file_name=$1
  config_name=$2
  out_folder_path=$3
  out_file_name=$1_$(date '+%Y_%m_%d').txt

  echo ">>> Run benchmark $file_name, store output to: $out_file_name"
  python -m gluefactory.eval."$file_name" --conf="$config_name" --overwrite > "$out_folder_path/$out_file_name"
}

prompt_continue() {
  while true; do
    read -p "Do you want to continue (y/n)? " yn
    case $yn in
        [Yy]* ) break;;    # Continue the script
        [Nn]* ) echo "Exiting script."; exit;;  # Exit the script
        * ) echo "Please answer yes (y) or no (n).";;
    esac
  done
}


echo "-->Running benchmarks<--"
echo "-using config: $conf"
echo "-set repo: $path_to_repo"
echo "-set env: $environment"
echo "-out-folder: $out_folder_path"

prompt_continue

# activate environment and enter repo
echo "> Activate venv at $environment..."
source "$environment"
echo "> Use/Enter repo at $path_to_repo"
cd "$path_to_repo" || exit


# Run benchmarks

echo "> Run common benchmarks..."
for i in "${LIST_OF_COMMON_BM_FILES[@]}"
do
   run_benchmark "$i" "$conf"
   # or do whatever with individual element of the array
done

if [ $use_extended_bm -eq 1 ]; then
  echo "> Run extended benchmarks... I hope the model returns lines, otherwise I will fail xD"
  for i in "${LIST_OF_EXTENDED_BM_FILES[@]}"
  do
     run_benchmark "$i" "$conf" "$out_folder_path"
  done
 else
  echo "> Run extended benchmarks... I hope the model returns lines, otherwise I will fail xD"
  for i in "${LIST_OF_NORMAL_BM_FILES[@]}"
  do
     run_benchmark "$i" "$conf" "$out_folder_path"
  done
fi

