#!/bin/bash

conf=0
use_extended_bm=0

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
        echo "Error: Count requires a value."
        exit 1
      fi
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

if [ $conf -eq 0 ]; then
    echo "Path to config is necessary!"
    exit 1
fi

run_benchmark() {
  file_name=$1
  config_name=$2
  out_file_name=$1_$2_$(date '+%Y_%m_%d').txt

  echo "Run benchmark $file_name, store output to: "
  python -m gluefactory.eval."$file_name" --conf="$config_name" --overwrite > out_file_name
}


LIST_OF_NORMAL_BM_FILES=("hpathes" "megadepth1500")
LIST_OF_EXTENDED_BM_FILES=("hpatches_extended" "megadepth1500")
LIST_OF_COMMON_BM_FILES=()

path_to_repo="/local/home/rkreft/glue-factory"
environment="/local/home/rkreft/shared_team_folder/jpl_venv/bin/activate"


echo "-->Running benchmarks<--"
echo "-using config: $conf"

# activate environment and enter repo
echo "Activate venv at $environment..."
source "$environment"
echo "Use repo at $path_to_repo"
cd "$path_to_repo"


# Run benchmarks

echo "Run common benchmarks..."
for i in "${LIST_OF_COMMON_BM_FILES[@]}"
do
   echo "$i"
   run_benchmark "$i" conf
   # or do whatever with individual element of the array
done

if [ $use_extended_bw -eq 1 ]; then
  echo "Run extended benchmarks... I hope the model returns lines, otherwise I will fail xD"
  for i in "${LIST_OF_EXTENDED_BM_FILES[@]}"
  do
     echo "$i"
     run_benchmark "$i" conf
  done
 else
  echo "Run extended benchmarks... I hope the model returns lines, otherwise I will fail xD"
  for i in "${LIST_OF_NORMAL_BM_FILES[@]}"
  do
     echo "$i"
     run_benchmark "$i" conf
  done
fi












