#!/bin/bash
#SBATCH -J STA
#SBATCH --constraint=avx
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-4:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END

PARAMS_FILE=${1:-"None"}


. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/miniconda3/bin/activate caiman
echo "caiman env activated"

python -u sta_clustering.py "$PARAMS_FILE"
