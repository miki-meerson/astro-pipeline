#!/bin/bash
#SBATCH -J PCA
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-1:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END

PARAMS_FILE=${1:-"None"}


. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/miniconda3/bin/activate caiman
echo "caiman env activated"

python pca_computation.py "$PARAMS_FILE"
