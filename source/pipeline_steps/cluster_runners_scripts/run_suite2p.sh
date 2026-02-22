#!/bin/bash
#SBATCH -J S2P
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-4:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END

PARAMS_FILE=${1:-"None"}


. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/miniconda3/bin/activate suite2p
echo "suite2p env activated"

CODE_DIR='/ems/elsc-labs/adam-y/Adam-Lab-Shared/FromExperimentToAnalysis/Miki/source/pipeline_steps'
cd $CODE_DIR
python suite2p_extraction.py "$PARAMS_FILE"