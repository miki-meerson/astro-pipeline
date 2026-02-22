import sys
import os
pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

### BASH runners ###
CLUSTER_RUNNERS_DIR = '/ems/elsc-labs/adam-y/Adam-Lab-Shared/FromExperimentToAnalysis/Miki/source/pipeline_steps/cluster_runners_scripts/'
MOTION_CORRECTION_BASH = 'run_motion_correction.sh'
PHOTOBLEACHING_BASH = 'run_photobleaching_correction.sh'
SPLIT_2CH_BASH = 'run_split_2ch.sh'
SUITE2P_BASH = 'run_suite2p.sh'

### pipeline dirs and scripts ###
PIPELINE_LOGS_DIR = os.path.join(pipeline_dir, "metadata", "logs")
PIPELINE_RUNNER_SCRIPT = 'step_manager.py'