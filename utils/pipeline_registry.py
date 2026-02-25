from utils import files_paths as paths
from utils import pipeline_constants as consts
import os

STEPS_REGISTRY = {
    consts.SPLIT_2CH: {
        "script": os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.SPLIT_2CH_BASH),
        "display_name": "**Split 2 Channels**",
        "default": False,
        "order": 1,
    },
    consts.MOTION_CORRECTION: {
        "script": os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.MOTION_CORRECTION_BASH),
        "display_name": "**Motion Correction**",
        "default": True,
        "order": 2,
    },
    consts.PHOTOBLEACHING_CORRECTION: {
        "script": os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.PHOTOBLEACHING_BASH),
        "display_name": "**Photobleaching Correction**",
        "default": True,
        "order": 3,
    },
    consts.SUITE2P_EXTRACTION: {
        "script": os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.SUITE2P_BASH),
        "display_name": "**Suite2p Extraction**",
        "default": False,
        "order": 4,
    },
}

ANALYSIS_STEPS_REGISTRY = {
    consts.PCA_COMPUTATION: {
        "script": os.path.join(paths.CLUSTER_RUNNERS_DIR, paths.PCA_BASH),
        "display_name": "**PCA**",
        "default": False,
    },
}

TABS_REGISTRY = {
    consts.RUN: {
        "display_name": "**_Run_**",
    },
    consts.ANALYSIS: {
        "display_name": "**_Analysis_**",
    },
    consts.MONITOR: {
        "display_name": "**_Monitor_**",
    },
}
