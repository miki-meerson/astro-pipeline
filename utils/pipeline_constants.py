### pipeline steps names ###
MOTION_CORRECTION = 'motion_correction'
PHOTOBLEACHING_CORRECTION = 'photobleaching_correction'
SPLIT_2CH = 'split_2ch'
SUITE2P_EXTRACTION = 'suite2p_extraction'

### tab names ###
RUN = "run"
MONITOR = "monitor"


### field names ###
RAW_VIDEO_PATH = "raw_video_path"
RAW_VIDEO_PATH_LINUX = "raw_video_path_linux"
HOME_DIR_LINUX = "home_dir_linux"
HOME_DIR = "home_dir"

### pipeline states ###
WAITING = 0
EXECUTING = 1
FINISHED = 2
FAILED = 3
CANCELLD = 4

### pipeline steps states ###
NOT_STARTED = "not started"
JOB_PENDING = "Pending"
JOB_RUNNING = "running"
JOB_FAILED = "failed"
JOB_FINISHED = "completed"
JOB_CANCELLD = "cancelled"

### SLURM states ###
SLURM_PENDING = "PENDING"
SLURM_RUNNING = "RUNNING"
SLURM_FAILED = "FAILED"
SLURM_FINISHED = "COMPLETED"
SLURM_CANCELLD = "CANCELLED"

### new SLURM commands ### 4.8.2025
RUN_JOB_COMMAND = "/usr/bin/sbatch"
JOB_STATE_COMMAND = "/usr/bin/sacct -j {} -n -o State -P"
JOB_STATE_COMMAND_2 = "/usr/bin/scontrol show job {}  | grep JobState"
CANCEL_JOB_COMMAND = "/usr/bin/scancel {}"
GET_LOG_PATH_COMMAND = "/usr/bin/scontrol show job {} | grep StdOut"

### directories and files names ###
PIPELINE_DIR = "pipeline_results"
RAW_VIDEO_TIF = "raw_video.tif"
SPLIT_DIR = "2ch_split"
SPLIT_NEURON_VIDEO_PATH = "2ch_split_neuron.tif"
SPLIT_ASTRO_VIDEO_PATH = "2ch_split_astro.tif"
MC_DIR = "motion_corrected"
SHIFTS_DIR = "shifts"
MEAN_IMAGE = 'mean_image'
MC_VIDEO_PATH = "motion_corrected.tif"
MC_SHIFTS_PATH = "motion_corrected_shifts.mat"
MC_SHIFTS = "mc_shifts"
PB_DIR = "pb_corrected"
PB_VIDEO_PATH = "photobleaching_corrected.tif"
PB_FIT_PATH = "pb_corrected_fit_params.npz"
S2P_DIR = "for_suite2p"
QC_DIR = "qc"
TRACES_DIR = "traces"
TRACES_PATH = "_traces.csv"
VIRMEN_PREFIX = "imaging_"
TS_DIR_PREFIX = "TS"
TS_XML = "ThorRealTimeDataSettings.xml"
PARAMS_DIR_NAME = "pipeline_params"
PARAMS_FILE_SUFFIX_NAME = "_params.json"

### motion correction ###
FRAME_RATE = "fr"
PW_RIGID = "pw_rigid"
GSIG_FILT = "gSig_filt"
GSIG_FILT_X = "gSig_filt_x"
GSIG_FILT_Y = "gSig_filt_y"
MAX_SHIFTS = "max_shifts"
MAX_SHIFTS_X = "max_shifts_x"
MAX_SHIFTS_Y = "max_shifts_y"
STRIDES = "strides"
STRIDES_X = "strides_x"
STRIDES_Y = "strides_y"
OVERLAPS = "overlaps"
OVERLAPS_X = "overlaps_x"
OVERLAPS_Y = "overlaps_y"
MAX_DEVIATION_RIGID = "max_deviation_rigid"
MEAN_XY_SHIFT = "mean_xy_shift"
MAX_SHIFT = "max_shift"
STD_SHIFT = "std_shift"

### columns names ###
EXPERIMENT_DATE = "experiment_date"
CAGE = "cage"
MOUSE_NAME = "mouse_name"
GUI_TIME = "gui_time"
STRAIN = "strain"
FOV = "FOV"
FRAME_RATE = "frame_rate"
MOVIE_PATH = "movie_path"
BEHAVIOR = "behavior"
EXPERIMENT_DETAILS = "exp_details"
WIDTH = "width"
HEIGHT = "height"
FRAMES_NUMBER = "frames_number"
ORIGINAL_MEAN_INTENSITY = "original_mean_intensity"


### photobleaching metrics - I(t)=A⋅e^(−t/τ)+C, corrected_mean(t)=mt+b ###
TRIMMED = "trimmed"
BLEACH_TAU = "bleach_tau"
BLEACH_C = "bleach_C"
BLEACH_R2 = "bleach_r2"
RESIDUAL_SLOPE_POST_CORRECTION = "residual_slope_post_correction"  # |m|
CORRECTED_MEAN_INTENSITY = "corrected_mean_intensity"

### global signal metrics (on corrected movie) ###
GLOBAL_DFF_STD = "global_dff_std"
GLOBAL_DFF_MEAN = "global_dff_mean"
GLOBAL_DFF_SKENESS = "global_dff_skewness"

 ### Messages ####
STEP_COMPLETED = "Everything worked well. The Script finished to run."

