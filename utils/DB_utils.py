## columns names ###
EXPERIMENT_DATE = "experiment_date"
CAGE = "cage"
MOUSE_NAME = "mouse_name"
STRAIN = "strain"
FOV = "FOV"
FRAME_RATE = "frame_rate"
MOVIE_PATH = "movie_path"
BEHAVIOR = "behavior"
EXP_DETAILS = "exp_details"
WIDTH = "width"
HEIGHT = "height"
FRAMES_NUMBER = "frames_number"
ORIGINAL_MEAN_INTENSITY = "original_mean_intensity"
SESSIONS_COUNTER = "sessions_counter"

### motion correction ###
MEAN_XY_SHIFT = "mean_xy_shift"
MAX_SHIFT = "max_shift"
STD_SHIFT = "std_shift"

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

