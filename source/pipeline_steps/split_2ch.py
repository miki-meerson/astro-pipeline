import sys
import os
import json
import tifffile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils

##### GUI params extraction functions #####

def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)

    raw_video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]
    fr = pipe_utils.get_frame_rate(raw_video_path)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)
    gui_time = gui_params[consts.GUI_TIME]

    return raw_video_path, gui_time, fr


def run_split_2ch(input_path):
    full_movie = tifffile.imread(input_path)
    n_frames, n_rows, n_cols = full_movie.shape
    middle_frame = n_frames // 2

    astro_movie = full_movie[:middle_frame, :, :]
    neuron_movie = full_movie[middle_frame:, :, :]

    return astro_movie, neuron_movie


def save_split_movies(pipeline_dir, astro_movie, neuron_movie, raw_video_path):
    split_dir = os.path.join(pipeline_dir, consts.SPLIT_DIR)
    pipe_utils.mkdir(split_dir)
    target_dtype = pipe_utils.get_signed_movie_dtype(raw_video_path)

    astro_path = os.path.join(split_dir, consts.SPLIT_ASTRO_VIDEO_PATH)
    astro_to_save = pipe_utils.cast_movie_for_tiff_save(astro_movie, target_dtype)
    tifffile.imwrite(astro_path, astro_to_save, bigtiff=True)

    split_neuron_path = os.path.join(split_dir, consts.SPLIT_NEURON_VIDEO_PATH)
    neuron_to_save = pipe_utils.cast_movie_for_tiff_save(neuron_movie, target_dtype)
    tifffile.imwrite(split_neuron_path, neuron_to_save, bigtiff=True)


def main(args):
    gui_params_path = args[1]
    raw_video_path, gui_time, fr = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    print("Splitting to 2 channels from movie:", raw_video_path)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)

    astro_movie, neuron_movie = run_split_2ch(raw_video_path)
    save_split_movies(pipeline_dir, astro_movie, neuron_movie, raw_video_path)

    print(consts.STEP_COMPLETED)
    return


if __name__ == "__main__":
    main(sys.argv)
