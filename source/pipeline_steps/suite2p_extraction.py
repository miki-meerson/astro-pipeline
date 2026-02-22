import shutil
import os
import sys
import json
import suite2p

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils

def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)
    raw_video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)
    gui_time = gui_params[consts.GUI_TIME]
    fr = pipe_utils.get_frame_rate(raw_video_path)

    pb_dir = os.path.join(pipeline_dir, consts.PB_DIR)
    suite2p_dir = os.path.join(pipeline_dir, consts.S2P_DIR)
    pipe_utils.mkdir(suite2p_dir)
    neuron_path = os.path.join(pb_dir, f"neuron_{consts.PB_VIDEO_PATH}")
    neuron_copy_path = os.path.join(suite2p_dir, f"neuron_{consts.PB_VIDEO_PATH}")
    shutil.copy2(neuron_path, neuron_copy_path)

    print(f"File '{neuron_path}' copied to '{suite2p_dir}' successfully.")
    return raw_video_path, gui_time, neuron_copy_path, suite2p_dir, fr


def run_suite2p_on_movie(movie_path, save_dir, fr):
    movie_dir = os.path.dirname(movie_path)
    db = {
        'data_path': [movie_dir],
        'tiff_list': [os.path.basename(movie_path)],
        'nplanes': 1,
        'nchannels': 1,
    }

    print("Running suite2p on movie '{}'.".format(movie_path))
    suite2p.run_s2p(db=db)
    print("Saved results to '{}'.".format(save_dir))


def main(args):
    gui_params_path = args[1]
    raw_video_path, gui_time, neuron_copy_path, suite2p_dir, fr = extract_params(gui_params_path)

    print("GUI TIME:", gui_time)

    run_suite2p_on_movie(neuron_copy_path, suite2p_dir, fr)

    print(consts.STEP_COMPLETED)
    return


if __name__ == "__main__":
    main(sys.argv)
