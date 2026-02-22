"""
based on: @caichangjia
"""
import os
import warnings

warnings.filterwarnings('ignore')
from scipy.io import savemat
import tifffile
import json
import sys
import numpy as np
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy.volparams import volparams
from caiman.summary_images import mean_image
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils

##### GUI params extraction functions #####

def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)
    video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]
    gui_time = gui_params[consts.GUI_TIME]
    fr = pipe_utils.get_frame_rate(video_path)
    mc_dict = extract_mc_params(gui_params, fr)
    split_2ch_flag = gui_params.get(consts.SPLIT_2CH, False)
    return video_path, gui_time, mc_dict, split_2ch_flag


def extract_mc_params(gui_params, fr):
    mc_params = {consts.PW_RIGID, consts.GSIG_FILT,
                 consts.MAX_SHIFTS_X, consts.MAX_SHIFTS_Y,
                 consts.STRIDES_X, consts.STRIDES_Y, consts.OVERLAPS_X,
                 consts.OVERLAPS_Y, consts.MAX_DEVIATION_RIGID}
    mc_dict = {key: gui_params[key] for key in gui_params.keys() & mc_params}
    for key in mc_dict.keys():
        if key != consts.PW_RIGID:
            mc_dict[key] = int(mc_dict[key])
    if mc_dict[consts.GSIG_FILT] == 0:  # without high pass filtering
        mc_dict[consts.GSIG_FILT] = None
    else:
        mc_dict[consts.GSIG_FILT] = [mc_dict[consts.GSIG_FILT],
                                     0]  # 0 is just placeholder and doesnt really used in the mc code
    mc_dict[consts.STRIDES] = (mc_dict[consts.STRIDES_X], mc_dict[consts.STRIDES_Y])
    mc_dict[consts.OVERLAPS] = (mc_dict[consts.OVERLAPS_X], mc_dict[consts.OVERLAPS_Y])
    mc_dict[consts.MAX_SHIFTS] = (mc_dict[consts.MAX_SHIFTS_X], mc_dict[consts.MAX_SHIFTS_Y])
    to_delete = [consts.MAX_SHIFTS_X, consts.MAX_SHIFTS_Y, consts.STRIDES_X, consts.STRIDES_Y, consts.OVERLAPS_X,
                 consts.OVERLAPS_Y, ]
    for key in to_delete:
        del mc_dict[key]
    mc_dict[consts.FRAME_RATE] = fr
    return mc_dict


##### Caiman motion correction #####

def run_motion_correction(video_path, mc_params):
    def set_mc_parameters(video_path, mc_dict):
        opts_dict = mc_dict
        opts_dict["fnames"] = video_path
        opts_dict["border_nan"] = 'copy'
        return volparams(params_dict=opts_dict)

    def cluster_setup():
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        return dview

    opts = set_mc_parameters(video_path, mc_params)
    dview = cluster_setup()

    try:
        mc = MotionCorrect(video_path, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
    except Exception as e:
        print(str(e))
        print("motion correction failed - trying to run another one with reduce shifts")
        mc_params[consts.MAX_SHIFTS] = (10, 10)
        opts = volparams(params_dict=mc_params)
        mc = MotionCorrect(video_path, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)

    mmap_path = mc.mmap_file[0] if isinstance(mc.mmap_file, (list, tuple)) else mc.mmap_file
    movie = cm.load(mmap_path)
    movie = movie.astype(np.float32).copy()

    mean_img = mean_image(mc.mmap_file[0], window=1000, dview=dview)

    return movie, mc.shifts_rig, mean_img


def apply_reg_shifts_to_movie(movie_path, shifts_mat):
    movie = tifffile.imread(movie_path).astype(np.float32)
    n_frames, n_row, n_col = movie.shape
    movie_reg = np.empty_like(movie)

    print(f"Applying shifs to movie: {movie_path}")

    for t in range(n_frames):
        dy, dx = shifts_mat[t]
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        movie_reg[t] = cv2.warpAffine(movie[t], M, (n_col, n_row),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REFLECT)

    return movie_reg


def run_2ch_motion_correction(astro_path, neuron_path, mc_params):
    """
    apply motion correction to neuron's movie,
    and then apply the same reg shifts to astrocyte movie

    :param astro_path: path to split astrocyte movie
    :param neuron_path: path to split neuron movie
    :param mc_params: motion correction parameters from GUI
    :return:
        mc_neuron: corrected neuron movie
        mc_astro: corrected astrocyte movie
        shifts_neuron: reg shifts mat
    """
    mc_neuron, shifts_neuron, mean_image_neuron = run_motion_correction(neuron_path, mc_params)
    print("Motion correction applied to neuron movie")

    mc_astro = apply_reg_shifts_to_movie(astro_path, shifts_neuron)
    mean_image_astro = np.mean(mc_astro, axis=0)
    print("Motion correction applied to astrocyte movie")

    return mc_neuron, mc_astro, shifts_neuron, mean_image_neuron, mean_image_astro


##### saving functions #####
def save_corrected_movie(movie, pipeline_dir, mean_image, channel_name=None):
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    pipe_utils.mkdir(mc_dir)

    if channel_name is None:
        path = os.path.join(mc_dir, consts.MC_VIDEO_PATH)
    else:
        path = os.path.join(mc_dir, f"{channel_name}_{consts.MC_VIDEO_PATH}")
    tifffile.imwrite(path, movie.astype(np.float32), bigtiff=True)

    mean_img = np.array(mean_image.tolist())
    if channel_name is None:
        np.save(os.path.join(mc_dir, consts.MEAN_IMAGE), mean_img)
    else:
        np.save(os.path.join(mc_dir, f"{channel_name}_{consts.MEAN_IMAGE}"), mean_img)
    # TODO free memory (memmap)

def save_mc_shifts(pipeline_dir, shifts_mat):
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    shifts_dir = os.path.join(mc_dir, consts.SHIFTS_DIR)
    pipe_utils.mkdir(shifts_dir)
    shifts_rig = {"motion_corrected_shifts": shifts_mat}
    savemat(os.path.join(shifts_dir, consts.MC_SHIFTS_PATH), shifts_rig)


def save_mc_traces(traces, pipeline_dir):
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    traces_dir = os.path.join(mc_dir, consts.TRACES_DIR)
    pipe_utils.mkdir(mc_dir)
    pipe_utils.mkdir(traces_dir)
    path = os.path.join(traces_dir, consts.TRACES_PATH)
    traces.to_csv(path, index=False, header=False)
    return


def save_motion_qc_metrics(pipeline_dir, shifts_mat):
    qc_dir = os.path.join(pipeline_dir, consts.QC_DIR)
    pipe_utils.mkdir(qc_dir)

    shifts_arr = np.asarray(shifts_mat, dtype=np.float32)
    shift_mag = np.linalg.norm(shifts_arr, axis=1)
    qc_metrics = {
        consts.MEAN_XY_SHIFT: float(np.mean(shift_mag)),
        consts.MAX_SHIFT: float(np.max(shift_mag)),
        consts.STD_SHIFT: float(np.std(shift_mag))
    }

    qc_path = os.path.join(qc_dir, "motion_correction_qc.json")
    with open(qc_path, "w") as fp:
        json.dump(qc_metrics, fp, indent=2)


def main(args):
    gui_params_path = args[1]
    video_path, gui_time, mc_params, split_2ch_flag = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    print("Motion Correcrion on:", video_path)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(video_path)
    video_path_tif = pipe_utils.raw_to_tif(video_path) if video_path.endswith(".raw") else video_path

    # 2ch movies are 2P movies post splitting (astro and neuron channels)
    # note - not saving traces in 2P movies because there is no SLM
    if split_2ch_flag:
        split_dir = os.path.join(pipeline_dir, consts.SPLIT_DIR)
        split_astro_path = os.path.join(split_dir, consts.SPLIT_ASTRO_VIDEO_PATH)
        split_neuron_path = os.path.join(split_dir, consts.SPLIT_NEURON_VIDEO_PATH)

        mc_neuron, mc_astro, shifts_neuron, mean_image_neuron, mean_image_astro = \
            run_2ch_motion_correction(split_astro_path, split_neuron_path, mc_params)

        save_corrected_movie(mc_neuron, pipeline_dir, mean_image_neuron, channel_name="neuron")
        save_corrected_movie(mc_astro, pipeline_dir, mean_image_astro, channel_name="astro")
        save_mc_shifts(pipeline_dir, shifts_neuron)
        save_motion_qc_metrics(pipeline_dir, shifts_neuron)
    else:
        # TODO change this flow such that saving the results will be less messy and happen the same way for one or two channels
        mc_movie, shifts_mat, mean_image = run_motion_correction(video_path_tif, mc_params)
        save_corrected_movie(mc_movie, pipeline_dir, mean_image)
        save_mc_shifts(pipeline_dir, shifts_mat)
        save_motion_qc_metrics(pipeline_dir, shifts_mat)

        rois = pipe_utils.get_rois_mask(video_path)
        traces = pipe_utils.trace_extraction(mc_movie, rois)
        save_mc_traces(traces, pipeline_dir)

    print(consts.STEP_COMPLETED)

    return


if __name__ == "__main__":
    main(sys.argv)
