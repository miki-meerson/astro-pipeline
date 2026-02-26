import sys
import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils


def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)

    analysis_video_path = gui_params.get(consts.ANALYSIS_VIDEO_PATH, "")
    home_dir = gui_params.get(consts.HOME_DIR_LINUX, gui_params.get(consts.HOME_DIR, ""))
    if not analysis_video_path and home_dir:
        analysis_video_path = pipe_utils.get_pb_video_path_from_home(home_dir)

    raw_video_path = gui_params.get(consts.RAW_VIDEO_PATH_LINUX, analysis_video_path)
    spatial_bin_factor = gui_params[consts.SPATIAL_BIN_FACTOR]

    if analysis_video_path:
        video_path = analysis_video_path
        parent_dir = os.path.dirname(video_path)
        if os.path.basename(parent_dir) == consts.PB_DIR:
            pipeline_dir = os.path.dirname(parent_dir)
        else:
            pipeline_dir = parent_dir
        pb_video_paths = {"full": video_path}
        channels = ["full"]
        gui_time = gui_params.get(consts.GUI_TIME, "")
        return raw_video_path, pb_video_paths, gui_time, channels, spatial_bin_factor, pipeline_dir

    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)
    gui_time = gui_params[consts.GUI_TIME]
    pb_dir = os.path.join(pipeline_dir, consts.PB_DIR)
    split_2ch_flag = gui_params.get(consts.IS_2CH, False)

    if split_2ch_flag:
        channels = ["neuron", "astro"]
        pb_video_paths = {
            ch: os.path.join(pb_dir, f"{ch}_{consts.PB_VIDEO_PATH}")
            for ch in channels
        }
    else:
        channels = ["full"]
        pb_video_paths = {
            "full": os.path.join(pb_dir, consts.PB_VIDEO_PATH)
        }

    return raw_video_path, pb_video_paths, gui_time, channels, spatial_bin_factor, pipeline_dir



def spatial_bin(movie, bin_factor=2):
    """ Spatially bin a 3D movie (frames, height, width) by averaging over bin_factor x bin_factor blocks. """
   
    # Keep numerical stability for downstream linalg (SVD/PCA does not support float16).
    movie = np.asarray(movie, dtype=np.float32)
    n_frames, n_row, n_col = movie.shape
    n_row_crop = (n_row // bin_factor) * bin_factor
    n_col_crop = (n_col // bin_factor) * bin_factor

    if n_row_crop != n_row or n_col_crop != n_col:
        movie = movie[:, :n_row_crop, :n_col_crop]

    n_row_binned = n_row_crop // bin_factor
    n_col_binned = n_col_crop // bin_factor

    tmp = movie.reshape(
        n_frames,
        n_row_binned, bin_factor,
        n_col_binned, bin_factor
    )

    return tmp.mean(axis=(2, 4), dtype=np.float32)



def compute_pca(movie_path, bin_factor):
    movie = tifffile.imread(movie_path)

    movie_binned = spatial_bin(movie, bin_factor)
    n_frames = movie_binned.shape[0]

    # Flatten
    X = movie_binned.reshape(n_frames, -1).T.astype(np.float32, copy=False)  # [pixels x time]
    X = X - X.mean(axis=1, keepdims=True)  # center per pixel

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # U[pixels, k], S[k], Vt[k, time]

    pca_dict = {
        'bin_factor': bin_factor,
        'U': U,
        'S': S,
        'Vt': Vt,
    }

    return pca_dict


def save_pca_data(pipeline_dir, pca_dict, channel_name=None):
    pca_dir = os.path.join(pipeline_dir, consts.PCA_DIR)
    pipe_utils.mkdir(pca_dir)

    if channel_name is None:
        pca_components_path = os.path.join(pca_dir, consts.PCA_COMPONENTS_PATH)
        pca_params_path = os.path.join(pca_dir, consts.PCA_PARAMS_PATH)
        sing_vals_fig_path = os.path.join(pca_dir, "singular_values.png")
    else:
        pca_components_path = os.path.join(pca_dir, f"{channel_name}_{consts.PCA_COMPONENTS_PATH}")
        pca_params_path = os.path.join(pca_dir, f"{channel_name}_{consts.PCA_PARAMS_PATH}")
        sing_vals_fig_path = os.path.join(pca_dir, f"{channel_name}_singular_values.png")

    np.savez(
        pca_components_path,
        U=pca_dict["U"],
        S=pca_dict["S"],
        Vt=pca_dict["Vt"]
    )

    np.savez(
        pca_params_path,
        spatial_bin_factor=pca_dict["bin_factor"] 
    )

    S = pca_dict['S']
    var_explained = S ** 2 / np.sum(S ** 2)
    cum_var = np.cumsum(var_explained)

    fig, _ = plt.subplots(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(var_explained, 'o-')
    plt.xlabel('PC index')
    plt.ylabel('Variance explained')
    plt.title('Variance per PC')

    plt.subplot(1, 2, 2)
    plt.plot(cum_var, 'o-')
    plt.axhline(0.95, color='r', linestyle='--', label='95%')
    plt.axhline(0.99, color='g', linestyle='--', label='99%')
    plt.xlabel('PC index')
    plt.ylabel('Cumulative variance')
    plt.title('Cumulative variance explained')
    plt.legend()
    plt.tight_layout()

    fig.savefig(sing_vals_fig_path, dpi=150)
    plt.close(fig)



def main(args):
    gui_params_path = args[1]
    raw_video_path, pb_video_paths, gui_time, channels, spatial_bin_factor, pipeline_dir = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)

    for ch in channels:
        if ch == "neuron": continue
        pb_video_path = pb_video_paths[ch]

        print(f"Computing PCA on:", pb_video_path)
        pca_dict = compute_pca(pb_video_path, spatial_bin_factor)
        save_pca_data(pipeline_dir, pca_dict, channel_name=None if ch == "full" else ch)
    
    print(consts.STEP_COMPLETED)
    return


if __name__ == "__main__":
    main(sys.argv)
