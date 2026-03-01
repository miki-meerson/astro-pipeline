import sys
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils


def extract_params(gui_param_path):
    with open(gui_param_path, "r") as fp:
        gui_params = json.load(fp)

    analysis_video_path = gui_params.get(consts.ANALYSIS_VIDEO_PATH, "")
    home_dir = gui_params.get(consts.HOME_DIR_LINUX, gui_params.get(consts.HOME_DIR, ""))
    if not analysis_video_path and home_dir:
        analysis_video_path = pipe_utils.get_pb_video_path_from_home(home_dir)

    raw_video_path = gui_params.get(consts.RAW_VIDEO_PATH_LINUX, analysis_video_path)
    spatial_bin_factor = int(gui_params.get(consts.SPATIAL_BIN_FACTOR, 4))
    k_keep = int(gui_params.get(consts.PCA_K_KEEP, 10))
    n_ics = int(gui_params.get(consts.ICA_N_COMPONENTS, 6))

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
        return pb_video_paths, gui_time, channels, spatial_bin_factor, pipeline_dir, k_keep, n_ics

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
        pb_video_paths = {"full": os.path.join(pb_dir, consts.PB_VIDEO_PATH)}

    return pb_video_paths, gui_time, channels, spatial_bin_factor, pipeline_dir, k_keep, n_ics


def spatial_bin(movie, bin_factor=2):
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
        n_row_binned,
        bin_factor,
        n_col_binned,
        bin_factor,
    )
    return tmp.mean(axis=(2, 4), dtype=np.float32)


def compute_pca(movie_path, bin_factor):
    movie = tifffile.imread(movie_path)
    movie_binned = spatial_bin(movie, bin_factor)
    n_frames = movie_binned.shape[0]

    x = movie_binned.reshape(n_frames, -1).T.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)

    return {
        "bin_factor": int(bin_factor),
        "binned_movie_size": tuple(int(v) for v in movie_binned.shape),
        "U": u,
        "S": s,
        "Vt": vt,
    }


def ica_from_pca(U, S, Vt, k_keep, n_ics=None, sortby="var", random_state=0, max_iter=1000, tol=1e-4):
    max_components = min(U.shape[1], S.shape[0], Vt.shape[0])
    k_keep = max(1, min(int(k_keep), max_components))
    n_ics = k_keep if n_ics is None else max(1, min(int(n_ics), k_keep))

    U_k = U[:, :k_keep]
    S_k = S[:k_keep]
    Vt_k = Vt[:k_keep, :]

    Vt_scaled = Vt_k.T.copy()
    Vt_scaled -= Vt_scaled.mean(axis=0, keepdims=True)

    ica = FastICA(
        n_components=n_ics,
        algorithm="parallel",
        fun="logcosh",
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
        fun_args={"alpha": 1.0},
    )
    ics_time = ica.fit_transform(Vt_scaled)
    mix_mat = ica.mixing_
    sep_mat = ica.components_
    ics_space = U_k @ np.diag(S_k) @ mix_mat

    if sortby.lower() == "var":
        scores = np.var(ics_time, axis=0)
    elif sortby.lower() == "kurt":
        scores = np.abs(kurtosis(ics_time, fisher=True, axis=0))
    else:
        raise ValueError("sortby must be 'var' or 'kurt'.")

    order = np.argsort(scores)[::-1]
    ics_time = ics_time[:, order]
    ics_space = ics_space[:, order]
    mix_mat = mix_mat[:, order]
    sep_mat = sep_mat[order, :]

    for i in range(ics_time.shape[1]):
        idx = int(np.argmax(np.abs(ics_time[:, i])))
        if ics_time[idx, i] < 0:
            ics_time[:, i] *= -1
            ics_space[:, i] *= -1
            mix_mat[:, i] *= -1
            sep_mat[i, :] *= -1

    return ics_time, ics_space, mix_mat, sep_mat, order


def _channel_path(pca_dir, base_name, channel_name):
    if channel_name is None:
        return os.path.join(pca_dir, base_name)
    return os.path.join(pca_dir, f"{channel_name}_{base_name}")


def visualize_pca(U, S, Vt, k_keep, movie_size, output_path, vmin_percentile=1, vmax_percentile=99):
    n_row, n_col = movie_size
    max_components = min(int(k_keep), U.shape[1], S.shape[0], Vt.shape[0])
    U_k = U[:, :max_components]
    S_k = np.diag(S[:max_components])
    Vt_k = Vt[:max_components, :]

    U_images = U_k.reshape(n_row, n_col, max_components)
    Vt_scaled = Vt_k.T @ S_k

    fig, axes = plt.subplots(max_components, 2, figsize=(10, 2.5 * max_components))
    if max_components == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(max_components):
        ax_img = axes[i, 0]
        vmin = np.percentile(U_images[:, :, i], vmin_percentile)
        vmax = np.percentile(U_images[:, :, i], vmax_percentile)
        ax_img.imshow(U_images[:, :, i], cmap="gray", vmin=vmin, vmax=vmax)
        ax_img.set_title(f"Spatial PC {i + 1}")
        ax_img.axis("off")

        ax_trace = axes[i, 1]
        ax_trace.plot(Vt_scaled[:, i], color="k", linewidth=0.8)
        ax_trace.set_title(f"Temporal PC {i + 1}")
        ax_trace.set_xlabel("Frame")
        ax_trace.set_ylabel("Amplitude")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def visualize_ica_results(ics_time, ics_space, movie_size, output_path, vmin_percentile=1, vmax_percentile=99):
    n_row, n_col = movie_size
    n_ics = ics_time.shape[1]
    ics_space_imgs = ics_space.reshape(n_row, n_col, n_ics)

    fig, axes = plt.subplots(n_ics, 2, figsize=(10, 2.5 * n_ics))
    if n_ics == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_ics):
        im = ics_space_imgs[:, :, i]
        lo = np.percentile(im, vmin_percentile)
        hi = np.percentile(im, vmax_percentile)
        vmax = max(abs(lo), abs(hi))
        vmin = -vmax

        ax_img = axes[i, 0]
        ax_img.imshow(im, cmap="seismic", vmin=vmin, vmax=vmax)
        ax_img.set_title(f"Spatial IC {i + 1}")
        ax_img.axis("off")

        ax_time = axes[i, 1]
        ax_time.plot(ics_time[:, i], color="k", linewidth=0.8)
        ax_time.set_title(f"Temporal IC {i + 1}")
        ax_time.set_xlabel("Frame")
        ax_time.set_ylabel("Amplitude")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pca_and_ica_data(pipeline_dir, pca_dict, k_keep, n_ics, channel_name=None):
    pca_dir = os.path.join(pipeline_dir, consts.PCA_DIR)
    pipe_utils.mkdir(pca_dir)

    pca_components_path = _channel_path(pca_dir, consts.PCA_COMPONENTS_PATH, channel_name)
    pca_params_path = _channel_path(pca_dir, consts.PCA_PARAMS_PATH, channel_name)
    pca_overview_path = _channel_path(pca_dir, consts.PCA_OVERVIEW_PATH, channel_name)
    sing_vals_fig_path = _channel_path(pca_dir, "singular_values.png", channel_name)
    ica_components_path = _channel_path(pca_dir, consts.ICA_COMPONENTS_PATH, channel_name)
    ica_params_path = _channel_path(pca_dir, consts.ICA_PARAMS_PATH, channel_name)
    ica_overview_path = _channel_path(pca_dir, consts.ICA_OVERVIEW_PATH, channel_name)

    S = pca_dict["S"]
    var_explained = S ** 2 / np.sum(S ** 2)
    cum_var = np.cumsum(var_explained)

    fig, _ = plt.subplots(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(var_explained, "o-")
    plt.xlabel("PC index")
    plt.ylabel("Variance explained")
    plt.title("Variance per PC")

    plt.subplot(1, 2, 2)
    plt.plot(cum_var, "o-")
    plt.axhline(0.95, color="r", linestyle="--", label="95%")
    plt.axhline(0.99, color="g", linestyle="--", label="99%")
    plt.xlabel("PC index")
    plt.ylabel("Cumulative variance")
    plt.title("Cumulative variance explained")
    plt.legend()
    plt.tight_layout()
    fig.savefig(sing_vals_fig_path, dpi=150)
    plt.close(fig)

    movie_size = pca_dict["binned_movie_size"][1:]
    visualize_pca(pca_dict["U"], pca_dict["S"], pca_dict["Vt"], k_keep, movie_size, pca_overview_path)


    ics_time, ics_space, mix_mat, sep_mat, order = ica_from_pca(
        pca_dict["U"],
        pca_dict["S"],
        pca_dict["Vt"],
        k_keep=k_keep,
        n_ics=n_ics,
    )

    np.savez(
        pca_components_path,
        U=pca_dict["U"],
        S=pca_dict["S"],
        Vt=pca_dict["Vt"],
    )

    np.savez(
        pca_params_path,
        spatial_bin_factor=pca_dict["bin_factor"],
        binned_movie_size=np.asarray(pca_dict["binned_movie_size"], dtype=np.int32),
        pca_k_keep=int(k_keep),
        ica_n_components=int(n_ics),
    )

    np.savez(
        ica_components_path,
        ics_time=ics_time,
        ics_space=ics_space,
        mix_mat=mix_mat,
        sep_mat=sep_mat,
        order=order,
    )

    np.savez(
        ica_params_path,
        pca_k_keep=int(min(k_keep, pca_dict["U"].shape[1])),
        ica_n_components=int(ics_time.shape[1]),
        sortby="var",
        binned_movie_size=np.asarray(pca_dict["binned_movie_size"], dtype=np.int32),
    )

    visualize_ica_results(ics_time, ics_space, movie_size, ica_overview_path)


def main(args):
    gui_params_path = args[1]
    pb_video_paths, gui_time, channels, spatial_bin_factor, pipeline_dir, k_keep, n_ics = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)

    for ch in channels:
        if ch == "neuron":
            continue

        pb_video_path = pb_video_paths[ch]
        print("Computing PCA on:", pb_video_path)
        pca_dict = compute_pca(pb_video_path, spatial_bin_factor)
        save_pca_and_ica_data(
            pipeline_dir,
            pca_dict,
            k_keep=k_keep,
            n_ics=n_ics,
            channel_name=None if ch == "full" else ch,
        )

    print(consts.STEP_COMPLETED)


if __name__ == "__main__":
    main(sys.argv)
