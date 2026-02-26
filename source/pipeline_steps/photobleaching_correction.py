import sys
import os
import json
import numpy as np
import tifffile
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    trimmed = gui_params.get(consts.TRIMMED, 3000)
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    split_2ch_flag = gui_params.get(consts.IS_2CH, False)
    if split_2ch_flag:
        trimmed = 0

    if split_2ch_flag:
        channels = ["neuron", "astro"]
        mc_video_paths = {
            ch: os.path.join(mc_dir, f"{ch}_{consts.MC_VIDEO_PATH}")
            for ch in channels
        }
    else:
        channels = ["full"]
        mc_video_paths = {
            "full": os.path.join(mc_dir, consts.MC_VIDEO_PATH)
        }

    return raw_video_path, mc_video_paths, channels, gui_time, fr, trimmed


def run_photobleaching_correction(fr, start_frame, stop_frame, mc_path):
    start_frame = int(start_frame) if start_frame is not None else 0
    mc_movie = tifffile.imread(mc_path)
    mc_movie = mc_movie[start_frame:, :, :]

    n_frames = mc_movie.shape[0]
    Y = mc_movie.reshape(n_frames, -1)

    stop = n_frames if stop_frame is None else int(stop_frame)

    # Mean trace across all pixels
    p = np.mean(Y, axis=1)
    t = np.arange(n_frames) / fr

    # Fitting range
    q = p[:stop]
    n_fit = len(q)

    # Initial parameter guesses
    offs = np.mean(q[int(0.75 * n_fit):])
    amp = np.mean(q[:int(0.25 * n_fit)]) - offs
    tau_init = 1000.0

    # Exponential function
    def expf(t, v):
        return v[0] + v[1] * np.exp(-t / v[2])

    # Objective: sum of squared residuals
    def objf(v):
        return np.sum((p[:stop] - expf(t[:stop], v)) ** 2)

    # Fit parameters [offset, amplitude, tau]
    initial_params = np.array([offs, amp, tau_init])
    result = minimize(objf, initial_params, method='Nelder-Mead')

    fitted_params = result.x
    fitted_curve = expf(t, fitted_params)
    fit_range = p[:stop]
    fit_pred = fitted_curve[:stop]
    fit_residual = fit_range - fit_pred
    fit_denom = np.sum((fit_range - np.mean(fit_range)) ** 2)
    bleach_r2 = 1.0 - (np.sum(fit_residual ** 2) / fit_denom) if fit_denom > 0 else 0.0

    # ---- Extract parameters ----
    pb_correct_dict = {'C': fitted_params[0],
                       'A': fitted_params[1],
                       'tau': fitted_params[2],
                       'fitted_curve': fitted_curve}

    # Divide each trace by fitted curve and rescale to mean
    corrected_movie = (Y.T / fitted_curve).T * np.mean(p)
    corrected_mean = np.mean(corrected_movie, axis=1)

    # Reshape back to original
    corrected_movie = corrected_movie.reshape(mc_movie.shape)

    slope, intercept = np.polyfit(t, corrected_mean, 1)
    eps = np.finfo(np.float32).eps
    f0 = np.percentile(corrected_mean, 20)
    if abs(f0) < eps:
        f0 = eps
    global_dff = (corrected_mean - f0) / f0
    dff_centered = global_dff - np.mean(global_dff)
    dff_std = float(np.std(global_dff))
    dff_skewness = 0.0
    if dff_std > 0:
        dff_skewness = float(np.mean((dff_centered / dff_std) ** 3))

    qc_metrics = {
        consts.TRIMMED: int(start_frame),
        consts.BLEACH_TAU: float(fitted_params[2]),
        consts.BLEACH_C: float(fitted_params[0]),
        consts.BLEACH_R2: float(bleach_r2),
        consts.RESIDUAL_SLOPE_POST_CORRECTION: float(abs(slope)),
        consts.CORRECTED_MEAN_INTENSITY: float(np.mean(corrected_mean)),
        consts.GLOBAL_DFF_STD: dff_std,
        consts.GLOBAL_DFF_MEAN: float(np.mean(global_dff)),
        consts.GLOBAL_DFF_SKENESS: dff_skewness
    }

    pb_correct_dict["original_mean"] = p
    pb_correct_dict["corrected_mean"] = corrected_mean
    pb_correct_dict["time_sec"] = t
    pb_correct_dict["corrected_mean_fit_line"] = slope * t + intercept
    pb_correct_dict["qc_metrics"] = qc_metrics

    return corrected_movie, pb_correct_dict



def save_pb_correct_data(pipeline_dir, movie_clean, pb_correct_dict, raw_video_path, channel_name=None):
    pb_dir = os.path.join(pipeline_dir, consts.PB_DIR)
    pipe_utils.mkdir(pb_dir)

    if channel_name is None:
        pb_movie_path = os.path.join(pb_dir, consts.PB_VIDEO_PATH)
        pb_fit_path = os.path.join(pb_dir, consts.PB_FIT_PATH)
    else:
        pb_movie_path = os.path.join(pb_dir, f"{channel_name}_{consts.PB_VIDEO_PATH}")
        pb_fit_path = os.path.join(pb_dir, f"{channel_name}_{consts.PB_FIT_PATH}")

    target_dtype = pipe_utils.get_signed_movie_dtype(raw_video_path)
    movie_to_save = pipe_utils.cast_movie_for_tiff_save(movie_clean, target_dtype)
    tifffile.imwrite(pb_movie_path, movie_to_save, bigtiff=True)

    np.savez(
        pb_fit_path,
        fitted_curve=pb_correct_dict["fitted_curve"],
        A=pb_correct_dict["A"],
        tau=pb_correct_dict["tau"],
        C=pb_correct_dict["C"]
    )


def save_pb_qc(pipeline_dir, pb_correct_dict, channel_name=None):
    qc_dir = os.path.join(pipeline_dir, consts.QC_DIR)
    pipe_utils.mkdir(qc_dir)

    suffix = "" if channel_name is None else f"_{channel_name}"
    qc_json_path = os.path.join(qc_dir, f"photobleaching_qc{suffix}.json")
    with open(qc_json_path, "w") as fp:
        json.dump(pb_correct_dict["qc_metrics"], fp, indent=2)

    t = pb_correct_dict["time_sec"]
    original_mean = pb_correct_dict["original_mean"]
    corrected_mean = pb_correct_dict["corrected_mean"]
    fitted_curve = pb_correct_dict["fitted_curve"]
    corrected_fit_line = pb_correct_dict["corrected_mean_fit_line"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, original_mean, label="Original mean intensity", linewidth=1.5)
    ax.plot(t, fitted_curve, label="Fitted exponential", linewidth=1.5)
    ax.plot(t, corrected_mean, label="Corrected mean intensity", linewidth=1.5)
    ax.plot(t, corrected_fit_line, label="Corrected linear fit", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Photobleaching QC")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig_path = os.path.join(qc_dir, f"photobleaching_qc_plot{suffix}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main(args):
    gui_params_path = args[1]
    raw_video_path, mc_video_paths, channels, gui_time, fr, trimmed = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)

    for ch in channels:
        mc_video_path = mc_video_paths[ch]
        print(f"Photobleaching correction on ({ch}):", mc_video_path)

        movie_clean, pb_correct_dict = run_photobleaching_correction(
            fr,
            start_frame=trimmed,
            stop_frame=None,
            mc_path=mc_video_path
        )

        save_pb_correct_data(
            pipeline_dir,
            movie_clean,
            pb_correct_dict,
            raw_video_path,
            channel_name=None if ch == "full" else ch
        )
        save_pb_qc(
            pipeline_dir,
            pb_correct_dict,
            channel_name=None if ch == "full" else ch
        )

    print(consts.STEP_COMPLETED)
    return


if __name__ == "__main__":
    main(sys.argv)
