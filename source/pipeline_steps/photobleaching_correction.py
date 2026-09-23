import sys
import os
import json
import numpy as np
import tifffile
from scipy.optimize import curve_fit
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

    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)
    gui_time = gui_params[consts.GUI_TIME]
    fit_start_frame = gui_params.get(
        consts.PB_FIT_START_FRAME,
        gui_params.get(consts.TRIMMED, 0)
    )
    fit_end_frame = gui_params.get(consts.PB_FIT_END_FRAME, None)
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    split_2ch_flag = gui_params.get(consts.IS_2CH, False)

    is2p = "hyp" in str(raw_video_path).lower()
    fr = pipe_utils.get_frame_rate(raw_video_path, is2p=is2p)

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

    return raw_video_path, mc_video_paths, channels, gui_time, fr, fit_start_frame, fit_end_frame


def run_photobleaching_correction(fr, fit_start_frame, fit_end_frame, mc_path):
    fit_start_frame = int(fit_start_frame) if fit_start_frame is not None else 0
    mc_movie_full = tifffile.imread(mc_path)
    n_total_frames = int(mc_movie_full.shape[0])
    if fit_start_frame < 0:
        fit_start_frame = 0
    if fit_start_frame >= n_total_frames:
        raise ValueError(
            f"Fit start frame ({fit_start_frame}) is outside movie length ({n_total_frames})."
        )
    fit_end_frame = n_total_frames - 1 if fit_end_frame is None else int(fit_end_frame)
    fit_end_frame = min(fit_end_frame, n_total_frames - 1)
    if fit_end_frame <= fit_start_frame:
        raise ValueError(
            f"Fit end frame ({fit_end_frame}) must be larger than fit start frame ({fit_start_frame})."
        )

    n_frames = n_total_frames
    Y = mc_movie_full.reshape(n_frames, -1)

    fit_slice = slice(fit_start_frame, fit_end_frame + 1)
    n_fit = fit_end_frame - fit_start_frame + 1
    if n_fit < 3:
        raise ValueError("Photobleaching fit range must contain at least 3 frames.")

    # Mean trace across all pixels
    p = np.mean(Y, axis=1)
    full_mean_trace = p
    frame_idx = np.arange(n_frames, dtype=np.int64)
    t_abs = frame_idx / fr
    t_fit = (frame_idx[fit_slice] - fit_start_frame) / fr

    # Fitting range
    q = p[fit_slice]

    # Initial parameter guesses
    offs = np.mean(q[int(0.75 * n_fit):])
    early_window = max(1, int(0.25 * n_fit))
    amp = np.mean(q[:early_window]) - offs
    tau_init = 1000.0

    # Exponential function
    def expf(tt, c, a, tau):
        return c + a * np.exp(-tt / tau)

    # Fit parameters [offset, amplitude, tau] on time relative to the selected fit start.
    initial_params = np.array([max(offs, 0.0), max(amp, 0.0), max(tau_init, 1.0 / fr)], dtype=np.float64)
    lower_bounds = np.array([0.0, 0.0, 1.0 / fr], dtype=np.float64)
    upper_bounds = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    try:
        fitted_params, _ = curve_fit(
            expf,
            t_fit,
            q,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )
    except Exception:
        fitted_params = initial_params
    t_full_for_fit = (frame_idx - fit_start_frame) / fr
    fitted_curve = expf(t_full_for_fit, *fitted_params)
    fit_range = q
    fit_pred = fitted_curve[fit_slice]
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
    corrected_movie = corrected_movie.reshape(mc_movie_full.shape)

    slope, intercept = np.polyfit(t_abs, corrected_mean, 1)
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
        consts.TRIMMED: int(fit_start_frame),
        consts.PB_FIT_START_FRAME: int(fit_start_frame),
        consts.PB_FIT_END_FRAME: int(fit_end_frame),
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
    pb_correct_dict["full_original_mean"] = full_mean_trace
    pb_correct_dict["corrected_mean"] = corrected_mean
    pb_correct_dict["frame_idx"] = frame_idx
    pb_correct_dict["time_sec"] = t_abs
    pb_correct_dict["fit_start_frame"] = int(fit_start_frame)
    pb_correct_dict["fit_end_frame"] = int(fit_end_frame)
    pb_correct_dict["corrected_mean_fit_line"] = slope * t_abs + intercept
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
        C=pb_correct_dict["C"],
        fit_start_frame=pb_correct_dict["fit_start_frame"],
        fit_end_frame=pb_correct_dict["fit_end_frame"]
    )


def save_pb_qc(pipeline_dir, pb_correct_dict, channel_name=None):
    qc_dir = os.path.join(pipeline_dir, consts.QC_DIR)
    pipe_utils.mkdir(qc_dir)

    suffix = "" if channel_name is None else f"_{channel_name}"
    qc_json_path = os.path.join(qc_dir, f"photobleaching_qc{suffix}.json")
    with open(qc_json_path, "w") as fp:
        json.dump(pb_correct_dict["qc_metrics"], fp, indent=2)

    frame_idx = pb_correct_dict["frame_idx"]
    original_mean = pb_correct_dict["original_mean"]
    full_original_mean = pb_correct_dict.get("full_original_mean")
    corrected_mean = pb_correct_dict["corrected_mean"]
    fitted_curve = pb_correct_dict["fitted_curve"]
    corrected_fit_line = pb_correct_dict["corrected_mean_fit_line"]
    fit_start = int(pb_correct_dict["qc_metrics"].get(consts.PB_FIT_START_FRAME, 0))
    fit_end = int(pb_correct_dict["qc_metrics"].get(consts.PB_FIT_END_FRAME, len(frame_idx) - 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    if full_original_mean is not None:
        full_frame_idx = np.arange(len(full_original_mean), dtype=np.int64)
        ax.plot(
            full_frame_idx,
            full_original_mean,
            label="Full mean intensity (pre-trim)",
            linewidth=1.0,
            alpha=0.35,
            color="gray",
        )

    ax.plot(frame_idx, original_mean, label="Original mean intensity", linewidth=1.5)
    ax.plot(frame_idx, fitted_curve, label="Fitted exponential", linewidth=1.5)
    ax.plot(frame_idx, corrected_mean, label="Corrected mean intensity", linewidth=1.5)
    ax.plot(frame_idx, corrected_fit_line, label="Corrected linear fit", linewidth=1.2, linestyle="--")
    ax.axvspan(fit_start, fit_end, color="tab:orange", alpha=0.12, label="Fit range")
    ax.axvline(fit_start, color="black", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.axvline(fit_end, color="black", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"Photobleaching QC (fit frames {fit_start}-{fit_end}, corrected full movie)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig_path = os.path.join(qc_dir, f"photobleaching_qc_plot{suffix}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main(args):
    gui_params_path = args[1]
    raw_video_path, mc_video_paths, channels, gui_time, fr, fit_start_frame, fit_end_frame = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)

    for ch in channels:
        mc_video_path = mc_video_paths[ch]
        print(f"Photobleaching correction on ({ch}):", mc_video_path)

        movie_clean, pb_correct_dict = run_photobleaching_correction(
            fr,
            fit_start_frame=fit_start_frame,
            fit_end_frame=fit_end_frame,
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
