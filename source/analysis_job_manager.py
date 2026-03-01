import datetime
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import streamlit as st
import tifffile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import pipeline_constants as consts
from utils import files_paths as paths
from utils import pipeline_utils as pipe_utils
from utils import pipeline_registry as steps_registry
from source import step_manager
from source.pipeline_steps import pca_computation as pca_step


ANALYSIS_STATUS_STEP_KEY = "analysis_status_step_key"
ANALYSIS_STATUS_JOB_ID_KEY = "analysis_status_job_id"
ANALYSIS_STATUS_LOG_FILE_KEY = "analysis_status_log_file"
ANALYSIS_STATUS_PARAMS_PATH_KEY = "analysis_status_params_path"
ANALYSIS_STATUS_STATE_KEY = "analysis_status_state"
ANALYSIS_STATUS_MODE_KEY = "analysis_status_mode"
ANALYSIS_STATUS_MESSAGE_KEY = "analysis_status_message"
ANALYSIS_STATUS_MESSAGE_LEVEL_KEY = "analysis_status_message_level"


def on_analysis_video_path_change():
    analysis_video_path = st.session_state.get(consts.ANALYSIS_VIDEO_PATH, "")
    st.session_state[consts.HOME_DIR] = os.path.dirname(analysis_video_path) if analysis_video_path else ""


def display_pca_params(gui_parameter_cls, number_input_token, pca_params_title):
    with st.expander(pca_params_title):
        gui_parameter_cls(consts.SPATIAL_BIN_FACTOR, number_input_token, 4)
        gui_parameter_cls(consts.PCA_K_KEEP, number_input_token, 10, display_name="PCA components to visualize / keep")
        gui_parameter_cls(consts.ICA_N_COMPONENTS, number_input_token, 6, display_name="ICA components")


def save_analysis_params(session_time, get_gui_params_fn):
    gui_params = get_gui_params_fn()
    gui_params[consts.GUI_TIME] = datetime.datetime.now().strftime("%d-%m-%Y___%H-%M-%S")

    analysis_video_path = st.session_state.get(consts.ANALYSIS_VIDEO_PATH, "")
    if not os.path.isfile(analysis_video_path):
        raise FileNotFoundError("Select a valid processed movie file.")

    home_dir = os.path.dirname(analysis_video_path)
    if not home_dir or not os.path.isdir(home_dir):
        raise ValueError("Could not determine a valid home directory from the selected movie.")

    gui_params[consts.HOME_DIR] = home_dir
    gui_params[consts.HOME_DIR_LINUX] = pipe_utils.windows_to_linux_path(home_dir)
    gui_params[consts.ANALYSIS_VIDEO_PATH] = analysis_video_path

    raw_video_path = st.session_state.get(consts.RAW_VIDEO_PATH, "")
    if raw_video_path:
        gui_params[consts.RAW_VIDEO_PATH] = raw_video_path
        gui_params[consts.RAW_VIDEO_PATH_LINUX] = pipe_utils.windows_to_linux_path(raw_video_path)
    else:
        gui_params[consts.RAW_VIDEO_PATH] = analysis_video_path
        gui_params[consts.RAW_VIDEO_PATH_LINUX] = gui_params[consts.ANALYSIS_VIDEO_PATH]

    gui_params[consts.SPATIAL_BIN_FACTOR] = int(st.session_state.get(consts.SPATIAL_BIN_FACTOR, 4))
    gui_params[consts.PCA_K_KEEP] = int(st.session_state.get(consts.PCA_K_KEEP, 10))
    gui_params[consts.ICA_N_COMPONENTS] = int(st.session_state.get(consts.ICA_N_COMPONENTS, 6))

    analysis_dir = os.path.join(paths.PIPELINE_LOGS_DIR, session_time, "analysis_jobs")
    pipe_utils.mkdir(analysis_dir)

    param_file_path = os.path.join(analysis_dir, gui_params[consts.GUI_TIME] + consts.PARAMS_FILE_SUFFIX_NAME)
    with open(param_file_path, "w") as fp:
        print("Saved gui_params to {}".format(param_file_path))
        json.dump(gui_params, fp, indent=4)
    return param_file_path


def submit_analysis_job(step_name, step_properties, params_path):
    if step_name == consts.PCA_COMPUTATION:
        local_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pipeline_steps",
            "pca_computation.py",
        )
        if not os.path.isfile(local_script_path):
            raise FileNotFoundError(f"Analysis script not found: {local_script_path}")

        log_file = os.path.join(
            os.path.dirname(params_path),
            f"{step_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        log_handle = open(log_file, "w", encoding="utf-8")

        process = subprocess.Popen(
            [sys.executable, local_script_path, params_path],
            cwd=os.path.dirname(local_script_path),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log_handle.close()

        return SimpleNamespace(
            job_id=str(process.pid),
            log_file=log_file,
            mode="local",
        )

    job = step_manager.ClusterJob(
        step_properties["script"],
        pipe_utils.windows_to_linux_path(params_path),
    )
    job.run_job()
    job.mode = "cluster"
    return job


def _get_analysis_step_label(step_name):
    step_properties = steps_registry.ANALYSIS_STEPS_REGISTRY.get(step_name, {})
    return step_properties.get("display_name", step_name)


def _render_analysis_status_message():
    status_message = st.session_state.get(ANALYSIS_STATUS_MESSAGE_KEY, "")
    status_message_level = st.session_state.get(ANALYSIS_STATUS_MESSAGE_LEVEL_KEY, "info")
    if not status_message:
        st.caption("Analysis status: no analysis job submitted yet.")
        return
    if status_message_level == "error":
        st.error(status_message)
    elif status_message_level == "success":
        st.success(status_message)
    else:
        st.info(status_message)


def _is_local_process_running(pid):
    if not pid:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "").strip()
        return bool(output) and "No tasks are running" not in output
    else: 
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False


def _read_log_file(log_file):
    if not log_file or not os.path.isfile(log_file):
        return ""
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _display_local_analysis_status(step_name, job_id, last_known_state):
    log_file = st.session_state.get(ANALYSIS_STATUS_LOG_FILE_KEY, "")
    step_label = _get_analysis_step_label(step_name)
    if _is_local_process_running(job_id):
        if last_known_state == "started":
            st.info(f"{step_label} status: started (process {job_id})")
        else:
            st.session_state[ANALYSIS_STATUS_STATE_KEY] = "in progress"
            st.info(f"{step_label} status: in progress (process {job_id})")
        return

    logs = _read_log_file(log_file)
    if consts.STEP_COMPLETED in logs:
        st.session_state[ANALYSIS_STATUS_STATE_KEY] = "finished"
        st.success(f"{step_label} status: finished (process {job_id})")
        return

    st.session_state[ANALYSIS_STATUS_STATE_KEY] = "failed"
    st.error(f"{step_label} status: failed (process {job_id})")
    if logs.strip():
        st.code(logs, language="text")


def display_analysis_status():
    step_name = st.session_state.get(ANALYSIS_STATUS_STEP_KEY, "")
    job_id = st.session_state.get(ANALYSIS_STATUS_JOB_ID_KEY)
    if not job_id:
        _render_analysis_status_message()
        return

    step_label = _get_analysis_step_label(step_name)
    last_known_state = st.session_state.get(ANALYSIS_STATUS_STATE_KEY, "started")
    if st.session_state.get(ANALYSIS_STATUS_MODE_KEY) == "local":
        _display_local_analysis_status(step_name, job_id, last_known_state)
        return

    step_properties = steps_registry.ANALYSIS_STEPS_REGISTRY.get(step_name)
    if not step_properties:
        st.error(f"Unknown analysis step: {step_name}")
        return

    params_path = st.session_state.get(ANALYSIS_STATUS_PARAMS_PATH_KEY, "")
    job = step_manager.ClusterJob(
        step_properties["script"],
        pipe_utils.windows_to_linux_path(params_path) if params_path else "",
    )
    job.job_id = str(job_id)
    job.log_file = st.session_state.get(ANALYSIS_STATUS_LOG_FILE_KEY)

    try:
        state = job.update_state_2()
    except Exception as e:
        if last_known_state == "finished":
            st.success(f"{step_label} status: finished (job {job_id})")
        elif last_known_state == "failed":
            st.error(f"{step_label} status: failed (job {job_id})")
        elif last_known_state == "in progress":
            st.info(f"{step_label} status: in progress (job {job_id})")
        else:
            st.info(f"{step_label} status: started (job {job_id})")
            st.caption(f"Status refresh unavailable: {e}")
        return

    if state in (consts.JOB_PENDING, consts.NOT_STARTED):
        st.session_state[ANALYSIS_STATUS_STATE_KEY] = "started"
        st.info(f"{step_label} status: started (job {job_id})")
    elif state == consts.JOB_RUNNING:
        st.session_state[ANALYSIS_STATUS_STATE_KEY] = "in progress"
        st.info(f"{step_label} status: in progress (job {job_id})")
    elif state == consts.JOB_FINISHED:
        st.session_state[ANALYSIS_STATUS_STATE_KEY] = "finished"
        st.success(f"{step_label} status: finished (job {job_id})")
    elif state in (consts.JOB_FAILED, consts.JOB_CANCELLD):
        st.session_state[ANALYSIS_STATUS_STATE_KEY] = "failed"
        st.error(f"{step_label} status: failed (job {job_id})")
    else:
        st.caption(f"{step_label} status: {state} (job {job_id})")


def _get_analysis_pipeline_dir():
    analysis_video_path = st.session_state.get(consts.ANALYSIS_VIDEO_PATH, "")
    if not analysis_video_path:
        return ""
    parent_dir = os.path.dirname(analysis_video_path)
    if os.path.basename(parent_dir) == consts.PB_DIR:
        return os.path.dirname(parent_dir)
    return parent_dir


def _load_npz_dict(path):
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=False)
    try:
        return {key: data[key] for key in data.files}
    finally:
        data.close()


def _save_npz_dict(path, data_dict):
    np.savez(path, **data_dict)



def display_analysis_results():
    pipeline_dir = _get_analysis_pipeline_dir()
    if not pipeline_dir:
        st.caption("Select an analysis movie to inspect PCA/ICA outputs.")
        return

    pca_dir = os.path.join(pipeline_dir, consts.PCA_DIR)
    if not os.path.isdir(pca_dir):
        st.caption("No PCA results found yet.")
        return

    pca_params = _load_npz_dict(os.path.join(pca_dir, consts.PCA_PARAMS_PATH))
    ica_params = _load_npz_dict(os.path.join(pca_dir, consts.ICA_PARAMS_PATH))
    ica_components = _load_npz_dict(os.path.join(pca_dir, consts.ICA_COMPONENTS_PATH))

    if pca_params:
        binned_size = tuple(int(v) for v in np.asarray(pca_params["binned_movie_size"]).tolist())
        st.caption(
            "Binned movie size: "
            f"{binned_size} | bin factor: {int(pca_params['spatial_bin_factor'])} | "
            f"k_keep: {int(pca_params.get('pca_k_keep', 0))} | "
            f"ICA components: {int(pca_params.get('ica_n_components', 0))}"
        )
    elif ica_params:
        st.caption(
            f"k_keep: {int(ica_params.get('pca_k_keep', 0))} | "
            f"ICA components: {int(ica_params.get('ica_n_components', 0))}"
        )

    if ica_components:
        st.caption(
            f"Saved ICA arrays: time {tuple(ica_components['ics_time'].shape)}, "
            f"space {tuple(ica_components['ics_space'].shape)}"
        )

    singular_values_path = os.path.join(pca_dir, "singular_values.png")
    pca_overview_path = os.path.join(pca_dir, consts.PCA_OVERVIEW_PATH)
    ica_overview_path = os.path.join(pca_dir, consts.ICA_OVERVIEW_PATH)

    with st.expander("PCA Components", expanded=True):
        if os.path.isfile(singular_values_path):
            st.image(singular_values_path, caption="PCA variance explained", width=700)
        if os.path.isfile(pca_overview_path):
            st.image(pca_overview_path, caption="PCA spatial and temporal components", width=700)

    with st.expander("ICA Components", expanded=False):
        if os.path.isfile(ica_overview_path):
            st.image(ica_overview_path, caption="ICA spatial and temporal components", width=700)


def display_analysis_buttons(session_time, get_gui_params_fn):
    cols = st.columns(len(steps_registry.ANALYSIS_STEPS_REGISTRY) + 1)

    with cols[0]:
        st.markdown("**_Analysis steps:_**")

    for i, (step_name, step_properties) in enumerate(steps_registry.ANALYSIS_STEPS_REGISTRY.items(), start=1):
        with cols[i]:
            if st.button(step_properties["display_name"], key=f"run_analysis_{i}"):
                try:
                    params_path = save_analysis_params(session_time, get_gui_params_fn)
                    job = submit_analysis_job(step_name, step_properties, params_path)
                    st.session_state[ANALYSIS_STATUS_STEP_KEY] = step_name
                    st.session_state[ANALYSIS_STATUS_JOB_ID_KEY] = job.job_id
                    st.session_state[ANALYSIS_STATUS_LOG_FILE_KEY] = job.log_file
                    st.session_state[ANALYSIS_STATUS_PARAMS_PATH_KEY] = params_path
                    st.session_state[ANALYSIS_STATUS_STATE_KEY] = "started"
                    st.session_state[ANALYSIS_STATUS_MODE_KEY] = getattr(job, "mode", "cluster")
                    st.session_state[ANALYSIS_STATUS_MESSAGE_KEY] = ""
                    st.session_state[ANALYSIS_STATUS_MESSAGE_LEVEL_KEY] = "info"
                    st.success(f"Submitted {step_properties['display_name']} (job {job.job_id})")
                except Exception as e:
                    st.session_state[ANALYSIS_STATUS_STEP_KEY] = step_name
                    st.session_state[ANALYSIS_STATUS_JOB_ID_KEY] = ""
                    st.session_state[ANALYSIS_STATUS_LOG_FILE_KEY] = ""
                    st.session_state[ANALYSIS_STATUS_PARAMS_PATH_KEY] = ""
                    st.session_state[ANALYSIS_STATUS_STATE_KEY] = "failed"
                    st.session_state[ANALYSIS_STATUS_MODE_KEY] = ""
                    st.session_state[ANALYSIS_STATUS_MESSAGE_KEY] = (
                        f"Failed to submit {step_properties['display_name']}: {e}"
                    )
                    st.session_state[ANALYSIS_STATUS_MESSAGE_LEVEL_KEY] = "error"
                    st.error(st.session_state[ANALYSIS_STATUS_MESSAGE_KEY])
