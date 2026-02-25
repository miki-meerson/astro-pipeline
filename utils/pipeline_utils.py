import os
import re
import numpy as np
import xml.etree.ElementTree as ET
import tifffile
import xmltodict
from matplotlib.path import Path
import pandas as pd
import glob
from utils import pipeline_constants as consts


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.umask(0)
        os.makedirs(dir_name, mode=0o777, exist_ok=True)
    return

def windows_to_linux_path(path):
    """
    Convert path on windows OS to the corresponding cluster path.
    """
    if path is None:
        return path

    # Already a linux path on the cluster mount.
    if isinstance(path, str) and path.startswith("/ems/"):
        return path

    pattern = "Adam-Lab-Shared"
    path_prefix = '/ems/elsc-labs/adam-y/Adam-Lab-Shared/'
    match = re.search(pattern, path)
    if match:
        path_suffix = path[match.end():].replace('\\','/')
        return path_prefix + path_suffix

    # Fallback: normalize slashes without forcing remap if marker is missing.
    return str(path).replace("\\", "/")

def get_raw_video_dimensions(raw_path):
    """
    extract video's width and height fro, experiment.xml file -
    a file that attached to each video that is taken with ThorImage in AdamLab
    """
    xml = get_experiment_xml_path(raw_path)
    tree = ET.parse(xml)
    root = tree.getroot()
    width = int(root[5].attrib['width'])
    height = int(root[5].attrib['height'])
    return width, height

def get_experiment_xml_path(raw_path):
    """
    return the path of experiment.xml file -
    a file that attached to each video that is taken with ThorImage in AdamLab.
    the function assumes that the path is in the same directory of the raw video
    """
    return os.path.join(os.path.split(raw_path)[0],'Experiment.xml')

def get_frame_rate(raw_path):
    xml_path = get_experiment_xml_path(raw_path)
    xml_data = open(xml_path,"r").read()
    xml_dict = xmltodict.parse(xml_data)
    exposure_time = float(xml_dict["ThorImageExperiment"]["Camera"]["@exposureTimeMS"])
    fr = np.round(1000/exposure_time).astype(int)
    return fr


def get_pipeline_results_dir(raw_path):
    """
    return the path (and also create it if not exist) of the directory
    that will contain many outputs of the pipeline
    """
    pipeline_dir = os.path.join(os.path.split(raw_path)[0], consts.PIPELINE_DIR)
    mkdir(pipeline_dir)
    return pipeline_dir

def get_last_modified_file(dirname, suffix):
    list_of_files = glob.glob(dirname + '/*' + suffix) # * means all if need specific format then *.csv
    last_path = max(list_of_files, key=os.path.getctime) # take the last modified file
    return last_path

def get_mc_video(raw_video_path):
    pipeline_results = os.path.join(os.path.split(raw_video_path)[0], consts.PIPELINE_DIR)
    if not os.path.exists(pipeline_results):
        pipeline_results = os.path.join(os.path.split(raw_video_path)[0], consts.OLD_VOLPY_DIR) # changed from consts.VOLPY_DIR
    mc_dir = os.path.join(pipeline_results, consts.MC_DIR)
    mc_path = get_last_modified_file(mc_dir, consts.MC_VIDEO_PATH)
    return mc_path

def get_denoised_path(fnames, gui_time):
    home = os.path.split(os.path.split(fnames)[0])[0]
    denoise_dir = os.path.join(home,'denoiser_files')
    deepinterpolation_dir = os.path.join(denoise_dir,'deepinterpolation')
    deepvid_dir = os.path.join(denoise_dir,'deepvid')
    gui_time_dir = os.path.join(deepinterpolation_dir, gui_time)
    if not os.path.exists(gui_time_dir):
        gui_time_dir = os.path.join(deepvid_dir, gui_time)
    denoised_file = os.path.join(gui_time_dir, "denoised_no_pad.tif")
    if os.path.isfile(denoised_file): # if merge done right after pipeline finished
        return denoised_file
    # else take the last file in denoise_dir
    list_of_files = glob.glob(denoise_dir + '/**/*.tif', recursive=True)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def raw_to_tif(raw_path, start_frame=None, end_frame=None):
    """
    convert the raw formatted vodeo file to a tif file and save it in pipeline results dir.
    find width and height in the experiment.xml file.
    if start and end frame supplied - save sliced video based on those values.
    """
    pipeline_dir = get_pipeline_results_dir(raw_path)
    width, height = get_raw_video_dimensions(raw_path)
    rawmovie_1d = np.fromfile(raw_path, dtype=np.uint16)
    movie_3d = np.reshape(rawmovie_1d,(-1,height,width))

    tif_video_path = os.path.join(pipeline_dir, consts.RAW_VIDEO_TIF)
    tifffile.imwrite(tif_video_path, movie_3d, bigtiff=True)
    return tif_video_path

def get_rois_mask_old(raw_video_path):
    """
    for given raw video path, looking for the "ROIs.xaml" file from ThorImage
    and generate binary mask as np array in the sahpe of (#ROIs, height, width)
    """
    xml_path = os.path.join(os.path.split(raw_video_path)[0],'ROIs.xaml')
    xml_data = open(xml_path, "r").read()
    xml_dict = xmltodict.parse(xml_data)
    polygons_struct = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIPoly"]
    # extract polygons of ROIS.
    poly_lst = []
    for i in range(len(polygons_struct)):
        p = polygons_struct[i]['@Points']
        if p not in poly_lst:
            poly_lst.append(p)
    # extract the coordinates of the rectangle ROI
    rect_data = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIRect"]
    bottom_left_x, bottom_left_y = [float(i) for i in rect_data["@BottomLeft"].split(',')]
    top_left_x, top_left_y = [float(i) for i in rect_data["@TopLeft"].split(',')]
    height = float(rect_data["@ROIHeight"])
    width = float(rect_data["@ROIWidth"])
    # generate list of polygons w.r.t the rectangle ROI
    corrected_polygons = []
    for polygon in poly_lst: # for each polygon
        corrected_points = []
        points = polygon.split(' ')
        for point in points:
            x, y = [float(i) for i in point.split(',')]
            # if the point exceeds the rectangle from above, left or right - trunc it
            x = min(max(x - bottom_left_x, 1),width)
            y = max(1,min(y - top_left_y, height))
            corrected_points.append((x, y))
        corrected_points.append(corrected_points[0])
        corrected_polygons.append(corrected_points)
    # generate masks
    width, height = get_raw_video_dimensions(raw_video_path)
    ROIs = []
    for poly in corrected_polygons:
        flipped_poly = [(j,i) for i,j in poly]
        polygon = flipped_poly
        poly_path = Path(polygon)
        x, y = np.mgrid[:height, :width]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
        mask = poly_path.contains_points(coors)
        mask = mask.reshape(height, width)
        if mask.sum() > 0: # fot the case that a point was signed in the slm
            ROIs.append(mask)
    ROIs = np.stack(ROIs)
    return ROIs

def get_rois_mask(raw_video_path):
    """
    For given raw video path, look for ThorImage "ROIs.xaml"
    and generate binary mask np array of shape (#ROIs, height, width).

    Robust to:
      - ROIRect missing (corrupt ROIs.xaml)

    NOTE: This version assumes ROIs.xaml uses the same keys as your original parser:
      xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIPoly"]
    """
    import os
    import numpy as np
    import xmltodict
    from matplotlib.path import Path

    xml_path = os.path.join(os.path.split(raw_video_path)[0], "ROIs.xaml")
    xml_data = open(xml_path, "r").read()
    xml_dict = xmltodict.parse(xml_data)

    # --- Collect ROIPoly entries robustly (handles 1 or many) ---
    polygons_struct = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIPoly"]
    if isinstance(polygons_struct, dict):
        polygons_struct = [polygons_struct]

    poly_lst = []
    for p in polygons_struct:
        pts = p.get("@Points", "")
        if pts and pts not in poly_lst:
            poly_lst.append(pts)

    # remove point ROIs / degenerate polygons
    poly_lst = [s for s in poly_lst if len(s.split()) >= 3]
    if not poly_lst:
        raise ValueError("No valid ROIPoly points found in ROIs.xaml")

    def parse_points_str(points_str: str) -> np.ndarray:
        pts = []
        for token in points_str.strip().split():
            x_str, y_str = token.split(",")
            pts.append((float(x_str), float(y_str)))
        return np.array(pts, dtype=float)  # (P,2) [x,y]

    # --- Try normal ROIRect path ---
    try:
        rect_data = xml_dict["ROICapsule"]["ROICapsule.ROIs"]["x:Array"]["ROIRect"]
    except Exception:
        rect_data = None

    # Target mask size (video dimensions)
    width_vid, height_vid = get_raw_video_dimensions(raw_video_path)

    corrected_polygons = []

    if rect_data is not None:
        # Normal case (your original logic)
        bottom_left_x, _bottom_left_y = [float(i) for i in rect_data["@BottomLeft"].split(",")]
        _top_left_x, top_left_y = [float(i) for i in rect_data["@TopLeft"].split(",")]
        rect_h = float(rect_data["@ROIHeight"])
        rect_w = float(rect_data["@ROIWidth"])

        for polygon in poly_lst:
            pts = parse_points_str(polygon)
            x = pts[:, 0]
            y = pts[:, 1]

            x = np.clip(x - bottom_left_x, 1, rect_w)
            y = np.clip(y - top_left_y, 1, rect_h)

            corrected = np.column_stack([x, y])
            corrected = np.vstack([corrected, corrected[0]])  # close
            corrected_polygons.append([tuple(t) for t in corrected])

    else:
        # Fallback: ROIRect missing (corrupt files) => scale-to-fit to video frame
        parsed = [parse_points_str(s) for s in poly_lst]
        all_pts = np.vstack(parsed) if parsed else np.zeros((0, 2))
        if all_pts.shape[0] == 0:
            raise ValueError("No valid ROIPoly points found in ROIs.xaml")

        minX, minY = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
        maxX, maxY = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

        denom_x = max(1e-12, (maxX - minX))
        denom_y = max(1e-12, (maxY - minY))
        sx = (width_vid - 1) / denom_x
        sy = (height_vid - 1) / denom_y

        for pts in parsed:
            x = (pts[:, 0] - minX) * sx + 1
            y = (pts[:, 1] - minY) * sy + 1

            x = np.clip(x, 1, width_vid)
            y = np.clip(y, 1, height_vid)

            corrected = np.column_stack([x, y])
            corrected = np.vstack([corrected, corrected[0]])  # close
            corrected_polygons.append([tuple(t) for t in corrected])

    # --- Generate masks ---
    ROIs = []
    for poly in corrected_polygons:
        flipped_poly = [(j, i) for i, j in poly]  # (row, col)
        poly_path = Path(flipped_poly)

        x, y = np.mgrid[:height_vid, :width_vid]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = poly_path.contains_points(coors).reshape(height_vid, width_vid)

        if mask.sum() > 0:  # ignore empty masks (e.g., point ROIs)
            ROIs.append(mask)

    if not ROIs:
        return np.zeros((0, height_vid, width_vid), dtype=bool)

    return np.stack(ROIs)

def trace_extraction(video, rois_mask, weights=None):
    """
    video - 3d np array represent a video.
    rois - a binary np array in the shape of (#cells, width, height).
            its represent the pixels corresponding to each cell in the video.
    weights - represent spatial components to extract the traces accordingly.
            if not supplied - just preform non-weighted mean over the cell
    """
    if weights is None:
        weights = rois_mask
    df_columns = ['cell ' + str(i+1) for i in range(len(rois_mask))]
    df = pd.DataFrame(columns=df_columns)
    for roi_num in range(len(rois_mask)):
        Xinds = np.where(np.any(rois_mask[roi_num] > 0, axis=1) > 0)[0]
        Yinds = np.where(np.any(rois_mask[roi_num] > 0, axis=0) > 0)[0]
        croped_video = video[:, Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1]
        cell_mask = weights[roi_num]
        croped_mask = cell_mask[Xinds[0]:Xinds[-1] + 1, Yinds[0]:Yinds[-1] + 1]
        masked_video = croped_video * croped_mask[np.newaxis,:,:]
        trace = masked_video.mean(axis=(1, 2))
        df[df.columns[roi_num]] = trace
    return df

def get_video_details(path):
    """
    given raw video path, the function look after the cage and mouse name
    """
    try:
        dir_parts = os.path.normpath(os.path.dirname(path)).replace("\\", "/").split("/")
        cage, mouse, date_behavior, fov, exp_details = dir_parts[-5:]
        date, behavior = date_behavior.rsplit("-", 1)

        return cage, mouse, fov, date, behavior, exp_details
    except:
        return None, None, None, None, None, None

