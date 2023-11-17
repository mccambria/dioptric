# -*- coding: utf-8 -*-
"""Various utility functions for widefield imaging and camera data processing

Created on August 15th, 2023

@author: mccambria
"""

# region Imports and constants

import matplotlib.pyplot as plt
import numpy as np
from numpy import inf
from utils import common
from utils import tool_belt as tb
from utils import positioning as pos
from utils import kplotlib as kpl
from utils.constants import CountFormat
from utils.constants import CollectionMode, LaserKey, LaserPosMode

# endregion
# region Plotting


def imshow(ax, img_array, count_format=None, **kwargs):
    """Version of kplotlib's imshow with additional defaults for a camera"""

    prev_font_size = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": kpl.FontSize.SMALL})

    config = common.get_config_dict()
    if count_format is None:
        count_format = config["count_format"]
    if count_format == CountFormat.RAW:
        cbar_label = "Counts"
    if count_format == CountFormat.KCPS:
        cbar_label = "Kcps"
    default_kwargs = {
        "cbar_label": cbar_label,
    }
    passed_kwargs = {**default_kwargs, **kwargs}
    kpl.imshow(ax, img_array, **passed_kwargs)

    plt.rcParams.update({"font.size": prev_font_size})


# endregion
# region Image processing


def img_str_to_array(img_str):
    config = common.get_config_dict()
    resolution = config["Camera"]["resolution"]
    img_array = np.frombuffer(img_str, dtype=np.uint16).reshape(*resolution)
    img_array = img_array.astype(int)
    return img_array


def counts_from_img_array(
    img_array, pixel_coords, radius=None, drift_adjust=True, pixel_drift=None
):
    # Make copies so we don't mutate the originals
    pixel_coords = pixel_coords.copy()
    if drift_adjust:
        pixel_coords = adjust_pixel_coords_for_drift(pixel_coords, pixel_drift)

    if radius is None:
        config = common.get_config_dict()
        radius = config["camera_spot_radius"]

    if type(radius) is list:
        max_radius = np.max(radius)
    else:
        max_radius = radius

    def check_dist(dist):
        if type(radius) is list:
            radius_pair = radius[0]
            ret_vals = (radius_pair[0] < dist) * (dist < radius_pair[1])
            for ind in range(len(radius)):
                radius_pair = radius[ind]
                ret_vals += (radius_pair[0] < dist) * (dist < radius_pair[1])
            return ret_vals
        else:
            return dist < radius

    if drift_adjust:
        if pixel_drift is None:
            pixel_drift = get_pixel_drift()
        pixel_x = pixel_coords[0] + pixel_drift[0]
        pixel_y = pixel_coords[1] + pixel_drift[1]
    else:
        pixel_x = pixel_coords[0]
        pixel_y = pixel_coords[1]

    # Don't work through all the pixels, just the ones that might be relevant
    left = int(np.floor(pixel_x - max_radius))
    right = int(np.ceil(pixel_x + max_radius))
    top = int(np.floor(pixel_y - max_radius))
    bottom = int(np.ceil(pixel_y + max_radius))
    x_crop = np.linspace(left, right, right - left + 1)
    y_crop = np.linspace(top, bottom, bottom - top + 1)
    x_crop_mesh, y_crop_mesh = np.meshgrid(x_crop, y_crop)
    img_array_crop = img_array[top : bottom + 1, left : right + 1]
    dist = np.sqrt((x_crop_mesh - pixel_x) ** 2 + (y_crop_mesh - pixel_y) ** 2)
    # inner_pixels = np.where(dist < radius, img_array_crop, np.nan)
    inner_pixels = np.where(check_dist(dist), img_array_crop, np.nan)
    inner_pixels = inner_pixels.flatten()
    inner_pixels = inner_pixels[~np.isnan(inner_pixels)]

    clamp = 300
    total_clamp = clamp * len(inner_pixels)
    counts = np.sum(inner_pixels) - total_clamp

    return counts


# endregion
# region Drift tracking


def get_pixel_drift():
    pixel_drift = common.get_registry_entry(["State"], "PIXEL_DRIFT")
    return np.array(pixel_drift)


def set_pixel_drift(drift):
    return common.set_registry_entry(["State"], "PIXEL_DRIFT", drift)


def reset_pixel_drift():
    return set_pixel_drift([0.0, 0.0])


def reset_all_drift():
    reset_pixel_drift()
    pos.reset_drift()
    scanning_optics = _get_scanning_optics()
    for coords_suffix in scanning_optics:
        pos.reset_drift(coords_suffix)


def adjust_pixel_coords_for_drift(pixel_coords, drift=None):
    """Current drift will be retrieved from registry if passed drift is None"""
    if drift is None:
        drift = get_pixel_drift()
    adjusted_coords = (np.array(pixel_coords) + np.array(drift)).tolist()
    return adjusted_coords


def get_nv_pixel_coords(nv_sig, drift_adjust=True):
    pixel_coords = nv_sig["pixel_coords"]
    if drift_adjust:
        pixel_coords = adjust_pixel_coords_for_drift(pixel_coords)
    return pixel_coords


def set_all_scanning_drift_from_pixel_drift(pixel_drift=None):
    scanning_optics = _get_scanning_optics()
    for coords_suffix in scanning_optics:
        set_scanning_drift_from_pixel_drift(pixel_drift, coords_suffix)


def _get_scanning_optics():
    config = common.get_config_dict()
    config_optics = config["Optics"]
    scanning_optics = []
    for optic_name in config_optics:
        optic_dict = config_optics[optic_name]
        if "pos_mode" in optic_dict and optic_dict["pos_mode"] == LaserPosMode.SCANNING:
            scanning_optics.append(optic_name)
    return scanning_optics

def set_scanning_drift_from_pixel_drift(pixel_drift=None, coords_suffix=None):
    scanning_drift = pixel_to_scanning_drift(pixel_drift, coords_suffix)
    pos.set_drift(scanning_drift, coords_suffix)


def set_pixel_drift_from_scanning_drift(scanning_drift=None, coords_suffix=None):
    pixel_drift = scanning_to_pixel_drift(scanning_drift, coords_suffix)
    set_pixel_drift(pixel_drift)


def pixel_to_scanning_drift(pixel_drift=None, coords_suffix=None):
    if pixel_drift is None:
        pixel_drift = get_pixel_drift()
    m_x, _, m_y, _ = _pixel_to_scanning_calibration(coords_suffix)
    scanning_drift = pos.get_drift(coords_suffix)
    if len(scanning_drift) > 2:
        z_scanning_drift = scanning_drift[2]
        return [m_x * pixel_drift[0], m_y * pixel_drift[1], z_scanning_drift]
    else:
        return [m_x * pixel_drift[0], m_y * pixel_drift[1]]


def scanning_to_pixel_drift(scanning_drift=None, coords_suffix=None):
    if scanning_drift is None:
        scanning_drift = pos.get_drift(coords_suffix)
    m_x, _, m_y, _ = _scanning_to_pixel_calibration(coords_suffix)
    return [m_x * scanning_drift[0], m_y * scanning_drift[1]]


# endregion
# region Scanning to pixel calibration


def set_nv_scanning_coords_from_pixel_coords(nv_sig, coords_suffix=None):
    pixel_coords = get_nv_pixel_coords(nv_sig)
    red_coords = pixel_to_scanning_coords(pixel_coords, coords_suffix)
    pos.set_nv_coords(nv_sig, red_coords, coords_suffix)
    return red_coords


def get_widefield_calibration_nvs():
    module = common.get_config_module()
    nv1 = module.widefield_calibration_nv1.copy()
    nv2 = module.widefield_calibration_nv2.copy()
    return nv1, nv2


def pixel_to_scanning_coords(pixel_coords, coords_suffix=None):
    """Convert camera pixel coordinates to scanning coordinates (e.g. galvo voltages)
    using two calibrated NV coordinate pairs from the config file

    Parameters
    ----------
    pixel_coords : list(numeric)
        Camera pixel coordinates to convert

    Returns
    -------
    list(numeric)
        Scanning coordinates
    """
    m_x, b_x, m_y, b_y = _pixel_to_scanning_calibration(coords_suffix)
    scanning_coords = [m_x * pixel_coords[0] + b_x, m_y * pixel_coords[1] + b_y]
    return scanning_coords


def _pixel_to_scanning_calibration(coords_suffix=None):
    """Get the linear parameters for the conversion"""

    nv1, nv2 = get_widefield_calibration_nvs()
    nv1_scanning_coords = pos.get_nv_coords(nv1, coords_suffix, drift_adjust=False)
    nv1_pixel_coords = nv1["pixel_coords"]
    nv2_scanning_coords = pos.get_nv_coords(nv2, coords_suffix, drift_adjust=False)
    nv2_pixel_coords = nv2["pixel_coords"]

    # Assume (independent) linear relations for both x and y

    scanning_diff = nv2_scanning_coords[0] - nv1_scanning_coords[0]
    pixel_diff = nv2_pixel_coords[0] - nv1_pixel_coords[0]
    m_x = scanning_diff / pixel_diff
    b_x = nv1_scanning_coords[0] - m_x * nv1_pixel_coords[0]

    scanning_diff = nv2_scanning_coords[1] - nv1_scanning_coords[1]
    pixel_diff = nv2_pixel_coords[1] - nv1_pixel_coords[1]
    m_y = scanning_diff / pixel_diff
    b_y = nv1_scanning_coords[1] - m_y * nv1_pixel_coords[1]

    return m_x, b_x, m_y, b_y


def scanning_to_pixel_coords(scanning_coords, coords_suffix=None):
    """Convert scanning coordinates (e.g. galvo voltages) to camera pixel coordinates
    using two calibrated NV coordinate pairs from the config file

    Parameters
    ----------
    scanning_coords : list(numeric)
        Scanning coordinates to convert

    Returns
    -------
    list(numeric)
        Camera pixel coordinates
    """

    m_x, b_x, m_y, b_y = _scanning_to_pixel_calibration(coords_suffix)
    pixel_coords = [m_x * scanning_coords[0] + b_x, m_y * scanning_coords[1] + b_y]
    return pixel_coords


def _scanning_to_pixel_calibration(coords_suffix=None):
    """Get the linear parameters for the conversion"""

    nv1, nv2 = get_widefield_calibration_nvs()
    nv1_scanning_coords = pos.get_nv_coords(nv1, coords_suffix, drift_adjust=False)
    nv1_pixel_coords = nv1["pixel_coords"]
    nv2_scanning_coords = pos.get_nv_coords(nv2, coords_suffix, drift_adjust=False)
    nv2_pixel_coords = nv2["pixel_coords"]

    # Assume (independent) linear relations for both x and y

    pixel_diff = nv2_pixel_coords[0] - nv1_pixel_coords[0]
    scanning_diff = nv2_scanning_coords[0] - nv1_scanning_coords[0]
    m_x = pixel_diff / scanning_diff
    b_x = nv1_pixel_coords[0] - m_x * nv1_scanning_coords[0]

    pixel_diff = nv2_pixel_coords[1] - nv1_pixel_coords[1]
    scanning_diff = nv2_scanning_coords[1] - nv1_scanning_coords[1]
    m_y = pixel_diff / scanning_diff
    b_y = nv1_pixel_coords[1] - m_y * nv1_scanning_coords[1]

    return m_x, b_x, m_y, b_y


# def set_calibration_coords(
#     nv1_pixel_coords, nv1_scanning_coords, nv2_pixel_coords, nv2_scanning_coords
# ):
#     calibration_directory = ["State", "WidefieldCalibration"]
#     pixel_coords_list = [nv1_pixel_coords, nv2_pixel_coords]
#     scanning_coords_list = [nv1_scanning_coords, nv2_scanning_coords]

#     nv_names = ["NV1", "NV2"]
#     for ind in range(2):
#         nv_name = nv_names[ind]
#         key = f"{nv_name}_PIXEL_COORDS"
#         pixel_coords = pixel_coords_list[ind]
#         common.set_registry_entry(calibration_directory, key, pixel_coords)
#         key = f"{nv_name}_SCANNING_COORDS"
#         scanning_coords = scanning_coords_list[ind]
#         common.set_registry_entry(calibration_directory, key, scanning_coords)

# endregion


if __name__ == "__main__":
    pixel_drift = [-5.56, 5.64]
    test = pixel_to_scanning_drift(pixel_drift)
    print(test)
