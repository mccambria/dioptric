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

# endregion


def imshow(ax, img_array, count_format=None, **kwargs):
    """Version of kplotlib's imshow with additional defaults for a camera"""

    config = common.get_config_dict()
    if count_format is None:
        count_format = config["count_format"]
    if count_format == CountFormat.RAW:
        cbar_label = "Counts"
    if count_format == CountFormat.KCPS:
        cbar_label = "Kcps"
    default_kwargs = {
        "x_label": "X",
        "y_label": "Y",
        "cbar_label": cbar_label,
    }
    passed_kwargs = {**default_kwargs, **kwargs}
    kpl.imshow(ax, img_array, **passed_kwargs)


def get_widefield_calibration_params():
    directory = ["State", "WidefieldCalibration"]

    # Get the last drift trackers
    last_scanning_drift = common.get_registry_entry(directory, "DRIFT")
    last_pixel_drift = common.get_registry_entry(directory, "PIXEL_DRIFT")

    # Get the NVs lastr coordinates
    nv1_scanning_coords = common.get_registry_entry(directory, "NV1_SCANNING_COORDS")
    nv1_pixel_coords = common.get_registry_entry(directory, "NV1_PIXEL_COORDS")
    nv2_scanning_coords = common.get_registry_entry(directory, "NV2_SCANNING_COORDS")
    nv2_pixel_coords = common.get_registry_entry(directory, "NV2_PIXEL_COORDS")

    return (
        nv1_scanning_coords,
        nv1_pixel_coords,
        nv2_scanning_coords,
        nv2_pixel_coords,
        last_scanning_drift,
        last_pixel_drift,
    )


def pixel_to_scanning_coords(pixel_coords):
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
    m_x, b_x, m_y, b_y = _pixel_to_scanning_coords()
    scanning_coords = [m_x * pixel_coords[0] + b_x, m_y * pixel_coords[1] + b_y]
    return scanning_coords


def _pixel_to_scanning_coords():
    """Get the linear parameters for the conversion"""

    ret_vals = get_widefield_calibration_params()
    (
        nv1_scanning_coords,
        nv1_pixel_coords,
        nv2_scanning_coords,
        nv2_pixel_coords,
    ) = ret_vals[0:4]

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


def scanning_to_pixel_coords(scanning_coords):
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

    m_x, b_x, m_y, b_y = _scanning_to_pixel_coords()
    pixel_coords = [m_x * scanning_coords[0] + b_x, m_y * scanning_coords[1] + b_y]
    return pixel_coords


def _scanning_to_pixel_coords():
    """Get the linear parameters for the conversion"""

    ret_vals = get_widefield_calibration_params()
    (
        nv1_scanning_coords,
        nv1_pixel_coords,
        nv2_scanning_coords,
        nv2_pixel_coords,
    ) = ret_vals[0:4]

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

    edge_pixels = []
    inner_pixels = []
    if pixel_drift is None:
        pixel_drift = get_pixel_drift()
    pixel_x = pixel_coords[0] + pixel_drift[0]
    pixel_y = pixel_coords[1] + pixel_drift[1]

    # Don't loop through all the pixels, just the ones that might be relevant
    left = int(np.floor(pixel_x - radius))
    right = int(np.ceil(pixel_x + radius))
    bottom = int(np.floor(pixel_y - radius))
    top = int(np.ceil(pixel_y + radius))
    for x in np.linspace(left, right, right - left + 1, dtype=int):
        for y in np.linspace(bottom, top, top - bottom + 1, dtype=int):
            dist = np.sqrt((x - pixel_x) ** 2 + (y - pixel_y) ** 2)
            val = img_array[y, x]
            if abs(dist - radius) < 0.5:
                edge_pixels.append(val)
            elif dist < radius:
                inner_pixels.append(val)

    bg = np.median(edge_pixels)
    total_bg = bg * len(inner_pixels)
    counts = np.sum(inner_pixels) - total_bg

    return counts


def get_pixel_drift():
    pixel_drift = common.get_registry_entry(["State"], "PIXEL_DRIFT")
    return np.array(pixel_drift)


def set_pixel_drift(drift):
    return common.set_registry_entry(["State"], "PIXEL_DRIFT", drift)


def reset_pixel_drift():
    return set_pixel_drift([0.0, 0.0])


def adjust_pixel_coords_for_drift(pixel_coords, drift=None):
    """Current drift will be retrieved from registry if passed drift is None"""
    if drift is None:
        drift = get_pixel_drift()
    adjusted_coords = (np.array(pixel_coords) + np.array(drift)).tolist()
    return adjusted_coords


def set_scanning_drift_from_pixel_drift(pixel_drift=None):
    scanning_drift = pixel_to_scanning_drift(pixel_drift)
    pos.set_drift(scanning_drift)


def pixel_to_scanning_drift(pixel_drift=None):
    scanning_drift = common.get_registry_entry(["State"], "DRIFT")
    if pixel_drift is None:
        pixel_drift = get_pixel_drift()
    m_x, _, m_y, _ = _pixel_to_scanning_coords()
    return [m_x * pixel_drift[0], m_y * pixel_drift[1], scanning_drift[2]]


if __name__ == "__main__":
    pixel_drift = [-5.56, 5.64]
    test = pixel_to_scanning_drift(pixel_drift)
    print(test)
