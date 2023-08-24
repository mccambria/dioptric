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
from scipy.optimize import minimize
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
    config = common.get_config_dict()
    config_pos = config["Positioning"]

    NV1_pixel_coords = config_pos["NV1_pixel_coords"]
    NV1_scanning_coords = config_pos["NV1_scanning_coords"]
    NV2_pixel_coords = config_pos["NV2_pixel_coords"]
    NV2_scanning_coords = config_pos["NV2_scanning_coords"]

    # Assume (independent) linear relations for both x and y

    scanning_diff = NV2_scanning_coords[0] - NV1_scanning_coords[0]
    pixel_diff = NV2_pixel_coords[0] - NV1_pixel_coords[0]
    m_x = scanning_diff / pixel_diff
    b_x = NV1_scanning_coords[0] - m_x * NV1_pixel_coords[0]

    scanning_diff = NV2_scanning_coords[1] - NV1_scanning_coords[1]
    pixel_diff = NV2_pixel_coords[1] - NV1_pixel_coords[1]
    m_y = scanning_diff / pixel_diff
    b_y = NV1_scanning_coords[1] - m_y * NV1_pixel_coords[1]

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
    config = common.get_config_dict()
    config_pos = config["Positioning"]

    NV1_pixel_coords = config_pos["NV1_pixel_coords"]
    NV1_scanning_coords = config_pos["NV1_scanning_coords"]
    NV2_pixel_coords = config_pos["NV2_pixel_coords"]
    NV2_scanning_coords = config_pos["NV2_scanning_coords"]

    # Assume (independent) linear relations for both x and y

    pixel_diff = NV2_pixel_coords[0] - NV1_pixel_coords[0]
    scanning_diff = NV2_scanning_coords[0] - NV1_scanning_coords[0]
    m_x = pixel_diff / scanning_diff
    b_x = NV1_pixel_coords[0] - m_x * NV1_scanning_coords[0]

    pixel_diff = NV2_pixel_coords[1] - NV1_pixel_coords[1]
    scanning_diff = NV2_scanning_coords[1] - NV1_scanning_coords[1]
    m_y = pixel_diff / scanning_diff
    b_y = NV1_pixel_coords[1] - m_y * NV1_scanning_coords[1]

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


def _circle_gaussian(x, y, amp, x0, y0, sigma, offset):
    ret_array = offset + amp * np.exp(
        -(1 / (2 * sigma**2)) * (((x - x0) ** 2) + ((y - y0) ** 2))
    )
    return ret_array


def optimize_pixel(
    img_array,
    pixel_coords,
    radius=None,
    set_drift=True,
    drift_adjust=True,
    pixel_drift=None,
):
    # Make copies so we don't mutate the originals
    original_pixel_coords = pixel_coords.copy()
    pixel_coords = pixel_coords.copy()
    if drift_adjust:
        pixel_coords = adjust_pixel_coords_for_drift(pixel_coords, pixel_drift)

    # Bounds and guesses
    if radius is None:
        config = common.get_config_dict()
        radius = config["camera_spot_radius"]
    initial_x = pixel_coords[0]
    initial_y = pixel_coords[1]
    bg_guess = int(img_array[round(initial_y), round(initial_x + radius)])
    amp_guess = int(img_array[round(initial_y), round(initial_x)]) - bg_guess
    guess = (amp_guess, *pixel_coords, radius / 2, bg_guess)
    diam = radius * 2
    half_range = radius
    # lower_bounds = (0, pixel_coords[0] - diam, pixel_coords[1] - diam, 0, 0)
    # upper_bounds = (inf, pixel_coords[0] + diam, pixel_coords[1] + diam, diam, inf)
    left = round(initial_x - half_range)
    right = round(initial_x + half_range)
    top = round(initial_y - half_range)
    bottom = round(initial_y + half_range)
    bounds = ((0, inf), (left, right), (top, bottom), (1, diam), (0, inf))
    shape = img_array.shape
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    x, y = np.meshgrid(x, y)

    def cost(fit_params):
        amp, x0, y0, sigma, offset = fit_params
        gaussian_array = _circle_gaussian(x, y, amp, x0, y0, sigma, offset)
        # Limit the range to the NV we're looking at
        diff_array = (
            gaussian_array[top:bottom, left:right] - img_array[top:bottom, left:right]
        )
        return np.sum(diff_array**2)

    res = minimize(cost, guess, bounds=bounds)
    popt = res.x

    # Testing
    # print(cost(guess))
    # print(cost(popt))
    # fig, ax = plt.subplots()
    # gaussian_array = _circle_gaussian(x, y, *popt)
    # kpl.imshow(ax, gaussian_array)
    # ax.set_xlim([pixel_coords[0] - 15, pixel_coords[0] + 15])
    # ax.set_ylim([pixel_coords[1] + 15, pixel_coords[1] - 15])

    opti_pixel_coords = popt[1:3]
    if set_drift:
        drift = (np.array(opti_pixel_coords) - np.array(original_pixel_coords)).tolist()
        set_pixel_drift(drift)
    return opti_pixel_coords


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
    if pixel_drift is None:
        pixel_drift = get_pixel_drift()
    m_x, _, m_y, _ = _pixel_to_scanning_coords()
    return [m_x * pixel_drift[0], m_y * pixel_drift[1]]


if __name__ == "__main__":
    pixel_drift = [-5.56, 5.64]
    test = pixel_to_scanning_drift(pixel_drift)
    print(test)
