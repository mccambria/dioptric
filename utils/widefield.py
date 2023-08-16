# -*- coding: utf-8 -*-
"""Various utility functions for widefield imaging and camera data processing

Created on August 15th, 2023

@author: mccambria
"""

# region Imports and constants

import matplotlib.pyplot as plt
import numpy as np
from utils import common
from scipy.optimize import curve_fit
from utils import tool_belt as tb
from utils import kplotlib as kpl

# endregion


def pixel_to_scanning(pixel_coords):
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

    scanning_coords = [m_x * pixel_coords[0] + b_x, m_y * pixel_coords[0] + b_y]
    return scanning_coords


def scanning_to_pixel(scanning_coords):
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

    pixel_coords = [m_x * scanning_coords[0] + b_x, m_y * scanning_coords[0] + b_y]
    return pixel_coords


def counts_from_img_array(img_array, pixel_coords, radius):
    edge_pixels = []
    inner_pixels = []
    pixel_x = pixel_coords[0]
    pixel_y = pixel_coords[1]

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


def _circle_gaussian(xy, amplitude, xo, yo, sigma, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(
        -(1 / (2 * sigma**2)) * (((x - xo) ** 2) + ((y - yo) ** 2))
    )
    return g.ravel()


def optimize_pixel(img_array, pixel_coords, radius, set_drift=True):
    round_x = round(pixel_coords[0])
    round_y = round(pixel_coords[1])
    round_r = round(radius)
    bg_guess = img_array[round_y, round_x + round_r]
    amp_guess = img_array[round_y, round_x] - bg_guess
    guess = (amp_guess, *pixel_coords, radius / 2, bg_guess)
    shape = img_array.shape
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    x, y = np.meshgrid(x, y)
    popt, pcov = curve_fit(_circle_gaussian, (x, y), img_array.ravel(), p0=guess)
    opti_pixel_coords = popt[1:3]
    return opti_pixel_coords


if __name__ == "__main__":
    file_name = "2023_08_15-14_34_47-johnson-nvref"
    data = tb.get_raw_data(file_name)
    img_array = np.array(data["img_array"])

    pixel_coords = [124.5, 196.5]
    radius = 8

    counts = counts_from_img_array(img_array, pixel_coords, radius)
    print(counts)

    opt = optimize_pixel(img_array, pixel_coords, radius)
    print(opt)

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    im = kpl.imshow(ax, img_array, x_label="X", y_label="Y", cbar_label="Pixel values")
    ax.set_xlim([pixel_coords[0] - 15, pixel_coords[0] + 15])
    ax.set_ylim([pixel_coords[1] + 15, pixel_coords[1] - 15])

    plt.show(block=True)
