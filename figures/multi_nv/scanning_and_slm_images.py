# -*- coding: utf-8 -*-
"""
Main text fig 2

Created on June 5th, 2024

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import resonance, spin_echo
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def main(scanning_datas, slm_data):
    scanning_images = np.array([el["img_array"] for el in scanning_datas])
    scanning_image = np.max(scanning_images, axis=0)
    slm_image = np.array(slm_data["ref_img_array"])
    for img in [scanning_image, slm_image]:
        widefield.replace_dead_pixel(img)

    buffer = 1
    shift = [1, 1]
    scanning_image = scanning_image[buffer:-buffer, buffer:-buffer]
    size = slm_image.shape[0]
    slm_image = slm_image[
        buffer + shift[1] : size - buffer + shift[1],
        buffer + shift[0] : size - buffer + shift[0],
    ]

    figsize = kpl.figsize
    figsize[1] = figsize[0] / 2
    fig, axes_pack = plt.subplots(1, 2, figsize=figsize)
    scanning_ax, slm_ax = axes_pack

    for ax, img_array in zip([scanning_ax, slm_ax], [scanning_image, slm_image]):
        kpl.imshow(ax, img_array, no_cbar=True)
        ax.axis("off")

    num_microns = 5
    scale = num_microns * widefield.get_camera_scale()
    kpl.scale_bar(scanning_ax, scale, f"{num_microns} µm", kpl.Loc.LOWER_RIGHT)

    for img_array in [scanning_image, slm_image]:
        fig, ax = plt.subplots()
        kpl.imshow(ax, img_array, no_cbar=True)
        ax.axis("off")


if __name__ == "__main__":
    kpl.init_kplotlib()

    scanning_file_ids = [
        1693359304252,
        1693375930943,
        1693374155876,
        1693372069315,
        1693363033340,
        1693376854962,
        1693361128042,
        1693376833728,
        1693379996949,
    ]
    slm_file_id = 1733583334808

    scanning_datas = [
        dm.get_raw_data(file_id=el, load_npz=True) for el in scanning_file_ids
    ]
    slm_data = dm.get_raw_data(file_id=slm_file_id, load_npz=True)

    main(scanning_datas, slm_data)

    plt.show(block=True)
