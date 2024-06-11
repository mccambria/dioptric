# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera

Created on April 9th, 2019

@author: mccambria
"""

import sys
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.optimize as optimize
from majorroutines.widefield.optimize import optimize_pixel
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import LaserKey, NVSig

if __name__ == "__main__":
    kpl.init_kplotlib()

    ######

    ### Basic widefield
    # # data = dm.get_raw_data(file_id=1556420881234, load_npz=True)
    # data = dm.get_raw_data(file_id=1556655608661, load_npz=True)
    # img_array = np.array(data["img_array"])
    # img_array = widefield.adus_to_photons(img_array)
    # img_array_offset = [0, 0]

    ### Just green

    # Same durations
    # data = dm.get_raw_data(file_id=1557498151070, load_npz=True)
    # img_array = np.array(data["img_array"])
    # img_array = widefield.adus_to_photons(img_array)
    # img_array_offset = [0, 0]

    # 3x longer on NV2 (other orientation)
    # data = dm.get_raw_data(file_id=1557494762280, load_npz=True)
    # img_array = np.array(data["img_array"])
    # img_array = widefield.adus_to_photons(img_array)
    # img_array_offset = [0, 0]

    ### Smiley histograms
    smiley_inds = list(range(6))

    # Ref
    # file_id = 1556690958663
    # img_array_offset = [3, 5]
    # ion_inds = []
    # img_array_ind = 1

    # Everything ionized
    # file_id = 1556690958663
    # img_array_offset = [3, 5]
    # ion_inds = [0, 1, 2, 3, 4, 5]
    # img_array_ind = 0

    # Every other
    # file_id = 1556723203013
    # img_array_offset = [0, -1]
    # ion_inds = [0, 2, 5]
    # img_array_ind = 0

    # Inverted every other
    # file_id = 1556745846534
    # img_array_offset = [2, 5]
    # ion_inds = [1, 3, 4]
    # img_array_ind = 0

    # Blinking
    # file_id = 1556830424482
    # img_array_offset = [0, 4]
    # ion_inds = [4, 5]
    # img_array_ind = 0

    # Winking
    # file_id = 1556850284411
    # img_array_offset = [-2, 0]
    # ion_inds = [5]
    # img_array_ind = 0

    ### Smiley SCC

    # downsample_factor = 4

    # Ref
    # file_id = 1557059855690
    # img_array_offset = [0, 0]
    # ion_inds = []
    # img_array_ind = 1

    # Everything spin -1
    # file_id = 1557059855690
    # img_array_offset = [3, 5]
    # ion_inds = [0, 1, 2, 3, 4, 5]
    # img_array_ind = 0

    ###

    data = dm.get_raw_data(file_id=file_id, load_npz=True, use_cache=False)
    img_arrays = np.array(data["img_arrays"])
    del data
    img_arrays = img_arrays[img_array_ind]
    img_array = np.mean(img_arrays, axis=(0, 1, 2))
    del img_arrays
    not_ion_inds = [ind for ind in smiley_inds if ind not in ion_inds]

    ######

    # widefield.replace_dead_pixel(img_array)
    # img_array = img_array[
    #     10 + img_array_offset[0] : 240 + img_array_offset[0],
    #     10 + img_array_offset[1] : 240 + img_array_offset[1],
    # ]

    pixel_coords_list = [
        [131.144, 129.272],  #  Smiley
        [161.477, 105.335],  #  Smiley
        [135.139, 104.013],
        [110.023, 87.942],
        [144.169, 163.787],
        [173.93, 78.505],  #  Smiley
        [171.074, 49.877],  #  Smiley
        [170.501, 132.597],
        [137.025, 74.662],
        [58.628, 139.616],
        # Smiley additions
        # [150.34, 119.249],  # Too much crosstalk
        [61.277, 76.387],
        [85.384, 33.935],
    ]
    pixel_coords_list = [
        widefield.adjust_pixel_coords_for_drift(el, [10 - 10, 38 - 10])
        for el in pixel_coords_list
    ]

    # coords_list = [
    #     widefield.get_nv_pixel_coords(nv, drift_adjust=True, drift=(4, 10))
    #     for nv in nv_list
    # ]
    # widefield.integrate_counts(img_array, coords_list[1])
    # print(coords_list)
    # scale = widefield.get_camera_scale()
    # um_coords_list = [
    #     (round(el[0] / scale, 3), round(el[1] / scale, 3)) for el in coords_list
    # ]
    # print(um_coords_list)

    # img_array = widefield.adus_to_photons(img_array)

    # Clean up dead pixel by taking average of nearby pixels
    # dead_pixel = [142, 109]
    # dead_pixel_x = dead_pixel[1]
    # dead_pixel_y = dead_pixel[0]
    # img_array[dead_pixel_y, dead_pixel_x] = np.mean(
    #     img_array[
    #         dead_pixel_y - 1 : dead_pixel_y + 1 : 2,
    #         dead_pixel_x - 1 : dead_pixel_x + 1 : 2,
    #     ]
    # )

    fig, ax = plt.subplots()
    # img_array = widefield.downsample_img_array(img_array, 4)
    kpl.imshow(ax, img_array, cbar_label="Photons")
    # kpl.imshow(ax, img_array, cbar_label="Photons", vmin=0.05, vmax=0.45)
    ax.axis("off")

    scale = widefield.get_camera_scale(downsample_factor)
    kpl.scale_bar(ax, scale, "1 µm", kpl.Loc.UPPER_RIGHT)

    # pixel_coords_list = [pixel_coords_list[ind] for ind in range(10)]
    # widefield.draw_circles_on_nvs(ax, pixel_coords_list=pixel_coords_list)
    # pixel_coords_list = [pixel_coords_list[ind] for ind in [0, 1, 5, 6, 10, 11]]
    # draw_coords_list = [pixel_coords_list[ind] for ind in ion_inds]
    # widefield.draw_circles_on_nvs(
    #     ax,
    #     pixel_coords_list=draw_coords_list,
    #     color=kpl.KplColors.DARK_GRAY,
    #     no_legend=True,
    #     linestyle="solid",
    # )
    # draw_coords_list = [pixel_coords_list[ind] for ind in not_ion_inds]
    # widefield.draw_circles_on_nvs(
    #     ax,
    #     pixel_coords_list=draw_coords_list,
    #     color=kpl.KplColors.DARK_GRAY,
    #     no_legend=True,
    #     linestyle="dashed",
    # )

    plt.show(block=True)
