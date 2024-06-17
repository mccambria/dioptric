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
from utils.widefield import crop_img_array

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
    [150.34, 119.249],  # Too much crosstalk
    [61.277 - 1, 76.387],
    [85.384 - 1, 33.935],
]
base_pixel_drift = [10, 38]


def main(
    file_id,
    diff=True,
    sig_or_ref=None,
    img_array_offset=[0, 0],
    vmin=None,
    vmax=None,
    draw_circles=False,
    draw_circles_inds=None,
    inverted=False,
):
    ### Unpacking

    data = dm.get_raw_data(file_id=file_id, load_npz=True, use_cache=False)
    global pixel_coords_list
    buffer = 20

    if "img_arrays" in data:
        img_arrays = np.array(data["img_arrays"])
        size = img_arrays.shape[-1]
        downsample_factor = round(250 / size)
        buffer = buffer // downsample_factor

        if diff:  # diff
            if not inverted:
                img_arrays = img_arrays[0] - img_arrays[1]
            else:
                img_arrays = img_arrays[1] - img_arrays[0]
        else:
            img_array_ind = 0 if sig_or_ref else 1
            img_arrays = img_arrays[img_array_ind]

        # Crop/center the images
        if "pixel_drifts" in data:
            pixel_drifts = data["pixel_drifts"]
            img_arrays = np.mean(img_arrays, axis=(1, 2))
            cropped_img_arrays = []
            num_runs = img_arrays.shape[0]
            for ind in range(num_runs):
                pixel_drift = pixel_drifts[ind]
                offset = [
                    img_array_offset[0] + (pixel_drift[1] - base_pixel_drift[1]),
                    img_array_offset[1] + (pixel_drift[0] - base_pixel_drift[0]),
                ]
                img_array = img_arrays[ind]
                cropped_img_array = crop_img_array(
                    img_array, offset=offset, buffer=buffer
                )
                cropped_img_arrays.append(cropped_img_array)
            img_array = np.mean(cropped_img_arrays, axis=(0))
        else:
            img_array = np.mean(img_arrays, axis=(0, 1, 2))
            img_array = crop_img_array(
                img_array, offset=img_array_offset, buffer=buffer
            )

        del img_arrays
    else:
        downsample_factor = 1
        img_array = np.array(data["img_array"])
        img_array = widefield.adus_to_photons(img_array)
        img_array = crop_img_array(img_array, offset=img_array_offset, buffer=buffer)
    del data

    ### Imshow

    # downsample_factor = 6
    # img_array = widefield.downsample_img_array(img_array, downsample_factor)

    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array, cbar_label="Photons", vmin=vmin, vmax=vmax)
    ax.axis("off")

    ### Scale bar

    scale = widefield.get_camera_scale(downsample_factor)
    # kpl.scale_bar(ax, scale, "1 µm", kpl.Loc.UPPER_RIGHT)
    kpl.scale_bar(ax, scale, "1 µm", kpl.Loc.LOWER_RIGHT)

    ### Draw circles

    if draw_circles:
        adj_pixel_coords_list = [
            widefield.adjust_pixel_coords_for_drift(
                el, [base_pixel_drift[0] - buffer, base_pixel_drift[1] - buffer]
            )
            for el in pixel_coords_list
        ]

        widefield.draw_circles_on_nvs(
            ax, pixel_coords_list=adj_pixel_coords_list, include_inds=draw_circles_inds
        )

        # Circles to indicate ionization / not ionization
        # smiley_inds = list(range(6))
        # not_ion_inds = [ind for ind in smiley_inds if ind not in ion_inds]
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


# fmt:off
if __name__ == "__main__":
    kpl.init_kplotlib()
    
    # Basic widefield
    # file_id = 1556655608661
    # main(file_id, draw_circles=False)
    # main(file_id, draw_circles=True)
    # main(file_id, draw_circles=True, draw_circles_inds=list(range(10)))
    # main(file_id, draw_circles=True, draw_circles_inds=[0, 1, 5, 6, 10, 11, 12])
    
    ### Missing tooth

    # # Green, same durations
    # main(1557498151070, img_array_offset=[16, 13], vmin=0.05, vmax=8.2)

    # # Green, 3x longer on dim NV
    # main(1557494466189, img_array_offset=[16, 13], vmin=0.05, vmax=8.2)

    # # Histograms: ref, sig, diff
    # file_id = 1556690958663
    # main(file_id, diff=False, sig_or_ref=False, img_array_offset=[3, 5], vmin=-0.29, vmax=0.04)
    # main(file_id, diff=False, sig_or_ref=True, img_array_offset=[3, 5], vmin=-0.29, vmax=0.04)
    # main(file_id, diff=True, img_array_offset=[3, 5])

    # Winking histogram
    # main(1557968425360, diff=True, vmin=-0.29, vmax=0.04)
    # main(1558050169335, diff=True, vmin=-0.29, vmax=0.04)

    # # Spin
    # main(1557059855690, diff=True, img_array_offset=[-2, 0], vmin=0, vmax=1.4)

    ### Complete smiley 
    
    # Green, same durations
    # file_id = 1558527830228
    # main(file_id, img_array_offset=[-3,-2], vmin=0.3, vmax=9.2)
    # main(file_id, img_array_offset=[-2,-1], draw_circles=True)

    # Green, 3x longer on dim NV
    # file_id = 1558551918959
    # main(file_id, img_array_offset=[0,1], vmin=0.3, vmax=9.2)
    # main(file_id, img_array_offset=[0,1], draw_circles=True)

    # Histograms: ref, sig, diff
    # file_id = 1558589699406
    # img_array_offset=[0,0]
    # main(file_id, img_array_offset=img_array_offset, diff=False, sig_or_ref=False, vmin=0.02, vmax=0.42)
    # main(file_id, img_array_offset=img_array_offset, diff=False, sig_or_ref=True, vmin=0.02, vmax=0.42)
    # main(file_id, img_array_offset=img_array_offset, diff=True, vmin=-0.32, vmax=0.02)
    # main(file_id, img_array_offset=img_array_offset, diff=True, vmin=-0.02, vmax=0.32, inverted=True) # Inverted
    # main(file_id, img_array_offset=img_array_offset, diff=True, draw_circles=True)

    # Winking histogram
    # file_id = 1558619706453
    # img_array_offset=[0,0]
    # main(file_id, img_array_offset=img_array_offset, diff=True, vmin=-0.32, vmax=0.02)
    # main(file_id, img_array_offset=img_array_offset, diff=True, vmin=-0.02, vmax=0.32, inverted=True)
    # main(file_id, img_array_offset=img_array_offset, diff=True, draw_circles=True)

    # Spin
    # file_id = 1558797947702
    # img_array_offset=[0,1]
    # file_id = 1558944220372
    # img_array_offset=[0,1]
    # file_id = 1559062712968
    # img_array_offset=[1,0]
    file_id = 1559550352430
    img_array_offset=[0,0]
    # main(file_id, img_array_offset=img_array_offset, diff=True, vmin=-0.005, vmax=0.046)
    main(file_id, img_array_offset=img_array_offset, diff=True)
    # main(file_id, img_array_offset=img_array_offset, diff=True, draw_circles=True)

    plt.show(block=True)
