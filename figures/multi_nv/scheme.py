# -*- coding: utf-8 -*-
"""
Main text fig 1

Created on June 5th, 2024

@author: mccambria
"""

import io
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield.charge_monitor import process_check_readout_fidelity
from majorroutines.widefield.charge_state_histograms import plot_histograms
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def main(nv_list, apparatus_img, image_data, histogram_data):
    ### Setup

    # fig = plt.figure(figsize=kpl.double_figsize)
    # fig, ax = plt.subplots(2, 1, figsize=kpl.double_figsize)
    layout = [
        ["apparatus", "smiley"],
        ["apparatus", "histogram"],
    ]
    w_factor = 0.65
    h_factor = 0.5 * 210 / 200
    figsize = kpl.double_figsize
    figsize[1] *= 1.1
    main_fig, axes_pack = plt.subplot_mosaic(
        layout,
        figsize=figsize,
        width_ratios=(w_factor, 1 - w_factor),
        height_ratios=(h_factor, 1 - h_factor),
    )

    ### Apparatus

    ax = axes_pack["apparatus"]
    ax.axis("off")
    ax.imshow(apparatus_img)
    # Get rid of white space for other axis decorations
    ax.set_position([0, 0, 0.94 * w_factor, 1])

    ### Image

    img_array = np.array(image_data["img_array"])
    img_array = widefield.adus_to_photons(img_array, k_gain=5000)

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

    # left, bottom, width, height
    size = 0.53
    inset_ax = main_fig.add_axes([0, 1 - size, size * figsize[1] / figsize[0], size])
    inset_ax.axis("off")
    kpl.imshow(inset_ax, img_array, no_cbar=True)

    scale = 2 * widefield.get_camera_scale()
    kpl.scale_bar(inset_ax, scale, "2 µm", kpl.Loc.UPPER_RIGHT)
    widefield.draw_circles_on_nvs(inset_ax, nv_list, drift=(13, 3))

    ### Smiley

    ax = axes_pack["smiley"]

    ### Charge states histogram

    nv_list = histogram_data["nv_list"]

    num_nvs = len(nv_list)
    counts = np.array(histogram_data["counts"])
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]

    nv_ind = 0
    sig_counts_list = sig_counts_lists[nv_ind]
    ref_counts_list = ref_counts_lists[nv_ind]
    ax = axes_pack["histogram"]
    fig = plot_histograms(
        sig_counts_list,
        ref_counts_list,
        ax=ax,
        no_text=True,
        no_title=True,
        density=True,
    )
    ax.set_xlim(-0.5, 90.5)
    combined_counts = np.append(sig_counts_list, ref_counts_list)
    threshold = widefield.determine_threshold(
        combined_counts,
        single_or_dual=True,
        nvn_ratio=None,
        no_print=True,
    )
    ax.axvline(threshold, color=kpl.KplColors.GRAY, ls="dashed")

    ### Adjustments

    main_fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)

    main_fig.text(0, 0.96, "(a)", color=kpl.KplColors.WHITE)
    main_fig.text(w_factor - 0.04, 0.96, "(b)")
    main_fig.text(w_factor - 0.04, 1 - h_factor + 0.05, "(c)")


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1537195144201)
    nv_list = data["nv_list"]
    apparatus_img = dm.get_img("multiplexed_apparatus", "png")
    # image_data = dm.get_raw_data(file_id=1537097641802, load_npz=True)
    image_data = dm.get_raw_data(file_id=1556655608661, load_npz=True)
    # histogram_data = dm.get_raw_data(file_id=1537195144201)
    histogram_data = dm.get_raw_data(file_id=1556934779836)

    main(nv_list, apparatus_img, image_data, histogram_data)

    plt.show(block=True)
