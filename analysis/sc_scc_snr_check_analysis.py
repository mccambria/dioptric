# -*- coding: utf-8 -*-
"""
Lighweight check of the SCC SNR

Created on Fall, 2024

@author: Saroj Chand
"""
import time
import traceback

import numpy as np
from matplotlib import pyplot as plt

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield


def process_and_plot(data):
    threshold = True
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    if threshold:
        sig_counts, ref_counts = widefield.threshold_counts(
            nv_list, sig_counts, ref_counts, dynamic_thresh=True
        )
        # thresh_method= "otsu"
        # sig_counts, ref_counts = widefield.threshold_counts(nv_list, sig_counts, ref_counts, method=thresh_method)

    ### Report the results

    # Include this block if the ref shots measure both ms=0 and ms=+/-1
    # avg_sig_counts, avg_sig_counts_ste, norms = widefield.average_counts(
    #     sig_counts, ref_counts
    # )
    # norms_ms0_newaxis = norms[0][:, np.newaxis]
    # norms_ms1_newaxis = norms[1][:, np.newaxis]
    # contrast = norms_ms1_newaxis - norms_ms0_newaxis
    # norm_counts = (avg_sig_counts - norms_ms0_newaxis) / contrast
    # norm_counts_ste = avg_sig_counts_ste / contrast

    avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)

    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_contrast, avg_contrast_ste = widefield.calc_contrast(sig_counts, ref_counts)

    # There's only one point, so only consider that
    step_ind = 0
    avg_sig_counts = avg_sig_counts[:, step_ind]
    avg_sig_counts_ste = avg_sig_counts_ste[:, step_ind]
    avg_ref_counts = avg_ref_counts[:, step_ind]
    avg_ref_counts_ste = avg_ref_counts_ste[:, step_ind]
    avg_snr = avg_snr[:, step_ind]
    avg_snr_ste = avg_snr_ste[:, step_ind]
    avg_contrast = avg_contrast[:, step_ind]
    avg_contrast_ste = avg_contrast_ste[:, step_ind]

    # Print
    # for ind in range(len(nv_list)):
    #     nv_sig = nv_list[ind]
    #     nv_num = widefield.get_nv_num(nv_sig)
    #     nv_ref_counts = tb.round_for_print(avg_ref_counts[ind], avg_ref_counts_ste[ind])
    #     nv_sig_counts = tb.round_for_print(avg_sig_counts[ind], avg_sig_counts_ste[ind])
    #     nv_snr = tb.round_for_print(avg_snr[ind], avg_snr_ste[ind])
    #     print(f"NV {nv_num}: a0={nv_ref_counts}, a1={nv_sig_counts}, SNR={nv_snr}")
    # print(f"Mean SNR: {np.mean(avg_snr)}")

    ### Plot

    # Normalized counts bar plots
    # fig, ax = plt.subplots()
    # for ind in range(num_nvs):
    #     nv_sig = nv_list[ind]
    #     nv_num = widefield.get_nv_num(nv_sig)
    #     kpl.plot_bars(ax, nv_num, norm_counts[ind], yerr=norm_counts_ste[ind])
    # ax.set_xlabel("NV index")
    # ax.set_ylabel("Contrast")

    # SNR bar plots
    # figsize = kpl.figsize
    # figsize[1] *= 1.5
    # counts_fig, axes_pack = plt.subplots(2, 1, sharex=True, figsize=figsize)
    # snr_fig, ax = plt.subplots()
    # for ind in range(len(nv_list)):
    #     nv_sig = nv_list[ind]
    #     nv_num = widefield.get_nv_num(nv_sig)
    #     kpl.plot_bars(
    #         axes_pack[0], nv_num, avg_ref_counts[ind], yerr=avg_ref_counts_ste[ind]
    #     )
    #     kpl.plot_bars(
    #         axes_pack[1], nv_num, avg_sig_counts[ind], yerr=avg_sig_counts_ste[ind]
    #     )
    #     kpl.plot_bars(ax, nv_num, avg_snr[ind], yerr=avg_snr_ste[ind])
    # axes_pack[0].set_xlabel("NV index")
    # ax.set_xlabel("NV index")
    # axes_pack[0].set_ylabel("NV- | prep in ms=0")
    # axes_pack[1].set_ylabel("NV- | prep in ms=1")
    # ax.set_ylabel("SNR")
    # return counts_fig, snr_fig

    # SNR histogram
    fig, ax = plt.subplots()
    kpl.histogram(ax, avg_snr, kpl.HistType.STEP, nbins=10)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Number of occurrences")

    # SNR vs red frequency
    coords_key = "laser_COBO_638_aod"
    distances = []
    for nv in nv_list:
        coords = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
        dist = np.sqrt((90 - coords[0]) ** 2 + (90 - coords[1]) ** 2)
        distances.append(dist)
    fig, ax = plt.subplots()
    kpl.plot_points(ax, distances, avg_snr)
    ax.set_xlabel("Distance from center frequencies (MHz)")
    ax.set_ylabel("SNR")


# import seaborn as sns
# import pandas as pd


# def process_and_plot(data, error_threshold=0.2):
#     threshold = True
#     nv_list = data["nv_list"]
#     counts = np.array(data["counts"])
#     sig_counts = counts[0]
#     ref_counts = counts[1]

#     # Apply threshold if needed
#     if threshold:
#         thresh_method = "otsu"
#         sig_counts, ref_counts = widefield.threshold_counts(
#             nv_list, sig_counts, ref_counts, method=thresh_method
#         )

#     # Report the results and return
#     avg_sig_counts, avg_sig_counts_ste, norms = widefield.average_counts(
#         sig_counts, ref_counts
#     )
#     norms_ms0_newaxis = norms[0][:, np.newaxis]
#     norms_ms1_newaxis = norms[1][:, np.newaxis]
#     contrast = norms_ms1_newaxis - norms_ms0_newaxis
#     norm_counts = (avg_sig_counts - norms_ms0_newaxis) / contrast
#     norm_counts_ste = avg_sig_counts_ste / contrast

#     # Ensure no negative yerr values
#     norm_counts_ste = np.abs(norm_counts_ste)

#     # Constrain norm_counts to be within [0, 1]
#     norm_counts_clipped = np.clip(norm_counts, 0, 1)

#     ### Plot 1: All Data
#     # Prepare data for seaborn plotting (with all data points)
#     all_nv_nums = [widefield.get_nv_num(nv) for nv in nv_list]
#     all_plot_data = pd.DataFrame(
#         {
#             "NV": all_nv_nums,
#             "Contrast": norm_counts_clipped.flatten(),
#             "Error": norm_counts_ste.flatten(),
#         }
#     )

#     # Set up the first plot with all data points
#     plt.figure(figsize=(15, 8))  # Adjust size for large numbers of NVs
#     sns.set(style="whitegrid")
#     ax_all = sns.barplot(x="NV", y="Contrast", data=all_plot_data, ci=None)

#     # Add error bars manually
#     for i, row in all_plot_data.iterrows():
#         ax_all.errorbar(
#             row["NV"], row["Contrast"], yerr=row["Error"], fmt="none", c="black"
#         )

#     # Customize plot
#     ax_all.set_xlabel("NV Index")
#     ax_all.set_ylabel("Normalized Contrast (0 to 1)")
#     ax_all.set_title("All NV Data")
#     plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
#     plt.tight_layout()

#     ### Plot 2: Filtered Good Data (Contrast between 0-1 and small error bars)
#     filtered_nv_list = []
#     filtered_norm_counts = []
#     filtered_norm_counts_ste = []

#     for i in range(len(norm_counts)):
#         if 0 <= norm_counts[i] <= 1 and norm_counts_ste[i] < error_threshold:
#             filtered_nv_list.append(widefield.get_nv_num(nv_list[i]))
#             filtered_norm_counts.append(norm_counts[i])
#             filtered_norm_counts_ste.append(norm_counts_ste[i])

#     # Prepare data for seaborn plotting (good data points only)
#     good_plot_data = pd.DataFrame(
#         {
#             "NV": filtered_nv_list,
#             "Contrast": np.array(filtered_norm_counts).flatten(),
#             "Error": np.array(filtered_norm_counts_ste).flatten(),
#         }
#     )

#     # Set up the second plot with good data points
#     plt.figure(figsize=(15, 8))  # Adjust size for large numbers of NVs
#     sns.set(style="whitegrid")
#     ax_good = sns.barplot(x="NV", y="Contrast", data=good_plot_data, ci=None)

#     # Add error bars manually for good data points
#     for i, row in good_plot_data.iterrows():
#         ax_good.errorbar(
#             row["NV"], row["Contrast"], yerr=row["Error"], fmt="none", c="black"
#         )

#     # Customize plot
#     ax_good.set_xlabel("NV Index")
#     ax_good.set_ylabel("Normalized Contrast (0 to 1)")
#     ax_good.set_title(f"Good NV Data (Error < {error_threshold})")
#     plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
#     plt.tight_layout()

#     # Show both plots
#     plt.show()

#     print(f"Mean normalized contrast (all data): {np.mean(norm_counts)}")
#     print(f"Mean normalized contrast (good data): {np.mean(filtered_norm_counts)}")

#     return


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1664917535036)
    # data = dm.get_raw_data(file_id=1575309155682)
    # data = dm.get_raw_data(file_id=1575323838562)
    figs = process_and_plot(data)
    kpl.show(block=True)
