# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on Fall 2023

@author: mccambria
"""

import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey
from utils.tool_belt import determine_threshold

# region Process and plotting functions


# def create_histogram(
#     sig_counts_list,
#     ref_counts_list,
#     no_title=True,
#     no_text=None,
#     ax=None,
#     density=False,
#     plot=False,  # Default to False to prevent histogram plotting
# ):
#     if not plot:
#         return None  # Skip plotting if plot is set to False

#     try:
#         laser_dict = tb.get_optics_dict(LaserKey.WIDEFIELD_CHARGE_READOUT)
#         readout = laser_dict["duration"]
#         readout_ms = int(readout / 1e6)
#         readout_s = readout / 1e9
#     except Exception:
#         readout_s = 0.05  # MCC default
#         pass

#     ### Histograms
#     num_reps = len(ref_counts_list)
#     labels = ["With ionization pulse", "Without ionization pulse"]
#     colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
#     counts_lists = [sig_counts_list, ref_counts_list]

#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = None
#     if not no_title:
#         ax.set_title(f"Charge prep hist, {num_reps} reps")
#     ax.set_xlabel("Integrated counts")
#     if density:
#         ax.set_ylabel("Probability")
#     else:
#         ax.set_ylabel("Number of occurrences")

#     for ind in range(2):
#         counts_list = counts_lists[ind]
#         label = labels[ind]
#         color = colors[ind]
#         kpl.histogram(ax, counts_list, label=label, color=color, density=density)

#     ax.legend()

#     # Calculate the normalized separation
#     if not no_text:
#         noise = np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
#         signal = np.mean(ref_counts_list) - np.mean(sig_counts_list)
#         snr = signal / noise
#         snr_time = snr / np.sqrt(readout_s)
#         snr = round(snr, 3)
#         snr_time = round(snr_time, 3)
#         snr_str = f"SNR:\n{snr} / sqrt(shots)\n{snr_time} / sqrt(s)"
#         print(snr_str)
#         snr_str = f"SNR: {snr}"
#         kpl.anchored_text(ax, snr_str, "center right", size=kpl.Size.SMALL)

#     if fig is not None:
#         return fig


# def process_and_plot(raw_data, plot_histograms=False):
#     ### Setup
#     nv_list = raw_data["nv_list"]
#     num_nvs = len(nv_list)
#     counts = np.array(raw_data["counts"])
#     sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
#     ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]
#     num_reps = raw_data["num_reps"]
#     num_runs = raw_data["num_runs"]
#     num_shots = num_reps * num_runs

#     ### Histograms and thresholding
#     threshold_list = []
#     prep_fidelity_list = []
#     hist_figs = []
#     DEFAULT_THRESHOLD = 0.0

#     for ind in range(num_nvs):
#         sig_counts_list = sig_counts_lists[ind]
#         ref_counts_list = ref_counts_lists[ind]

#         # Conditionally plot histograms only if plot_histograms=True
#         fig = create_histogram(
#             sig_counts_list, ref_counts_list, density=True, plot=plot_histograms
#         )
#         if fig:
#             hist_figs.append(fig)
#         all_counts_list = np.append(sig_counts_list, ref_counts_list)

#         try:
#             threshold = determine_threshold(all_counts_list, nvn_ratio=0.5)
#         except:
#             threshold = DEFAULT_THRESHOLD
#         threshold_list.append(threshold)

#         prep_fidelity_list.append(
#             np.sum(np.less(sig_counts_list, threshold)) / num_shots
#         )

#     print(threshold_list)
#     print([round(el, 3) for el in prep_fidelity_list])

#     ### Images
#     if "img_arrays" not in raw_data:
#         return

#     laser_key = LaserKey.WIDEFIELD_CHARGE_READOUT
#     laser_dict = tb.get_optics_dict(laser_key)
#     readout_laser = laser_dict["name"]
#     readout = laser_dict["duration"]
#     readout_ms = readout / 10**6

#     img_arrays = raw_data["img_arrays"]
#     mean_img_arrays = np.mean(img_arrays, axis=(1, 2, 3))
#     sig_img_array = mean_img_arrays[0]
#     ref_img_array = mean_img_arrays[1]
#     diff_img_array = sig_img_array - ref_img_array
#     img_arrays_to_save = [sig_img_array, ref_img_array, diff_img_array]
#     title_suffixes = ["sig", "ref", "diff"]
#     img_figs = []

#     for ind in range(3):
#         img_array = img_arrays_to_save[ind]
#         title_suffix = title_suffixes[ind]
#         fig, ax = plt.subplots()
#         title = f"{readout_laser}, {readout_ms} ms, {title_suffix}"
#         kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
#         img_figs.append(fig)

#     return img_arrays_to_save, img_figs, hist_figs


def create_histogram(
    sig_counts_list,
    ref_counts_list,
    no_title=True,
    no_text=None,
    ax=None,
    density=False,
    plot=False,  # Default to False to prevent histogram plotting
    nv_index=None,  # Add NV index as an optional parameter
):
    if not plot:
        return None  # Skip plotting if plot is set to False

    try:
        laser_dict = tb.get_virtual_laser_dict(VirtualLaserKey.WIDEFIELD_CHARGE_READOUT)
        readout = laser_dict["duration"]
        readout_ms = int(readout / 1e6)
        readout_s = readout / 1e9
    except Exception:
        readout_s = 0.05  # MCC default
        pass

    ### Histograms
    num_reps = len(ref_counts_list)
    labels = ["With ionization pulse", "Without ionization pulse"]
    colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
    counts_lists = [sig_counts_list, ref_counts_list]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    if not no_title:
        ax.set_title(f"Charge prep hist, {num_reps} reps")
    ax.set_xlabel("Integrated counts")
    if density:
        ax.set_ylabel("Probability")
    else:
        ax.set_ylabel("Number of occurrences")

    for ind in range(2):
        counts_list = counts_lists[ind]
        label = labels[ind]
        color = colors[ind]
        kpl.histogram(ax, counts_list, label=label, color=color, density=density)

    ax.legend()

    # Calculate the normalized separation (SNR)
    if not no_text:
        noise = np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
        signal = np.mean(ref_counts_list) - np.mean(sig_counts_list)
        snr = signal / noise
        snr_time = snr / np.sqrt(readout_s)
        snr = round(snr, 3)
        snr_time = round(snr_time, 3)

        # Add NV index in the SNR text
        if nv_index is not None:
            snr_str = f"nv{nv_index}\nSNR: {snr} / sqrt(shots)\n{snr_time} / sqrt(s)"
        else:
            snr_str = f"SNR:\n{snr} / sqrt(shots)\n{snr_time} / sqrt(s)"

        print(snr_str)
        snr_str = f"NV{nv_index} SNR: {snr}"  # Display NV index as well
        kpl.anchored_text(ax, snr_str, "center right", size=kpl.Size.SMALL)

    if fig is not None:
        return fig


def process_and_plot(raw_data, plot_histograms=False):
    ### Setup
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]
    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    ### Histograms and thresholding
    threshold_list = []
    prep_fidelity_list = []
    snr_list = []
    hist_figs = []
    DEFAULT_THRESHOLD = 0.0

    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        # Plot histograms with NV index and SNR included
        fig = create_histogram(
            sig_counts_list,
            ref_counts_list,
            density=True,
            plot=plot_histograms,
            nv_index=ind,
        )
        if fig:
            hist_figs.append(fig)
        all_counts_list = np.append(sig_counts_list, ref_counts_list)

        try:
            threshold = determine_threshold(all_counts_list, nvn_ratio=0.5)
        except:
            threshold = DEFAULT_THRESHOLD
        threshold_list.append(threshold)

        prep_fidelity_list.append(
            np.sum(np.less(sig_counts_list, threshold)) / num_shots
        )
        # Calculate SNR
        noise = np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
        signal = np.mean(ref_counts_list) - np.mean(sig_counts_list)
        snr = signal / noise
        snr_list.append(round(snr, 3))

    print(f"Threshold: {threshold_list}")
    print(f"Fidelity: {[round(el, 3) for el in prep_fidelity_list]}")
    print(f"SNR: {snr_list}")
    ### Images
    if "img_arrays" not in raw_data:
        return

    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout_laser = laser_dict["name"]
    readout = laser_dict["duration"]
    readout_ms = readout / 10**6

    img_arrays = raw_data["img_arrays"]
    mean_img_arrays = np.mean(img_arrays, axis=(1, 2, 3))
    sig_img_array = mean_img_arrays[0]
    ref_img_array = mean_img_arrays[1]
    diff_img_array = sig_img_array - ref_img_array
    img_arrays_to_save = [sig_img_array, ref_img_array, diff_img_array]
    title_suffixes = ["sig", "ref", "diff"]

    img_figs = []

    for ind in range(3):
        img_array = img_arrays_to_save[ind]
        title_suffix = title_suffixes[ind]
        fig, ax = plt.subplots()
        title = f"{readout_laser}, {readout_ms} ms, {title_suffix}"
        kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
        img_figs.append(fig)

    return img_arrays_to_save, img_figs, hist_figs


# endregion


def main(
    nv_list,
    num_reps,
    num_runs,
    verify_charge_states=False,
    diff_polarize=False,
    diff_ionize=True,
    ion_include_inds=None,
    plot_histograms=False,  # Set plot_histograms default to False
):
    ### Initial setup
    seq_file = "charge_state_histograms.py"
    num_steps = 1

    if verify_charge_states:
        charge_prep_fn = base_routine.charge_prep_loop
    else:
        charge_prep_fn = None

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        pol_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.CHARGE_POL)
        ion_coords_list = widefield.get_coords_list(
            nv_list, VirtualLaserKey.ION, include_inds=ion_include_inds
        )
        seq_args = [
            pol_coords_list,
            ion_coords_list,
            diff_polarize,
            diff_ionize,
            verify_charge_states,
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        save_images=True,
        save_images_avg_reps=False,
        charge_prep_fn=charge_prep_fn,
    )

    ### Processing

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    try:
        imgs, img_figs, hist_figs = process_and_plot(
            raw_data, plot_histograms=plot_histograms
        )

        title_suffixes = ["sig", "ref", "diff"]
        num_figs = len(img_figs)
        for ind in range(num_figs):
            fig = img_figs[ind]
            title = title_suffixes[ind]
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{title}")
            dm.save_figure(fig, file_path)

        num_nvs = len(nv_list)
        for nv_ind in range(num_nvs):
            fig = hist_figs[nv_ind]
            nv_sig = nv_list[nv_ind]
            nv_name = nv_sig.name
            file_path = dm.get_file_path(__file__, timestamp, nv_name)
            dm.save_figure(fig, file_path)

        sig_img_array, ref_img_array, diff_img_array = imgs
        keys_to_compress = ["sig_img_array", "ref_img_array", "diff_img_array"]

    except Exception:
        print(traceback.format_exc())
        sig_img_array = None
        ref_img_array = None
        diff_img_array = None
        keys_to_compress = None

    try:
        del raw_data["img_arrays"]
    except Exception:
        pass

    ### Save raw data

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    raw_data |= {
        "timestamp": timestamp,
        "diff_polarize": diff_polarize,
        "diff_ionize": diff_ionize,
        "sig_img_array": sig_img_array,
        "ref_img_array": ref_img_array,
        "diff_img_array": diff_img_array,
        "img_array-units": "photons",
    }
    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    tb.reset_cfm()

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()
    data = dm.get_raw_data(file_id=1642395666145)
    process_and_plot(data, plot_histograms=False)  # Ensure histograms are not plotted


# old version
# import os
# import sys
# import time
# import traceback

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import ndimage
# from scipy.optimize import curve_fit
# from scipy.special import factorial

# from majorroutines.widefield import base_routine, optimize
# from utils import common, widefield
# from utils import data_manager as dm
# from utils import kplotlib as kpl
# from utils import positioning as pos
# from utils import tool_belt as tb
# from utils.constants import LaserKey, NVSig
# from utils.tool_belt import determine_threshold

# # region Process and plotting functions


# def create_histogram(
#     sig_counts_list,
#     ref_counts_list,
#     no_title=True,
#     no_text=None,
#     ax=None,
#     density=False,
# ):
#     try:
#         laser_dict = tb.get_optics_dict(LaserKey.WIDEFIELD_CHARGE_READOUT)
#         readout = laser_dict["duration"]
#         readout_ms = int(readout / 1e6)
#         readout_s = readout / 1e9
#     except Exception:
#         readout_s = 0.05  # MCC
#         pass

#     ### Histograms

#     num_reps = len(ref_counts_list)

#     labels = ["With ionization pulse", "Without ionization pulse"]
#     colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
#     counts_lists = [sig_counts_list, ref_counts_list]
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = None
#     if not no_title:
#         ax.set_title(f"Charge prep hist, {num_reps} reps")
#     ax.set_xlabel("Integrated counts")
#     if density:
#         ax.set_ylabel("Probability")
#     else:
#         ax.set_ylabel("Number of occurrences")
#     for ind in range(2):
#         counts_list = counts_lists[ind]
#         label = labels[ind]
#         color = colors[ind]
#         # kpl.histogram(ax, counts_list, num_bins, label=labels[ind])
#         kpl.histogram(ax, counts_list, label=label, color=color, density=density)
#     ax.legend()

#     # Calculate the normalized separation
#     if not no_text:
#         noise = np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
#         signal = np.mean(ref_counts_list) - np.mean(sig_counts_list)
#         snr = signal / noise
#         snr_time = snr / np.sqrt(readout_s)
#         snr = round(snr, 3)
#         snr_time = round(snr_time, 3)
#         snr_str = f"SNR:\n{snr} / sqrt(shots)\n{snr_time} / sqrt(s)"
#         print(snr_str)
#         # kpl.anchored_text(ax, snr_str, "center right", size=kpl.Size.SMALL)
#         snr_str = f"SNR: {snr}"
#         kpl.anchored_text(ax, snr_str, "center right", size=kpl.Size.SMALL)

#     if fig is not None:
#         return fig


# def process_and_plot(raw_data):
#     ### Setup

#     nv_list = raw_data["nv_list"]

#     num_nvs = len(nv_list)
#     counts = np.array(raw_data["counts"])
#     sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
#     ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]
#     num_reps = raw_data["num_reps"]
#     num_runs = raw_data["num_runs"]
#     num_shots = num_reps * num_runs

#     ### Histograms and thresholding

#     threshold_list = []
#     prep_fidelity_list = []
#     hist_figs = []
#     # prior_thresholds = [27.5, 27.5, 24.5, 22.5, 25.5, 25.5, 20.5, 19.5, 16.5, 18.5]
#     DEFAULT_THRESHOLD = 0.0
#     for ind in range(num_nvs):
#         sig_counts_list = sig_counts_lists[ind]
#         ref_counts_list = ref_counts_lists[ind]
#         fig = create_histogram(sig_counts_list, ref_counts_list, density=True)
#         hist_figs.append(fig)
#         all_counts_list = np.append(sig_counts_list, ref_counts_list)
#         try:
#             threshold = determine_threshold(all_counts_list, nvn_ratio=0.5)
#         except:
#             threshold = DEFAULT_THRESHOLD
#         threshold_list.append(threshold)
#         # threshold = prior_thresholds[ind]
#         prep_fidelity_list.append(
#             # np.sum(np.greater(ref_counts_list, threshold)) / num_shots
#             np.sum(np.less(sig_counts_list, threshold)) / num_shots
#         )
#     print(threshold_list)
#     print([round(el, 3) for el in prep_fidelity_list])

#     ### Images

#     if "img_arrays" not in raw_data:
#         return

#     laser_key = LaserKey.WIDEFIELD_CHARGE_READOUT
#     laser_dict = tb.get_optics_dict(laser_key)
#     readout_laser = laser_dict["name"]
#     readout = laser_dict["duration"]
#     readout_ms = readout / 10**6

#     img_arrays = raw_data["img_arrays"]
#     mean_img_arrays = np.mean(img_arrays, axis=(1, 2, 3))
#     sig_img_array = mean_img_arrays[0]
#     ref_img_array = mean_img_arrays[1]
#     diff_img_array = sig_img_array - ref_img_array
#     img_arrays_to_save = [sig_img_array, ref_img_array, diff_img_array]
#     title_suffixes = ["sig", "ref", "diff"]
#     img_figs = []
#     for ind in range(3):
#         img_array = img_arrays_to_save[ind]
#         title_suffix = title_suffixes[ind]
#         fig, ax = plt.subplots()
#         title = f"{readout_laser}, {readout_ms} ms, {title_suffix}"
#         kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
#         img_figs.append(fig)

#     ### MLE state estimation

#     # shape = widefield.get_img_array_shape()
#     # ref_img_arrays = img_arrays[1]
#     # ref_img_arrays = ref_img_arrays.reshape((num_shots, *shape))
#     # nvn_dist_params_list = []
#     # for nv_ind in range(num_nvs):
#     #     nvn_img_arrays = []
#     #     nv = nv_list[nv_ind]
#     #     threshold = threshold_list[nv_ind] + 5  # + 5 to decrease nv0 prob
#     #     ref_counts_list = ref_counts_lists[nv_ind]
#     #     for shot_ind in range(num_shots):
#     #         if ref_counts_list[shot_ind] > threshold:
#     #             nvn_img_arrays.append(ref_img_arrays[shot_ind])
#     #     mean_img_array = np.mean(nvn_img_arrays, axis=0)
#     #     popt = optimize.optimize_pixel_with_img_array(
#     #         mean_img_array, nv, return_popt=True
#     #     )
#     #     # bg, amp, sigma
#     #     nvn_dist_params_list.append((popt[-1], popt[0], popt[-2]))
#     # print(nvn_dist_params_list)

#     return img_arrays_to_save, img_figs, hist_figs


# # endregion


# def main(
#     nv_list,
#     num_reps,
#     num_runs,
#     verify_charge_states=False,
#     diff_polarize=False,
#     diff_ionize=True,
#     ion_include_inds=None,
# ):
#     ### Some initial setup
#     seq_file = "charge_state_histograms.py"
#     num_steps = 1

#     if verify_charge_states:
#         charge_prep_fn = base_routine.charge_prep_loop
#     else:
#         # charge_prep_fn = base_routine.charge_prep_no_verification
#         charge_prep_fn = None

#     pulse_gen = tb.get_server_pulse_gen()

#     ### Collect the data

#     def run_fn(shuffled_step_inds):
#         pol_coords_list = widefield.get_coords_list(nv_list, LaserKey.CHARGE_POL)
#         ion_coords_list = widefield.get_coords_list(
#             nv_list, LaserKey.ION, include_inds=ion_include_inds
#         )
#         seq_args = [
#             pol_coords_list,
#             ion_coords_list,
#             diff_polarize,
#             diff_ionize,
#             verify_charge_states,
#         ]
#         seq_args_string = tb.encode_seq_args(seq_args)
#         pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

#     raw_data = base_routine.main(
#         nv_list,
#         num_steps,
#         num_reps,
#         num_runs,
#         run_fn=run_fn,
#         save_images=True,
#         save_images_avg_reps=False,
#         charge_prep_fn=charge_prep_fn,
#         # uwave_ind_list=[0, 1],  # MCC
#     )

#     ### Processing

#     timestamp = dm.get_time_stamp()
#     repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
#     repr_nv_name = repr_nv_sig.name

#     try:
#         imgs, img_figs, hist_figs = process_and_plot(raw_data)

#         title_suffixes = ["sig", "ref", "diff"]
#         for ind in range(len(img_figs)):
#             fig = img_figs[ind]
#             title = title_suffixes[ind]
#             file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{title}")
#             dm.save_figure(fig, file_path)

#         num_nvs = len(nv_list)
#         for nv_ind in range(num_nvs):
#             fig = hist_figs[nv_ind]
#             nv_sig = nv_list[nv_ind]
#             nv_name = nv_sig.name
#             file_path = dm.get_file_path(__file__, timestamp, nv_name)
#             dm.save_figure(fig, file_path)

#         sig_img_array, ref_img_array, diff_img_array = imgs
#         keys_to_compress = [
#             "sig_img_array",
#             "ref_img_array",
#             "diff_img_array",
#         ]

#     except Exception:
#         print(traceback.format_exc())
#         sig_img_array = None
#         ref_img_array = None
#         diff_img_array = None
#         keys_to_compress = None

#     try:
#         del raw_data["img_arrays"]
#     except Exception:
#         pass
#     # keys_to_compress = ["img_arrays"]

#     ### Save raw data

#     file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
#     raw_data |= {
#         "timestamp": timestamp,
#         "diff_polarize": diff_polarize,
#         "diff_ionize": diff_ionize,
#         "sig_img_array": sig_img_array,
#         "ref_img_array": ref_img_array,
#         "diff_img_array": diff_img_array,
#         "img_array-units": "photons",
#     }
#     dm.save_raw_data(raw_data, file_path, keys_to_compress)

#     tb.reset_cfm()

#     return raw_data


# if __name__ == "__main__":
#     kpl.init_kplotlib()
#     data = dm.get_raw_data(file_id=1642395666145)
#     process_and_plot(data)

#     ### Images

#     # nv_list = data["nv_list"]

#     # sig_img_array = np.array(data["sig_img_array"])
#     # ref_img_array = np.array(data["ref_img_array"])
#     # diff_img_array = np.array(data["diff_img_array"])
#     # img_arrays = [sig_img_array, ref_img_array, diff_img_array]
#     # titles = ["With ionization pulse", "Without ionization pulse", "difference"]

#     # for ind in range(3):
#     #     img_array = img_arrays[ind]
#     #     fig, ax = plt.subplots()
#     #     title = titles[ind]
#     #     img_array[142, 109] = np.mean(img_array[141:143:2, 108:110:2])
#     #     kpl.imshow(
#     #         ax,
#     #         img_array,
#     #         title=title,
#     #         cbar_label="Photons",
#     #         vmin=0 if ind < 1 else None,
#     #         vmax=np.max(ref_img_array) if ind < 1 else None,
#     #     )
#     #     scale = widefield.get_camera_scale()
#     #     kpl.scale_bar(ax, scale, "1 µm", kpl.Loc.UPPER_RIGHT)

#     #     widefield.draw_circles_on_nvs(ax, nv_list, drift=(-1, -11))

#     kpl.show(block=True)
