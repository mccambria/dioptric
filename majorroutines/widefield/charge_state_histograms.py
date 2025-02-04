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

from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey


# region Process and plotting functions
def plot_histograms(
    sig_counts_list, ref_counts_list, no_title=True, ax=None, density=False
):
    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    # laser_dict = tb.get_virtual_laser_dict(laser_key)
    # readout = laser_dict["duration"]
    # readout_ms = int(readout / 1e6)
    # readout_s = readout / 1e9

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

    if fig is not None:
        return fig


def process_and_plot(
    raw_data, do_plot_histograms=False, prob_dist: ProbDist = ProbDist.COMPOUND_POISSON
):
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
    readout_fidelity_list = []
    prep_fidelity_list = []
    red_chi_sq_list = []
    hist_figs = []

    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        # Only use ref counts for threshold determination
        popt, _, red_chi_sq = fit_bimodal_histogram(
            ref_counts_list, prob_dist, no_print=True
        )
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, do_print=True, ret_fidelity=True
        )
        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        if popt is not None:
            prep_fidelity = 1 - popt[0]
        else:
            prep_fidelity = np.nan
        # prep_fidelity = (
        #     np.count_nonzero(np.array(ref_counts_list) > threshold) / num_shots
        # )  # MCC
        prep_fidelity_list.append(prep_fidelity)
        red_chi_sq_list.append(red_chi_sq)

        # Plot histograms with NV index and SNR included
        nv_num = widefield.get_nv_num(nv_list[ind])
        # if do_plot_histograms and nv_num == 154:
        if do_plot_histograms:
            fig = plot_histograms(sig_counts_list, ref_counts_list, density=True)
            ax = fig.gca()

            # Ref counts fit line
            if popt is not None:
                single_mode_num_params = bimodal_histogram.get_single_mode_num_params(
                    prob_dist
                )
                single_mode_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)
                bimodal_pdf = bimodal_histogram.get_bimodal_pdf(prob_dist)

                x_vals = np.linspace(0, np.max(ref_counts_list), 1000)
                line = popt[0] * single_mode_pdf(
                    x_vals, *popt[1 : 1 + single_mode_num_params]
                )
                kpl.plot_line(ax, x_vals, line, color=kpl.KplColors.RED)
                line = (1 - popt[0]) * single_mode_pdf(
                    x_vals, *popt[1 + single_mode_num_params :]
                )
                kpl.plot_line(ax, x_vals, line, color=kpl.KplColors.GREEN)
                line = bimodal_pdf(x_vals, *popt)
                kpl.plot_line(ax, x_vals, line, color=kpl.KplColors.BLUE)

            # Threshold line
            if threshold is not None:
                ax.axvline(threshold, color=kpl.KplColors.GRAY, ls="dashed")

            # Add text of the fidelities
            snr_str = (
                f"NV{nv_num}\n"
                f"Readout fidelity: {round(readout_fidelity, 3)}\n"
                f"Charge prep. fidelity {round(prep_fidelity, 3)}"
            )
            kpl.anchored_text(ax, snr_str, kpl.Loc.CENTER_RIGHT, size=kpl.Size.SMALL)

            # if readout_fidelity < 0.8:
            #     kpl.show(block=True)
            # plt.close(fig)
            kpl.show(block=True)
            fig = None

            if fig is not None:
                hist_figs.append(fig)

    print(f"readout_fidelity_list: {readout_fidelity_list}")
    print(f"prep_fidelity_list: {prep_fidelity_list}")
    print(f"red_chi_sq_list: {red_chi_sq_list}")
    print(f"thresholds: {threshold_list}")

    # Report out the results
    threshold_list = np.array(threshold_list)
    readout_fidelity_list = np.array(readout_fidelity_list)
    prep_fidelity_list = np.array(prep_fidelity_list)

    # Scatter readout vs prep fidelity
    fig, ax = plt.subplots()
    kpl.plot_points(ax, readout_fidelity_list, prep_fidelity_list)
    ax.set_xlabel("Readout fidelity")
    ax.set_ylabel("NV- preparation fidelity")

    # Plot prep fidelity vs distance from center
    # coords_key = "laser_INTE_520_aod"
    # distances = []
    # for nv in nv_list:
    #     coords = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
    #     dist = np.sqrt((110 - coords[0]) ** 2 + (110 - coords[1]) ** 2)
    #     distances.append(dist)
    # fig, ax = plt.subplots()
    # kpl.plot_points(ax, distances, prep_fidelity_list)
    # ax.set_xlabel("Distance from center frequencies (MHz)")
    # ax.set_ylabel("NV- preparation fidelity")

    # Report averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_list)
    std_readout_fidelity = np.nanstd(readout_fidelity_list)
    avg_prep_fidelity = np.nanmean(prep_fidelity_list)
    std_prep_fidelity = np.nanstd(prep_fidelity_list)
    str_readout_fidelity = tb.round_for_print(
        avg_readout_fidelity, std_readout_fidelity
    )
    str_prep_fidelity = tb.round_for_print(avg_prep_fidelity, std_prep_fidelity)
    print(f"Average readout fidelity: {str_readout_fidelity}")
    print(f"Average NV- preparation fidelity: {str_prep_fidelity}")

    return hist_figs


def plot_avg_images(raw_data):
    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout_laser = laser_dict["physical_name"]
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

    return img_arrays_to_save, img_figs


def main(
    nv_list,
    num_reps,
    num_runs,
    ion_do_target_inds=None,
    verify_charge_states=False,
    do_plot_histograms=False,
):
    ### Initial setup
    seq_file = "charge_state_histograms.py"
    num_steps = 1

    # Turn the list of NV indices to ionize into a list of True/False for
    # each NV according to whether it should be targeted
    if ion_do_target_inds is None:
        ion_do_target_list = None
    else:
        num_nvs = len(nv_list)
        ion_do_target_list = [ind in ion_do_target_inds for ind in range(num_nvs)]

    if verify_charge_states:
        charge_prep_fn = base_routine.charge_prep_loop
    else:
        charge_prep_fn = None

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(nv_list, VirtualLaserKey.CHARGE_POL)
        )
        ion_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.ION)
        seq_args = [
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
            ion_coords_list,
            ion_do_target_list,
            verify_charge_states,
        ]
        # print("seq_args:", seq_args)
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

    # Images
    try:
        imgs, img_figs = plot_avg_images(raw_data)
        # Save
        sig_img_array, ref_img_array, diff_img_array = imgs
        keys_to_compress = ["sig_img_array", "ref_img_array", "diff_img_array"]
        title_suffixes = ["sig", "ref", "diff"]
        num_figs = len(img_figs)
        for ind in range(num_figs):
            fig = img_figs[ind]
            title = title_suffixes[ind]
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{title}")
            dm.save_figure(fig, file_path)
    except Exception:
        print(traceback.format_exc())
        sig_img_array = None
        ref_img_array = None
        diff_img_array = None
        keys_to_compress = None

    # Histograms
    try:
        hist_figs = process_and_plot(raw_data, do_plot_histograms=do_plot_histograms)
        # Save
        if hist_figs is not None:
            num_nvs = len(nv_list)
            for nv_ind in range(num_nvs):
                fig = hist_figs[nv_ind]
                nv_sig = nv_list[nv_ind]
                nv_name = nv_sig.name
                file_path = dm.get_file_path(__file__, timestamp, nv_name)
                dm.save_figure(fig, file_path)
    except Exception:
        print(traceback.format_exc())

    try:
        del raw_data["img_arrays"]
    except Exception:
        pass

    ### Save raw data

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    raw_data |= {
        "timestamp": timestamp,
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
    # data = dm.get_raw_data(file_id=1733583334808, load_npz=False)
    data = dm.get_raw_data(file_id=1766803842180, load_npz=False)  # 50ms readout
    # data = dm.get_raw_data(file_id=1766834596476, load_npz=False)  # 100ms readout
    process_and_plot(data, do_plot_histograms=True)
    kpl.show(block=True)
