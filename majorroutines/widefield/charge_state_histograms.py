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

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial

from majorroutines.widefield import optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import LaserKey


def create_histogram(sig_counts_list, ref_counts_list, no_title=True):
    try:
        laser_dict = tb.get_laser_dict(LaserKey.CHARGE_READOUT)
        readout = laser_dict["duration"]
        readout_ms = int(readout / 1e6)
        readout_s = readout / 1e9
    except Exception:
        readout_s = 0.05  # MCC
        pass

    ### Histograms

    num_reps = len(ref_counts_list)

    labels = ["With ionization pulse", "Without ionization pulse"]
    colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
    counts_lists = [sig_counts_list, ref_counts_list]
    fig, ax = plt.subplots()
    if not no_title:
        ax.set_title(f"Charge prep hist, {num_reps} reps")
    ax.set_xlabel("Integrated counts")
    ax.set_ylabel("Number of occurrences")
    for ind in range(2):
        counts_list = counts_lists[ind]
        label = labels[ind]
        color = colors[ind]
        # kpl.histogram(ax, counts_list, num_bins, label=labels[ind])
        kpl.histogram(ax, counts_list, label=label, color=color)
    ax.legend()

    # Calculate the normalized separation
    if True:
        noise = np.sqrt(np.var(ref_counts_list) + np.var(sig_counts_list))
        signal = np.mean(ref_counts_list) - np.mean(sig_counts_list)
        snr = signal / noise
        snr_time = snr / np.sqrt(readout_s)
        snr = round(snr, 3)
        snr_time = round(snr_time, 3)
        snr_str = f"SNR:\n{snr} / sqrt(shots)\n{snr_time} / sqrt(s)"
        print(snr_str)
        # kpl.anchored_text(ax, snr_str, "center right", size=kpl.Size.SMALL)
        snr_str = f"SNR: {snr}"
        kpl.anchored_text(ax, snr_str, "center right", size=kpl.Size.SMALL)

    return fig


def poisson_dist(x, rate):
    return (rate**x) * np.exp(-rate) / factorial(x)


def poisson_cdf(x, rate):
    """Cumulative distribution function for poisson pdf. Integrates
    up to and including x"""
    x_floor = int(np.floor(x))
    val = 0
    for ind in range(x_floor):
        val += poisson_dist(ind, rate)
    return val


def bimodal_dist(x, prob_nv0, mean_counts_nv0, mean_counts_nvn):
    prob_nvn = 1 - prob_nv0
    val_nv0 = poisson_dist(x, mean_counts_nv0)
    val_nvn = poisson_dist(x, mean_counts_nvn)
    return prob_nv0 * val_nv0 + prob_nvn * val_nvn


def determine_threshold(counts_list):
    # Use the ref and assume there's some population in both NV- and NV0

    # Histogram the counts
    counts_list = [round(el) for el in counts_list]
    max_count = max(counts_list)
    x_vals = np.linspace(0, max_count, max_count + 1)
    hist, _ = np.histogram(
        counts_list, bins=max_count + 1, range=(0, max_count), density=True
    )

    # Fit the histogram
    prob_nv0_guess = 0.3
    mean_counts_nv0_guess = np.quantile(counts_list, 0.2)
    mean_counts_nvn_guess = np.quantile(counts_list, 0.8)
    guess_params = (prob_nv0_guess, mean_counts_nv0_guess, mean_counts_nvn_guess)
    popt, _ = curve_fit(bimodal_dist, x_vals, hist, p0=guess_params)
    print(popt)

    # Find the optimum threshold by maximizing readout fidelity
    # I.e. find threshold that maximizes:
    # F = (1/2)P(call NV- | actually NV-) + (1/2)P(call NV0 | actually NV0)
    _, mean_counts_nv0, mean_counts_nvn = popt
    mean_counts_nv0 = round(mean_counts_nv0)
    mean_counts_nvn = round(mean_counts_nvn)
    num_steps = mean_counts_nvn - mean_counts_nv0
    thresh_options = np.linspace(
        mean_counts_nv0 + 0.5, mean_counts_nvn - 0.5, num_steps
    )
    fidelities = []
    for val in thresh_options:
        nv0_fid = poisson_cdf(val, mean_counts_nv0)
        nvn_fid = 1 - poisson_cdf(val, mean_counts_nvn)
        fidelities.append((1 / 2) * (nv0_fid + nvn_fid))

    best_fidelity = max(fidelities)
    best_threshold = thresh_options[np.argmax(fidelities)]
    print(f"Optimum threshold: {best_threshold}")
    print(f"Fidelity: {best_fidelity}")

    return popt


def main(
    nv_list,
    num_reps=100,
    pol_duration=None,
    ion_duration=None,
    diff_polarize=False,
    diff_ionize=True,
):
    ### Setup

    kpl.init_kplotlib()

    ### Collect the data

    ret_vals = _collect_data(
        nv_list, num_reps, pol_duration, ion_duration, diff_polarize, diff_ionize
    )
    (
        sig_img_array_list,
        ref_img_array_list,
        sig_img_array,
        ref_img_array,
        diff_img_array,
        timestamp,
    ) = ret_vals

    ### Process

    sig_counts_lists, ref_counts_lists = process_data(
        nv_list, sig_img_array_list, ref_img_array_list
    )

    ### Plot and save

    num_nvs = len(nv_list)
    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]
        fig = create_histogram(sig_counts_list, ref_counts_list)
        nv_sig = nv_list[ind]
        nv_name = nv_sig.name
        file_path = dm.get_file_path(__file__, timestamp, nv_name)
        dm.save_figure(fig, file_path)

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    keys_to_compress = [
        "sig_counts_lists",
        "ref_counts_lists",
        "sig_img_array",
        "ref_img_array",
        "diff_img_array",
    ]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    raw_data = {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "num_reps": num_reps,
        "diff_polarize": diff_polarize,
        "diff_ionize": diff_ionize,
        "sig_counts_lists": sig_counts_lists,
        "ref_counts_lists": ref_counts_lists,
        "counts-units": "photons",
        "sig_img_array": sig_img_array,
        "ref_img_array": ref_img_array,
        "diff_img_array": diff_img_array,
        "img_array-units": "ADUs",
    }
    dm.save_raw_data(raw_data, file_path, keys_to_compress)


def process_data(nv_list, sig_img_array_list, ref_img_array_list):
    # Get the actual num_reps in case something went wrong
    num_reps = len(ref_img_array_list)
    num_nvs = len(nv_list)

    # Get a nice average image for optimization
    avg_img_array = np.sum(ref_img_array_list, axis=0) / num_reps
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    optimize.optimize_pixel_with_img_array(avg_img_array, repr_nv_sig)

    sig_counts_lists = [[] for ind in range(num_nvs)]
    ref_counts_lists = [[] for ind in range(num_nvs)]

    for nv_ind in range(num_nvs):
        nv_sig = nv_list[nv_ind]
        pixel_coords = widefield.get_nv_pixel_coords(nv_sig)
        sig_counts_list = sig_counts_lists[nv_ind]
        ref_counts_list = ref_counts_lists[nv_ind]
        for rep_ind in range(num_reps):
            img_array = sig_img_array_list[rep_ind]
            sig_counts = widefield.integrate_counts_from_adus(img_array, pixel_coords)
            sig_counts_list.append(sig_counts)
            img_array = ref_img_array_list[rep_ind]
            ref_counts = widefield.integrate_counts_from_adus(img_array, pixel_coords)
            ref_counts_list.append(ref_counts)

    sig_counts_lists = np.array(sig_counts_lists)
    ref_counts_lists = np.array(ref_counts_lists)
    return sig_counts_lists, ref_counts_lists


def _collect_data(
    nv_list,
    num_reps=100,
    pol_duration=None,
    ion_duration=None,
    diff_polarize=False,
    diff_ionize=True,
):
    ### Some initial setup

    # First NV to represent the others
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)

    tb.reset_cfm()
    laser_key = LaserKey.CHARGE_READOUT
    optimize.prepare_microscope(repr_nv_sig)
    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()

    laser_dict = tb.get_laser_dict(laser_key)
    readout_laser = laser_dict["name"]
    tb.set_filter(repr_nv_sig, laser_key)

    pos.set_xyz_on_nv(repr_nv_sig)

    ### Load the pulse generator

    readout = laser_dict["duration"]
    readout_ms = readout / 10**6

    seq_args = widefield.get_base_scc_seq_args(nv_list)
    seq_args.extend([pol_duration, ion_duration, diff_polarize, diff_ionize])
    seq_file = "charge_state_histograms.py"

    # print(seq_args)
    # print(seq_file)
    # return

    ### Collect the data

    sig_img_array_list = []
    ref_img_array_list = []

    def rep_fn(rep_ind):
        sig_img_str = camera.read()
        ref_img_str = camera.read()

        sig_img_array = widefield.img_str_to_array(sig_img_str)
        ref_img_array = widefield.img_str_to_array(ref_img_str)

        sig_img_array_list.append(sig_img_array)
        ref_img_array_list.append(ref_img_array)

    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    camera.arm()
    widefield.rep_loop(num_reps, rep_fn)
    camera.disarm()

    ### Process and plot

    sig_img_array = np.sum(sig_img_array_list, axis=0)
    ref_img_array = np.sum(ref_img_array_list, axis=0)
    diff_img_array = sig_img_array - ref_img_array
    sig_img_array = sig_img_array / num_reps
    ref_img_array = ref_img_array / num_reps
    diff_img_array = diff_img_array / num_reps

    img_arrays = [sig_img_array, ref_img_array, diff_img_array]
    title_suffixes = ["sig", "ref", "diff"]
    figs = []
    for ind in range(3):
        img_array = img_arrays[ind]
        title_suffix = title_suffixes[ind]
        fig, ax = plt.subplots()
        title = f"{readout_laser}, {readout_ms} ms, {title_suffix}"
        kpl.imshow(ax, img_array, title=title, cbar_label="ADUs")
        figs.append(fig)

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    nv_name = repr_nv_sig.name
    # Save sub figs
    for ind in range(3):
        fig = figs[ind]
        title_suffix = title_suffixes[ind]
        name = f"{nv_name}-{title_suffix}"
        fig_file_path = dm.get_file_path(__file__, timestamp, name)
        dm.save_figure(fig, fig_file_path)

    return (
        sig_img_array_list,
        ref_img_array_list,
        sig_img_array,
        ref_img_array,
        diff_img_array,
        timestamp,
    )


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = "2023_11_20-17_38_07-johnson-nv0_2023_11_09"
    # data = dm.get_raw_data(file_name)
    data = dm.get_raw_data(file_id=1482405937799, no_npz=False)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    sig_counts_lists = data["sig_counts_lists"]
    ref_counts_lists = data["ref_counts_lists"]
    num_shots = len(sig_counts_lists[0])

    # determine_thresholds(nv_list, sig_counts_lists)

    # x = np.linspace(0, 50, 51)
    # print(bimodal_dist(x, 0.3, 20, 50))

    ### Histograms

    if True:
        for ind in range(num_nvs):
            print()
            nv_sig = nv_list[ind]
            sig_counts_list = sig_counts_lists[ind]
            ref_counts_list = ref_counts_lists[ind]
            fig = create_histogram(sig_counts_list, ref_counts_list)

            ax = fig.gca()
            popt = determine_threshold(ref_counts_list)
            x_vals = np.linspace(0, max(ref_counts_list), 1000)
            kpl.plot_line(ax, x_vals, num_shots * bimodal_dist(x_vals, *popt))

    ### Labeled images

    if False:
        num_nvs = len(nv_list)
        pixel_coords_list = []
        for nv in nv_list:
            coords = nv["pixel_coords"]
            adj_coords = [coords[0] - 7, coords[1] - 26]
            pixel_coords_list.append(adj_coords)

        sig_img_array = np.array(data["sig_img_array"])
        ref_img_array = np.array(data["ref_img_array"])
        diff_img_array = np.array(data["diff_img_array"])
        # sig_counts_list, ref_counts_list = process_data(
        #     nv_list, sig_img_array, ref_img_array
        # )
        sig_img_array_cts = widefield.adus_to_photons(sig_img_array)
        ref_img_array_cts = widefield.adus_to_photons(ref_img_array)
        img_arrays = [
            sig_img_array_cts,
            ref_img_array_cts,
            sig_img_array_cts - ref_img_array_cts,
        ]

        titles = ["With ionization pulse", "Without ionization pulse", "Difference"]
        for ind in range(3):
            img_array = img_arrays[ind]
            fig, ax = plt.subplots()
            title = titles[ind]
            if ind in [0, 1]:
                vmin = 0
                vmax = 0.7
            else:
                vmin = -0.45
                vmax = 0.1
            kpl.imshow(
                ax, img_array, title=title, cbar_label="Counts", vmin=vmin, vmax=vmax
            )

            for ind in range(len(pixel_coords_list)):
                # if ind != 8:
                #     continue
                # print(nv_list[ind]["name"])
                pixel_coords = pixel_coords_list[ind]
                pixel_coords = [el + 1 for el in pixel_coords]
                color = kpl.data_color_cycler[ind]
                # kpl.draw_circle(
                #     ax, pixel_coords, color=color, radius=1.5, outline=True, label=ind
                # )
                kpl.draw_circle(ax, pixel_coords, color=color, radius=9, label=ind)

    plt.show(block=True)
