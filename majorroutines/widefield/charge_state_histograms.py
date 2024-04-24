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

from majorroutines.widefield import base_routine, optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import LaserKey, NVSig


def create_histogram(sig_counts_list, ref_counts_list, no_title=True):
    try:
        laser_dict = tb.get_optics_dict(LaserKey.WIDEFIELD_CHARGE_READOUT)
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
    """counts_list should probably be the ref since we need some population
    in both NV- and NV0"""

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
    # F = (1/2)P(say NV- | actually NV-) + (1/2)P(say NV0 | actually NV0)
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

    return popt, best_threshold


def main(
    nv_list,
    num_reps,
    num_runs,
    charge_prep_verification=True,
    diff_polarize=False,
    diff_ionize=True,
):
    ### Some initial setup
    uwave_ind = 0
    seq_file = "charge_state_histograms.py"
    num_steps = 1

    if charge_prep_verification:
        charge_prep_fn = base_routine.charge_prep_loop
    else:
        charge_prep_fn = base_routine.charge_prep_no_verification

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        pol_coords_list = widefield.get_coords_list(nv_list, LaserKey.CHARGE_POL)
        ion_coords_list = widefield.get_coords_list(nv_list, LaserKey.ION)
        seq_args = [pol_coords_list, ion_coords_list, diff_polarize, diff_ionize]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        save_images=True,
        charge_prep_fn=charge_prep_fn,
    )

    ### Process and plot

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    # file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    # dm.save_raw_data(raw_data, file_path)

    # Images
    laser_key = LaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_optics_dict(laser_key)
    readout_laser = laser_dict["name"]
    readout = laser_dict["duration"]
    readout_ms = readout / 10**6

    img_arrays = raw_data["img_arrays"]
    del raw_data["img_arrays"]
    mean_img_arrays = np.mean(img_arrays, axis=(1, 2))
    sig_img_array = mean_img_arrays[0]
    ref_img_array = mean_img_arrays[1]
    diff_img_array = sig_img_array - ref_img_array
    img_arrays = [sig_img_array, ref_img_array, diff_img_array]
    title_suffixes = ["sig", "ref", "diff"]
    figs = []
    for ind in range(3):
        img_array = img_arrays[ind]
        title_suffix = title_suffixes[ind]
        fig, ax = plt.subplots()
        title = f"{readout_laser}, {readout_ms} ms, {title_suffix}"
        kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
        figs.append(fig)
        file_path = dm.get_file_path(
            __file__, timestamp, f"{repr_nv_name}-{title_suffixes[ind]}"
        )
        dm.save_figure(fig, file_path)

    # Histograms
    num_nvs = len(nv_list)
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]

    num_nvs = len(nv_list)
    threshold_list = []
    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]
        fig = create_histogram(sig_counts_list, ref_counts_list)
        all_counts_list = np.append(sig_counts_list, ref_counts_list)
        _, threshold = determine_threshold(all_counts_list)
        threshold_list.append(threshold)
        nv_sig = nv_list[ind]
        nv_name = nv_sig.name
        file_path = dm.get_file_path(__file__, timestamp, nv_name)
        dm.save_figure(fig, file_path)
    print(threshold_list)

    ### Save and clean up

    keys_to_compress = [
        "sig_img_array",
        "ref_img_array",
        "diff_img_array",
    ]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    raw_data |= {
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
        "img_array-units": "photons",
    }
    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    tb.reset_cfm()

    return raw_data


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1506800366384)  # Yellow on, -60 C, 50 ms readout
    # data = dm.get_raw_data(file_id=1506826857288)  # Yellow off, -60 C, 50 ms readout
    # data = dm.get_raw_data(file_id=1506865734454)  # Yellow off, -65 C, 50 ms readout
    # data = dm.get_raw_data(file_id=1506877254094)  # Yellow off, -60 C, 100 ms readout
    # data = dm.get_raw_data(file_id=1506936719223)  # EM gain 100
    # data = dm.get_raw_data(file_id=1506961050354)  # EM gain 5000
    data = dm.get_raw_data(file_id=1508790988173)  # Background subtraction

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    # sig_counts_lists = np.array(data["sig_counts_lists"])
    # ref_counts_lists = np.array(data["ref_counts_lists"])
    # num_shots = len(sig_counts_lists[0])

    mean_vals = np.array(data["mean_vals"])
    # mean_vals = np.array(data["median_vals"])
    sig_mean_vals = mean_vals[0].flatten()
    ref_mean_vals = mean_vals[1].flatten()
    # sig_mean_vals = widefield.adus_to_photons(mean_vals[0].flatten(), em_gain=5000)
    # ref_mean_vals = widefield.adus_to_photons(mean_vals[1].flatten(), em_gain=5000)
    sig_mean_vals = moving_average(sig_mean_vals, 20)
    ref_mean_vals = moving_average(ref_mean_vals, 20)
    sig_norms = sig_mean_vals / np.mean(sig_mean_vals)
    ref_norms = ref_mean_vals / np.mean(ref_mean_vals)
    fig, ax = plt.subplots()
    kpl.plot_line(ax, range(len(sig_mean_vals)), sig_mean_vals, label="Sig")
    kpl.plot_line(ax, range(len(ref_mean_vals)), ref_mean_vals, label="Ref")
    ax.set_xlabel("Shot index")
    ax.set_ylabel("Average photon count / pixel")
    ax.legend()
    kpl.show(block=True)

    img_arrays = np.array(data["img_arrays"])
    img_arrays_photons = widefield.adus_to_photons(img_arrays)

    fig, ax = plt.subplots()
    kpl.imshow(ax, np.mean(img_arrays_photons, axis=(0, 1, 2, 3)))

    # widefield.charge_state_mle(nv_list, np.mean(img_arrays, axis=(0, 1, 2, 3)))

    num_runs = data["num_runs"]
    num_steps = data["num_steps"]
    num_reps = data["num_reps"]
    num_shots = num_runs * num_steps * num_reps

    states = np.array([0 for ind in range(num_nvs)])
    states_thresh = np.array([0 for ind in range(num_nvs)])

    start = time.time()
    for run_ind in range(num_runs):
        for step_ind in range(num_steps):
            for rep_ind in range(num_reps):
                img_array = img_arrays[0, run_ind, step_ind, rep_ind]
                states_list, states_thresh_list = widefield.charge_state_mle(
                    nv_list, img_array
                )
                states += states_list
                states_thresh += states_thresh_list
    stop = time.time()
    print(stop - start)

    print(states / num_shots)
    print(states_thresh / num_shots)

    kpl.show(block=True)
