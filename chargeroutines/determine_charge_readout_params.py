# -*- coding: utf-8 -*-
"""
Modified version of determine_charge_readout_dur. Runs a single experiment
at a given power and use time-resolved counting to determine histograms at
difference readout durations.

Created on Thu Apr 22 14:09:39 2021

@author: mccambria
"""

import copy
import numpy as np
from random import shuffle

import matplotlib.pyplot as plt
import labrad
import time
import sys
import random
import scipy.stats as stats
import utils.tool_belt as tool_belt
import chargeroutines.photonstatistics as model
optimization_type = tool_belt.get_optimization_style()
if optimization_type == 'DISCRETE':
    import majorroutines.optimize_digital as optimize
if optimization_type == 'CONTINUOUS':
    import majorroutines.optimize as optimize


# %%


def calc_histogram(nv0, nvm, dur, bins=None):

    # Counts are in us, readout is in ns
    dur_us = dur / 1e3
    # print(nv0)
    nv0_counts = [
        np.count_nonzero(np.array(rep) < dur_us) for rep in nv0
    ]  # ???
    nvm_counts = [
        np.count_nonzero(np.array(rep) < dur_us) for rep in nvm
    ]  # ???
    # print(nv0_counts)
    max_0 = max(nv0_counts)
    max_m = max(nvm_counts)
    if bins == None:

        occur_0, bin_edges_0 = np.histogram(
            nv0_counts, np.linspace(0, max_0, max_0 + 1)  # 200)  #
        )
        occur_m, bin_edge_m = np.histogram(
            nvm_counts, np.linspace(0, max_m, max_m + 1)  # 200)  #
        )
    elif bins != None:
        occur_0, bin_edges_0 = np.histogram(
            nv0_counts, bins  # np.linspace(0, max_0, max_0 + 1) #200)  #
        )
        occur_m, bin_edge_m = np.histogram(
            nvm_counts, bins  # np.linspace(0, max_m,  max_m + 1) #200)  #
        )

    # Histogram returns bin edges. A bin is defined with the first point
    # inclusive and the last exclusive - eg a count a 2 will fall into
    # bin [2,3) - so just drop the last bin edge for our x vals
    x_vals_0 = bin_edges_0[:-1]
    x_vals_m = bin_edge_m[:-1]

    return occur_0, x_vals_0, occur_m, x_vals_m


def calc_overlap(occur_0, x_vals_0, occur_m, x_vals_m, num_reps):

    min_max_x_vals = int(min(x_vals_0[-1], x_vals_m[-1]))
    occur_0_clip = occur_0[0:min_max_x_vals]
    occur_m_clip = occur_m[0:min_max_x_vals]
    overlap = np.sum(np.minimum(occur_0_clip, occur_m_clip))
    fractional_overlap = overlap / num_reps
    return fractional_overlap


def calc_separation(
    occur_0,
    x_vals_0,
    occur_m,
    x_vals_m,
    num_reps,
    report_means=False,
    report_stds=False,
):

    mean_0 = sum(occur_0 * x_vals_0) / num_reps
    std_0 = np.sqrt(sum(occur_0 * (x_vals_0 - mean_0) ** 2) / (num_reps - 1))
    mean_m = sum(occur_m * x_vals_m) / num_reps
    std_m = np.sqrt(sum(occur_m * (x_vals_m - mean_m) ** 2) / (num_reps - 1))
    avg_std = (std_0 + std_m) / 2
    norm_sep = (mean_m - mean_0) / avg_std
    if report_means:
        print(mean_0)
        print(mean_m)
    if report_stds:
        print(std_0)
        print(std_m)
    return norm_sep


def single_nv_photon_statistics_model(readout_time, NV0, NVm, do_plot=True):
    """
    A function to take the NV histograms after red and green initialization,
    and use a model to plot the expected histograms if the NV is perfectly
    initialized in NV- or NV0

    for the fit,
    g0 =  Ionization rate from NV- to NV0
    g1 = recombination rate from NV0 to NV-
    y1 = fluorescnece rate of NV1
    y0 = Fluorescence rate of NV0
    """
    NV0_hist = np.array(NV0)
    NVm_hist = np.array(NVm)
    tR = readout_time
    combined_hist = NVm_hist.tolist() + NV0_hist.tolist()
    random.shuffle(combined_hist)

    # fit = [g0,g1,y1,y0]
    guess = [10 * 10 ** -4, 100 * 10 ** -4, 1000 * 10 ** -4, 500 * 10 ** -4]
    fit, dev = model.get_curve_fit(tR, 0, combined_hist, guess)

    if do_plot:
        u_value0, freq0 = model.get_Probability_distribution(NV0_hist.tolist())
        u_valuem, freqm = model.get_Probability_distribution(NVm_hist.tolist())
        u_value2, freq2 = model.get_Probability_distribution(combined_hist)
        curve = model.get_photon_distribution_curve(
            tR, u_value2, fit[0], fit[1], fit[2], fit[3]
        )

        A1, A1pcov = model.get_curve_fit_to_weight(
            tR, 0, NV0_hist.tolist(), [0.5], fit
        )
        A2, A2pcov = model.get_curve_fit_to_weight(
            tR, 0, NVm_hist.tolist(), [0.5], fit
        )

        nv0_curve = model.get_photon_distribution_curve_weight(
            u_value0, tR, fit[0], fit[1], fit[2], fit[3], A1[0]
        )
        nvm_curve = model.get_photon_distribution_curve_weight(
            u_valuem, tR, fit[0], fit[1], fit[2], fit[3], A2[0]
        )
        fig4, ax = plt.subplots()
        ax.plot(u_value0, 0.5 * np.array(freq0), "-ro")
        ax.plot(u_valuem, 0.5 * np.array(freqm), "-go")
        ax.plot(u_value2, freq2, "-bo")
        ax.plot(u_value2, curve)
        ax.plot(u_valuem, 0.5 * np.array(nvm_curve), "green")
        ax.plot(u_value0, 0.5 * np.array(nv0_curve), "red")
        textstr = "\n".join(
            (
                r"$g_0(s^{-1}) =%.2f$" % (fit[0] * 10 ** 3,),
                r"$g_1(s^{-1})  =%.2f$" % (fit[1] * 10 ** 3,),
                r"$y_0 =%.2f$" % (fit[3],),
                r"$y_1 =%.2f$" % (fit[2],),
            )
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.6,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )
        plt.xlabel("Number of counts")
        plt.ylabel("Probability Density")
        plt.show()
    return fit


def calculate_threshold_with_model(
    readout_time, nv0_array, nvm_array, max_x_val, power, nd_filter=None,plot_model_hists=True
):
    """
    Using the histograms of the NV- and NV0 measurement, and modeling them as
    an NV perfectly prepared in either NV- or NV0, detemines the optimum
    value of single shot counts to determine either NV- or NV0.

    the fit finds
    mu_0 = the mean counts of NV0
    mu_m = the mean counts of NV-
    fidelity = given the threshold, tthe fidelity is related to how accurate
        we can identify the charge state from a single shot measurement.
        Best to shoot for values of > 80%
    threshold = the number of counts that, above this value, identify the charge
        state as NV-. And below this value, identify as NV0.
    """
    tR = readout_time / 10 ** 6
    tR_us = readout_time / 10 ** 3
    fit_rate = single_nv_photon_statistics_model(tR, nv0_array, nvm_array,do_plot=plot_model_hists)
    max_x_val = int(max_x_val)
    x_data = np.linspace(0, 100, 101)
    thresh_para = model.calculate_threshold(tR, x_data, fit_rate)
    mu_0 = fit_rate[3] * tR
    mu_m = fit_rate[2] * tR
    fidelity = thresh_para[1]
    threshold = thresh_para[0]
    # print(title_text)
    print("Threshold: {} counts, fidelity: {:.3f}".format(threshold, fidelity))
    
    if plot_model_hists:

        plot_x_data = np.linspace(0, max_x_val, max_x_val + 1)
        fig3, ax = plt.subplots()
        ax.plot(
            plot_x_data,
            model.get_PhotonNV0_list(plot_x_data, tR, fit_rate, 0.5),
            "-o",
        )
        ax.plot(
            plot_x_data,
            model.get_PhotonNVm_list(plot_x_data, tR, fit_rate, 0.5),
            "-o",
        )
        plt.axvline(x=thresh_para[0], color="red")
        # mu_0 = fit_rate[3] * tR
        # mu_m = fit_rate[2] * tR
        textstr = "\n".join(
            (
                r"$\mu_0=%.2f$" % (mu_0),
                r"$\mu_-=%.2f$" % (mu_m),
                r"$fidelity =%.2f$" % (thresh_para[1]),
                r"$threshold = %.1f$" % (thresh_para[0],),
            )
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.65,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )
        if nd_filter:
            title_text = "{} us readout, {} V, {}".format(
                int(tR_us), power, nd_filter
            )
        else:
            title_text = "{} us readout, {} V".format(int(tR_us), power)
        ax.set_title(title_text)
        plt.xlabel("Number of counts")
        plt.ylabel("Probability Density")
        return threshold, fidelity, mu_0, mu_m, fig3
    
    else:
        # print('i made it here too')
        return threshold, fidelity, mu_0, mu_m, ''
    


def calculate_threshold_no_model(
    readout_time,
    nv0_hist,
    nvm_hist,
    mu_0,
    mu_m,
    x_vals_0,
    x_vals_m,
    power,
    nd_filter=None,
):

    thresh, fid = model.calculate_threshold_from_experiment(
        x_vals_0, x_vals_m, mu_0, mu_m, nv0_hist, nvm_hist
    )

    fig3, ax = plt.subplots(1, 1)
    ax.plot(x_vals_0, nv0_hist, "r-o", label="Test red pulse")
    ax.plot(x_vals_m, nvm_hist, "g-o", label="Test green pulse")
    ax.set_xlabel("Counts")
    ax.set_ylabel("Occur.")
    plt.axvline(x=thresh, color="red")
    textstr = "\n".join(
        (
            r"$\mu_0=%.2f$" % (mu_0),
            r"$\mu_-=%.2f$" % (mu_m),
            r"$threshold = %.1f$" % (thresh),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.65,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )
    if nd_filter:
        title_text = "{} us readout, {} V, {}".format(
            int(readout_time / 1e3), power, nd_filter
        )
    else:
        title_text = "{} us readout, {} V".format(
            int(readout_time / 1e3), power
        )
    ax.set_title(title_text)
    return thresh, fid, fig3


def plot_threshold(
    nv_sig,
    readout_dur,
    nv0_counts,
    nvm_counts,
    power,
    fit_threshold_full_model=False,
    nd_filter=None,
    do_save=False,
    plot_model_hists=True,
    bins=None,
):

    """
    determine the number of counts that acts as the threshold of counts to
    determine the charge state in a single shot.

    Using the full photon statistics model can take a while
    (and if the histograms are not well seperated, then it doesn't work)
    Set the 'fit_threshold_full_model' to False, and the threshold will be
    roughly estimated using the experimental histograms.

    Note on the photon statistics model:
    Sam Li had written code based of a paper by B.J. Shields
    (https://link.aps.org/doi/10.1103/PhysRevLett.114.136402)
    to estimate the optimum
    threshold value of counts to determine if a single shot is NV0 or NV-.
    It uses a model for singla NV fluorescene from either NV- or NV0 based on
    the measured histogram, and then from that model can extract the threshold.
    """

    occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(
        nv0_counts, nvm_counts, readout_dur,bins,
    )
    
    max_x_val = max(list(x_vals_0) + list(x_vals_m)) + 10

    num_reps = len(nv0_counts)
    mean_0 = sum(occur_0 * x_vals_0) / num_reps
    mean_m = sum(occur_m * x_vals_m) / num_reps

    if fit_threshold_full_model:
        # print('i made it here')
        dur_us = readout_dur / 1e3
        nv0_counts_list = [
            np.count_nonzero(np.array(rep) < dur_us) for rep in nv0_counts
        ]
        nvm_counts_list = [
            np.count_nonzero(np.array(rep) < dur_us) for rep in nvm_counts
        ]
        threshold, fidelity, mu_0, mu_m, fig = calculate_threshold_with_model(
            readout_dur,
            nv0_counts_list,
            nvm_counts_list,
            max_x_val,
            power,
            nd_filter,
            plot_model_hists
        )
    else:
        threshold, fidelity, fig = calculate_threshold_no_model(
            readout_dur,
            occur_0,
            occur_m,
            mean_0,
            mean_m,
            x_vals_0,
            x_vals_m,
            power,
            nd_filter,
        )

    timestamp = tool_belt.get_time_stamp()

    if do_save and plot_model_hists:
        
        file_path = tool_belt.get_file_path(
            __file__, timestamp, nv_sig["name"] + "-threshold"
        )
        tool_belt.save_figure(fig, file_path)
    return threshold, fidelity, nv0_counts,nvm_counts


def determine_opti_readout_dur(nv0, nvm, max_readout_dur,exp_dur=0,bins=None):

    if max_readout_dur <= 1000e6:
        readout_dur_linspace = np.arange(1e6, max_readout_dur, 1e6)
    else:
        readout_dur_linspace = np.arange(10e6, max_readout_dur, 10e6)

    # Round to nearest ms
    # readout_dur_linspace = [
    #     int(1e6 * round(val / 1e6)) for val in readout_dur_linspace
    # ]
    #round to nearest us
    readout_dur_linspace = [
        int(1e3 * round(val / 1e3)) for val in readout_dur_linspace
    ]  
    # print(readout_dur_linspace)

    sensitivities = []
    separations = []
    num_reps = len(nv0)

    for dur in readout_dur_linspace:
        occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(
            nv0, nvm, dur, bins
        )
        separation = calc_separation(
            occur_0, x_vals_0, occur_m, x_vals_m, num_reps
        )
        separations.append(separation)
        # print(dur)
        sensitivities.append(separation / np.sqrt((dur + exp_dur) * 10 **(-6)))

    max_sensitivity = max(sensitivities)
    opti_readout_dur_ind = sensitivities.index(max_sensitivity)
    opti_readout_dur = readout_dur_linspace[opti_readout_dur_ind]
    print(np.array(separations))
    print(np.array(sensitivities))
    plt.figure()
    plt.scatter(readout_dur_linspace,sensitivities)
    plt.show()

    return opti_readout_dur


def plot_histogram(
    nv_sig,
    nv0,
    nvm,
    dur,
    power,
    bins=None,
    # total_seq_time_sec,
    nd_filter=None,
    do_save=True,
    report_means=True,
    report_stds=True,
):

    num_reps = len(nv0)
    occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(nv0, nvm, dur, bins)
    # overlap = calc_overlap(occur_0, x_vals_0, occur_m, x_vals_m, num_reps)
    # print("fractional overlap: {}".format(overlap))
    separation = calc_separation(
        occur_0,
        x_vals_0,
        occur_m,
        x_vals_m,
        num_reps,
        report_means,
        report_stds,
    )
    sensitivity = separation * np.sqrt(dur * 10 ** 9)
    print(f"Normalized separation / sqrt(Hz): {sensitivity}")

    fig_hist, ax = plt.subplots(1, 1)
    # print(x_vals_0.tolist())
    # print(occur_0.tolist())
    ax.plot(x_vals_0, occur_0, "r-o", label="Initial red pulse")
    ax.plot(x_vals_m, occur_m, "g-o", label="Initial green pulse")
    ax.set_xlabel("Counts")
    ax.set_ylabel("Occur.")
    ax.set_xlim(0)
    # ax.set_title("{} ms readout, {} V".format(int(dur / 1e6), power))
    
    if nd_filter:
        title_text = "{} us readout, {} V, {}".format(
            int(dur / 1e3), power, nd_filter
        )
    else:
        title_text = "{} us readout, {} V".format(int(dur / 1e3), power)
    ax.set_title(title_text)
    ax.legend()
    fig_hist.tight_layout()

    if do_save:
        timestamp = tool_belt.get_time_stamp()
        file_path = tool_belt.get_file_path(
            __file__, timestamp, nv_sig["name"] + "_histogram"
        )
        tool_belt.save_figure(fig_hist, file_path)
        # Sleep for a second so we don't overwrite any other histograms
        time.sleep(1.1)


def process_timetags(apd_gate_channel, timetags, channels):

    processed_timetags = []
    gate_open_channel = apd_gate_channel
    gate_close_channel = -gate_open_channel

    channels_array = np.array(channels)
    gate_open_inds = np.where(channels_array == gate_open_channel)[0]
    gate_close_inds = np.where(channels_array == gate_close_channel)[0]

    num_reps = len(gate_open_inds)
    for rep_ind in range(num_reps):
        open_ind = gate_open_inds[rep_ind]
        close_ind = gate_close_inds[rep_ind]
        open_timetag = timetags[open_ind]
        rep_processed_timetags = timetags[open_ind + 1 : close_ind]
        rep_processed_timetags = [
            val - open_timetag for val in rep_processed_timetags
        ]
        processed_timetags.append(rep_processed_timetags)

    return processed_timetags


def measure_histograms_sub(
    cxn, nv_sig, opti_nv_sig, seq_file, seq_args, apd_indices, num_reps
):
    
    tagger_server = tool_belt.get_tagger_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)

    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(seq_file, seq_args_string)
    period = ret_vals[0]
    period_sec = period / 10 ** 9

    # Some initial parameters
    opti_period = 2.5 * 60
    num_reps_per_cycle = round(opti_period / period_sec)
    num_reps_remaining = num_reps
    timetags = []
    channels = []

    while num_reps_remaining > 0:

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        coords = nv_sig["coords"]
        opti_coords_list = []
        opti_coords = optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)
        drift = tool_belt.get_drift()
        adjusted_nv_coords = coords + np.array(drift)
        tool_belt.set_xyz(cxn, adjusted_nv_coords)
        # print(num_reps_remaining)

        # Make sure the lasers are at the right powers
        # Initial Calculation and setup
        tool_belt.set_filter(cxn, nv_sig, "nv0_prep_laser")
        tool_belt.set_filter(cxn, nv_sig, "nv-_prep_laser")
        tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
        _ = tool_belt.set_laser_power(cxn, nv_sig, "nv0_prep_laser")
        _ = tool_belt.set_laser_power(cxn, nv_sig, "nv-_prep_laser")
        _ = tool_belt.set_laser_power(cxn, nv_sig, "charge_readout_laser")

        # Load the APD
        tagger_server.start_tag_stream(apd_indices)
        tagger_server.clear_buffer()

        # Run the sequence
        if num_reps_remaining > num_reps_per_cycle:
            num_reps_to_run = num_reps_per_cycle
        else:
            num_reps_to_run = num_reps_remaining
        
        print(seq_args_string,num_reps_to_run)
        pulsegen_server.stream_immediate(
            seq_file, num_reps_to_run, seq_args_string
        )
        # print(num_reps_to_run)

        ret_vals = tagger_server.read_tag_stream(num_reps_to_run)
        buffer_timetags, buffer_channels = ret_vals
        # We don't care about picosecond resolution here, so just round to us
        # We also don't care about the offset value, so subtract that off
        if len(timetags) == 0:
            offset = np.int64(buffer_timetags[0])
        buffer_timetags = [
            int((np.int64(val) - offset) / 1e6) for val in buffer_timetags
        ]
        timetags.extend(buffer_timetags)
        channels.extend(buffer_channels)

        tagger_server.stop_tag_stream()

        num_reps_remaining -= num_reps_per_cycle

    return timetags, channels, period_sec


# Apply a gren or red pulse, then measure the counts under yellow illumination.
# Repeat num_reps number of times and returns the list of counts after red illumination, then green illumination
# Use with DM on red and green
def measure_histograms(nv_sig, opti_nv_sig, apd_indices, num_reps,extra_green_initialization):

    with labrad.connect() as cxn:
        nv0, nvm, total_seq_time_sec = measure_histograms_with_cxn(
            cxn, nv_sig, opti_nv_sig, apd_indices, num_reps,extra_green_initialization
        )

    return nv0, nvm, total_seq_time_sec


def measure_histograms_with_cxn(
    cxn, nv_sig, opti_nv_sig, apd_indices, num_reps,extra_green_initialization
):
    # Only support a single APD for now
    apd_index = apd_indices[0]

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv-_prep_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv0_prep_laser")

    readout_laser_power = tool_belt.set_laser_power(
        cxn, nv_sig, "charge_readout_laser"
    )

    readout_pulse_time = nv_sig["charge_readout_dur"]

    # Pulse sequence to do a single pulse followed by readout
    readout_on_2nd_pulse = 2
    
    if extra_green_initialization:
        seq_file = "simple_readout_three_pulse.py"
        first_init_laser_dur = 50e3
        first_init_laser_key = nv_sig["nv-_prep_laser"]
        first_init_laser_power =1
        
        gen_seq_args = lambda init_laser: [
            first_init_laser_dur,
            nv_sig["{}_dur".format(init_laser)],
            readout_pulse_time,
            first_init_laser_key,
            nv_sig[init_laser],
            nv_sig["charge_readout_laser"],
            first_init_laser_power,
            tool_belt.set_laser_power(cxn, nv_sig, init_laser),
            readout_laser_power,
            2,
            apd_index,
        ]
        
    else:
        seq_file = "simple_readout_two_pulse.py"
        gen_seq_args = lambda init_laser: [
            nv_sig["{}_dur".format(init_laser)],
            readout_pulse_time,
            nv_sig[init_laser],
            nv_sig["charge_readout_laser"],
            tool_belt.set_laser_power(cxn, nv_sig, init_laser),
            readout_laser_power,
            2,
            apd_index,
        ]
    # seq_args = gen_seq_args("nv0_prep_laser")
    # print(seq_args)
    # return

    apd_gate_channel = tool_belt.get_apd_gate_channel(cxn, apd_index)

    # Green measurement
    seq_args = gen_seq_args("nv-_prep_laser")
    print(seq_args)
    timetags, channels, period_sec = measure_histograms_sub(
        cxn, nv_sig, opti_nv_sig, seq_file, seq_args, apd_indices, num_reps
    )
    nvm = process_timetags(apd_gate_channel, timetags, channels)

    # Red measurement
    seq_args = gen_seq_args("nv0_prep_laser")
    print(seq_args)
    timetags, channels, period_sec = measure_histograms_sub(
        cxn, nv_sig, opti_nv_sig, seq_file, seq_args, apd_indices, num_reps
    )
    nv0 = process_timetags(apd_gate_channel, timetags, channels)

    tool_belt.reset_cfm(cxn)

    return nv0, nvm, period_sec * 2


def determine_readout_dur_power(
    nv_sig,
    opti_nv_sig,
    apd_indices,
    num_reps=500,
    max_readout_dur=1e9,
    bins=None,
    readout_powers=None,
    plot_readout_durs=None,
    fit_threshold_full_model=False,
    extra_green_initialization=False,
):

    if readout_powers is None:
        readout_powers = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]

    tool_belt.init_safe_stop()

    for p in readout_powers:

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        nv0_power = []
        nvm_power = []

        nv_sig_copy = copy.deepcopy(nv_sig)
        nv_sig_copy["charge_readout_dur"] = max_readout_dur
        nv_sig_copy["charge_readout_laser_power"] = p

        nv0, nvm, total_seq_time_sec = measure_histograms(
            nv_sig_copy, opti_nv_sig, apd_indices, num_reps,extra_green_initialization
        )
        nv0_power.append(nv0)
        nvm_power.append(nvm)

        timestamp = tool_belt.get_time_stamp()
        file_path = tool_belt.get_file_path(
            __file__, timestamp, nv_sig["name"]
        )
        raw_data = {
            "timestamp": timestamp,
            "nv_sig": nv_sig_copy,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            "num_reps": num_reps,
            "nv0": nv0,
            "nv0-units": "list(list(us))",
            "nvm": nvm,
            "nvm-units": "list(list(us))",
        }

        tool_belt.save_raw_data(raw_data, file_path)

        if plot_readout_durs is not None:
            for dur in plot_readout_durs:
                nd_filter = nv_sig["charge_readout_laser_filter"]

                plot_histogram(nv_sig, nv0, nvm, dur, p, nd_filter=nd_filter)

                if fit_threshold_full_model:
                    print(
                        "Calculating threshold values...\nMay take up to a few"
                        " minutes..."
                    )
                plot_threshold(
                    nv_sig,
                    dur,
                    nv0,
                    nvm,
                    p,
                    fit_threshold_full_model,
                    nd_filter=nd_filter,
                    do_save=True,
                )

        print("data collected!")

    return

def measure_reinit_spin_dur(nv_sig, apd_indices, num_reps,state):
    """
    not finished
    """

    with labrad.connect() as cxn:
        sig_counts = measure_reion_dur_with_cxn(cxn, nv_sig, apd_indices, num_reps,state)
        
    return sig_counts

def measure_reinit_spin_dur_cxn(cxn, nv_sig, apd_indices, num_reps,state):

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup    
    tagger_server = tool_belt.get_tagger_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)
    
    readout_time = nv_sig['spin_readout_dur']
    
    # tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv-_reionization_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv0_prep_laser")
        
    # readout_time = nv_sig['charge_readout_dur']
    nvm_reion_time = nv_sig['nv-_reionization_dur']
    spin_reinit_time = nv_sig['spin_reinit_laser_dur']
    
    green_laser_name = nv_sig['nv-_reionization_laser']
    red_laser_name = nv_sig['nv0_prep_laser']
    # yellow_laser_name = nv_sig['charge_readout_laser']
    # sig_gen_name = tool_belt.get_signal_generator_name_no_cxn(state)
    
    num_reps = int(num_reps)
    opti_coords_list = []
    
    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    
    sig_gen_name = tool_belt.get_signal_generator_name_no_cxn(state)    
    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.set_amp(uwave_power)
    pi_pulse = tool_belt.get_pi_pulse_dur(nv_sig['rabi_{}'.format(state.value)])


# first_init_pulse_time, init_pulse_time, readout_time, first_init_laser_key, init_laser_key, readout_laser_key,\
#   first_init_laser_power,init_laser_power, read_laser_power, readout_on_pulse_ind, apd_index  = args
    # Estimate the lenth of the sequance        
# (
#     readout_time, reion_time, ion_time, tau, shelf_time, uwave_tau_max,
#     green_laser_name, yellow_laser_name, red_laser_name,
#     sig_gen, apd_index, reion_power, ion_power, shelf_power, readout_power,
# ) = args     
    file_name = 'rabi_scc.py'        
    seq_args = [
        readout_time,
        nvm_reion_time,
        spin_reinit_time,
        pi_pulse,
        0,
        pi_pulse,
        green_laser_name,
        green_laser_name,
        green_laser_name,
        sig_gen_name,
        apd_indices[0],
        tool_belt.set_laser_power(cxn, nv_sig, 'spin_reinit_laser'),
        tool_belt.set_laser_power(cxn, nv_sig, 'spin_laser'),
        tool_belt.set_laser_power(cxn, nv_sig, 'spin_laser'),
        tool_belt.set_laser_power(cxn, nv_sig, 'spin_laser'),
        ]

    seq_args_string = tool_belt.encode_seq_args(seq_args)
    
    print(seq_args)
    ret_vals = pulsegen_server.stream_load(file_name, seq_args_string)
      
    
    seq_time = int(ret_vals[0])
    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  #s

    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    opti_coords_list.append(opti_coords)
    
    # Load the APD
    tagger_server.start_tag_stream(apd_indices)

    pulsegen_server.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = tagger_server.read_counter_separate_gates(1)
    sample_counts = new_counts[0]

    count = sum(sample_counts[0:2])
    sig_counts = new_counts[0]
    ref_counts = new_counts[1]
    # print(sample_counts)
    
    tagger_server.stop_tag_stream()
    tool_belt.reset_cfm(cxn)

    return sig_counts

def measure_reion_dur(nv_sig, apd_indices, num_reps):

    with labrad.connect() as cxn:
        sig_counts = measure_reion_dur_with_cxn(cxn, nv_sig, apd_indices, num_reps)
        
    return sig_counts

def measure_reion_dur_with_cxn(cxn, nv_sig, apd_indices, num_reps):

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup    
    tagger_server = tool_belt.get_tagger_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)
    
    tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv-_reionization_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv0_prep_laser")
        
    readout_time = nv_sig['charge_readout_dur']
    nvm_reion_time = nv_sig['nv-_reionization_dur']
    nv0_init_time = nv_sig['nv0_prep_laser_dur']
    
    green_laser_name = nv_sig['nv-_reionization_laser']
    red_laser_name = nv_sig['nv0_prep_laser']
    yellow_laser_name = nv_sig['charge_readout_laser']
    # sig_gen_name = tool_belt.get_signal_generator_name_no_cxn(state)
    
    num_reps = int(num_reps)
    opti_coords_list = []


# first_init_pulse_time, init_pulse_time, readout_time, first_init_laser_key, init_laser_key, readout_laser_key,\
#   first_init_laser_power,init_laser_power, read_laser_power, readout_on_pulse_ind, apd_index  = args
    # Estimate the lenth of the sequance            
    file_name = 'simple_readout_three_pulse.py'        
    seq_args = [
        nv0_init_time,
        nvm_reion_time,
        readout_time,
        red_laser_name,
        green_laser_name,
        yellow_laser_name,
        tool_belt.set_laser_power(cxn, nv_sig, 'nv0_prep_laser'),
        tool_belt.set_laser_power(cxn, nv_sig, 'nv-_reionization_laser'),
        tool_belt.set_laser_power(cxn, nv_sig, 'charge_readout_laser'),
        2,
        apd_indices[0]]

    seq_args_string = tool_belt.encode_seq_args(seq_args)
    
    print(seq_args)
    ret_vals = pulsegen_server.stream_load(file_name, seq_args_string)
      
    
    seq_time = int(ret_vals[0])
    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  #s

    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    opti_coords_list.append(opti_coords)
    
    # Load the APD
    tagger_server.start_tag_stream(apd_indices)

    pulsegen_server.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = tagger_server.read_counter_simple(num_reps)
    sig_counts = new_counts
    # print(sample_counts)
    
    tagger_server.stop_tag_stream()
    tool_belt.reset_cfm(cxn)

    return sig_counts

def plot_reion_dur(reion_durs, sig_counts_array, sig_counts_ste_array, title):
    
    fig = plt.figure()
    
    plt.scatter(reion_durs,sig_counts_array)
    plt.errorbar(reion_durs,sig_counts_array,yerr=sig_counts_ste_array)
    plt.title(title)
    plt.xlabel('NV- Initialization Pulse Duration')
    plt.ylabel('Counts')
    
    plt.show()
    
    return fig

def determine_reion_dur(nv_sig, apd_indices, num_reps, reion_durs):
    
    num_steps = len(reion_durs)
    
    # create some arrays for data
    sig_counts_array = np.zeros(num_steps)
    sig_counts_ste_array = np.copy(sig_counts_array)
    ref_counts_array = np.copy(sig_counts_array)
    ref_counts_ste_array = np.copy(sig_counts_array)
    snr_array = np.copy(sig_counts_array)
    

    dur_ind_master_list = []
    
    dur_ind_list = list(range(0, num_steps))
    shuffle(dur_ind_list)
    print(reion_durs)
    # Step through the pulse lengths for the test laser
    for ind in dur_ind_list:
        t = reion_durs[ind]
        dur_ind_master_list.append(ind)
        print('Reionization dur: {} ns'.format(t))
        nv_sig_copy = copy.deepcopy(nv_sig)
        nv_sig_copy['nv-_reionization_dur'] = t
        sig_counts = measure_reion_dur(nv_sig_copy, apd_indices, num_reps)
        # print('measured: ',sig_counts)
        
        sig_counts_ste = stats.sem(sig_counts)
            
        sig_counts_array[ind] = np.average(sig_counts)
        sig_counts_ste_array[ind] = sig_counts_ste
        
        # avg_snr = tool_belt.calc_snr(sig_counts, ref_counts)
        # snr_array[ind] = avg_snr
 
    #plot
    title = 'Sweep NV- initialization pulse duration'
    fig = plot_reion_dur(reion_durs, sig_counts_array, sig_counts_ste_array, title)
    # Save
    
    reion_durs = np.array(reion_durs)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'reion_durs': reion_durs.tolist(),
            'reion_durs-units': 'ns',
            'num_reps':num_reps,
            'sig_counts_array': sig_counts_array.tolist(),
            'sig_counts_ste_array': sig_counts_ste_array.tolist(),
            'dur_ind_master_list': dur_ind_master_list
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'] + '-reion_pulse_dur')
    
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)
    
    print(' \nRoutine complete!')
    return

def determine_reinit_spin_dur(nv_sig, apd_indices, num_reps, reinit_durs):
    
    num_steps = len(reinit_durs)
    
    # create some arrays for data
    sig_counts_array = np.zeros(num_steps)
    sig_counts_ste_array = np.copy(sig_counts_array)
    ref_counts_array = np.copy(sig_counts_array)
    ref_counts_ste_array = np.copy(sig_counts_array)
    snr_array = np.copy(sig_counts_array)
    

    dur_ind_master_list = []
    
    dur_ind_list = list(range(0, num_steps))
    shuffle(dur_ind_list)
    print(reinit_durs)
    # Step through the pulse lengths for the test laser
    for ind in dur_ind_list:
        t = reinit_durs[ind]
        dur_ind_master_list.append(ind)
        print('m_s=0 reinitialization dur: {} ns'.format(t))
        nv_sig_copy = copy.deepcopy(nv_sig)
        nv_sig_copy['spin_reinit_laser_dur'] = t
        sig_counts = measure_reinit_spin_dur(nv_sig_copy, apd_indices, num_reps)
        # print('measured: ',sig_counts)
        
        sig_counts_ste = stats.sem(sig_counts)
            
        sig_counts_array[ind] = np.average(sig_counts)
        sig_counts_ste_array[ind] = sig_counts_ste
        
        # avg_snr = tool_belt.calc_snr(sig_counts, ref_counts)
        # snr_array[ind] = avg_snr
 
    #plot
    title = 'Sweep m_s = 0 re-initialization pulse duration'
    fig = plot_reion_dur(reinit_durs, sig_counts_array, sig_counts_ste_array, title)
    # Save
    
    reinit_durs = np.array(reinit_durs)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'reinit_durs': reinit_durs.tolist(),
            'reinit_durs-units': 'ns',
            'num_reps':num_reps,
            'sig_counts_array': sig_counts_array.tolist(),
            'sig_counts_ste_array': sig_counts_ste_array.tolist(),
            'dur_ind_master_list': dur_ind_master_list
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'] + '-reinit_pulse_dur')
    
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)
    
    print(' \nRoutine complete!')
    return


#%%


if __name__ == "__main__":

    ############ Replots ############

    # if False:
    if True:
        # tool_belt.init_matplotlib()
        # file_name = "2022_11_04-13_31_23-johnson-search"
        filenames = ['2022_11_21-15_24_59-johnson-search']
        # file_name = "2022_08_09-15_22_25-rubin-nv1"
        powers_all = []
        thresholds_all = []
        fidelities_all = []
            
        # opti_readout_dur = determine_opti_readout_dur(
        #     nv0, nvm, max_readout_dur,exp_dur=.1e6
        # )
        # print(opti_readout_dur)
        # opti_readout_dur = 100e6
        # do_save = True
        
        # readout_dur = opti_readout_dur
            
        times = [100e3]
        # times = [4e6,1e6,400e3,100e3,50e3,10e3]
        # times = [2e6,4e6]
        
        for rd in times:
            
            powers = []
            thresholds = []
            fidelities = []
            
            readout_dur = rd
            
            for file_name in filenames:
                
                data = tool_belt.get_raw_data(file_name)
                nv_sig = data["nv_sig"]
                nv0 = data["nv0"]
                nvm = data["nvm"]
                readout_power = nv_sig["charge_readout_laser_power"]
                max_readout_dur = nv_sig["charge_readout_dur"]
                
                try:
                    threshold, fidelity,n0,nm = plot_threshold(
                        nv_sig,
                        readout_dur,
                        nv0,
                        nvm,
                        readout_power,
                        fit_threshold_full_model=True,
                        nd_filter=None,
                        plot_model_hists=True,
                        bins=None
                    )
                except:
                    threshold=np.nan
                    fidelity=np.nan
            
                thresholds.append(threshold)
                fidelities.append(round(fidelity,3))
                powers.append(readout_power)
                
            powers_all.append(powers)
            thresholds_all.append(thresholds)
            fidelities_all.append(fidelities)
                
        print(powers_all)
        print(thresholds_all)
        print(fidelities_all)
        
        # data_to_save = {"powers": powers_all,
        #                 "thresholds": thresholds_all,
        #                 "fidelities": fidelities_all
        #                 }
        # timestamp = tool_belt.get_time_stamp()
        # file_path = tool_belt.get_file_path(
            # __file__, timestamp, nv_sig["name"]+'analysis_data'
        # )
        # tool_belt.save_raw_data(data_to_save, file_path)
        
        # # file_path = "E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_Carr/branch_opx-setup/determine_charge_readout_params/2022_11/"
        # analysis_data = tool_belt.get_raw_data(file_path)
        # powers = analysis_data["powers"]
        # fidelities = analysis_data["fidelities"]
        
        # for i in range(len(times)):
            
            
        #     plt.figure()
            
        #     plt.scatter(powers[i],fidelities[i])
        #     plt.title('{} ms readout'.format(times[i]))
        #     plt.show()
            
            
        # plot_histogram(nv_sig, nv0, nvm, 700e6, readout_power)
            

        # readout_durs = [10e6, 25e6, 50e6, 100e6, 200e6]
        # for dur in readout_durs:
        #     plot_histogram(nv_sig, nv0, nvm, dur, readout_power)

        # plt.show(block=True)
        # sys.exit()

    ########################