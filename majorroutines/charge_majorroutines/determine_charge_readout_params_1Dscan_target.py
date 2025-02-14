# -*- coding: utf-8 -*-
"""
Modified version of determine_charge_readout_dur. Runs a single experiment
at a given power and use time-resolved counting to determine histograms at
difference readout durations.

Created on Thu Apr 22 14:09:39 2021

@author: Carter Fox
"""

import copy
import random
import sys
import time

import chargeroutines.photonstatistics as model
import labrad
import matplotlib.pyplot as plt
import numpy
import numpy as np

# import majorroutines.optimize_digital as optimize
import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt

# %%


def calc_histogram(nv0, nvm, dur, bins):
    # Counts are in us, readout is in ns
    dur_us = dur / 1e3
    nv0_counts = [np.count_nonzero(np.array(rep) < dur_us) for rep in nv0]  # ???
    nvm_counts = [np.count_nonzero(np.array(rep) < dur_us) for rep in nvm]  # ???
    # print(nv0_counts)
    max_0 = max(nv0_counts)
    max_m = max(nvm_counts)
    if bins == None:
        occur_0, bin_edges_0 = np.histogram(
            nv0_counts,
            np.linspace(0, max_0, max_0 + 1),  # 200)  #
        )
        occur_m, bin_edge_m = np.histogram(
            nvm_counts,
            np.linspace(0, max_m, max_m + 1),  # 200)  #
        )
    elif bins != None:
        occur_0, bin_edges_0 = np.histogram(
            nv0_counts,
            bins,  # np.linspace(0, max_0, max_0 + 1) #200)  #
        )
        occur_m, bin_edge_m = np.histogram(
            nvm_counts,
            bins,  # np.linspace(0, max_m,  max_m + 1) #200)  #
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
    occur_0, x_vals_0, occur_m, x_vals_m, num_reps, report_averages=False
):
    mean_0 = sum(occur_0 * x_vals_0) / num_reps
    std_0 = np.sqrt(sum(occur_0 * (x_vals_0 - mean_0) ** 2) / (num_reps - 1))
    mean_m = sum(occur_m * x_vals_m) / num_reps
    std_m = np.sqrt(sum(occur_m * (x_vals_m - mean_m) ** 2) / (num_reps - 1))
    avg_std = (std_0 + std_m) / 2
    norm_sep = (mean_m - mean_0) / avg_std
    if report_averages:
        print(mean_0)
        print(mean_m)
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
    guess = [10 * 10**-4, 100 * 10**-4, 1000 * 10**-4, 500 * 10**-4]
    fit, dev = model.get_curve_fit(tR, 0, combined_hist, guess)

    if do_plot:
        u_value0, freq0 = model.get_Probability_distribution(NV0_hist.tolist())
        u_valuem, freqm = model.get_Probability_distribution(NVm_hist.tolist())
        u_value2, freq2 = model.get_Probability_distribution(combined_hist)
        curve = model.get_photon_distribution_curve(
            tR, u_value2, fit[0], fit[1], fit[2], fit[3]
        )

        A1, A1pcov = model.get_curve_fit_to_weight(tR, 0, NV0_hist.tolist(), [0.5], fit)
        A2, A2pcov = model.get_curve_fit_to_weight(tR, 0, NVm_hist.tolist(), [0.5], fit)

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
                r"$g_0(s^{-1}) =%.2f$" % (fit[0] * 10**3,),
                r"$g_1(s^{-1})  =%.2f$" % (fit[1] * 10**3,),
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
    readout_time, nv0_array, nvm_array, max_x_val, power, nd_filter=None
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
    tR = readout_time / 10**6
    fit_rate = single_nv_photon_statistics_model(tR, nv0_array, nvm_array)
    max_x_val = int(max_x_val)
    x_data = np.linspace(0, 100, 101)
    thresh_para = model.calculate_threshold(tR, x_data, fit_rate)

    plot_x_data = np.linspace(0, max_x_val, max_x_val + 1)
    fig3, ax = plt.subplots()
    ax.plot(plot_x_data, model.get_PhotonNV0_list(plot_x_data, tR, fit_rate, 0.5), "-o")
    ax.plot(plot_x_data, model.get_PhotonNVm_list(plot_x_data, tR, fit_rate, 0.5), "-o")
    plt.axvline(x=thresh_para[0], color="red")
    mu_0 = fit_rate[3] * tR
    mu_m = fit_rate[2] * tR
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
        title_text = "{} ms readout, {} V, {}".format(int(tR), power, nd_filter)
    else:
        title_text = "{} ms readout, {} V".format(int(tR), power)
    ax.set_title(title_text)
    plt.xlabel("Number of counts")
    plt.ylabel("Probability Density")

    fidelity = thresh_para[1]
    threshold = thresh_para[0]
    print(title_text)
    print("Threshold: {} counts, fidelity: {:.3f}".format(threshold, fidelity))
    return threshold, fidelity, mu_0, mu_m, fig3


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
    ax.set_xlim(0)
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
        title_text = "{} ms readout, {} V, {}".format(
            int(readout_time / 1e6), power, nd_filter
        )
    else:
        title_text = "{} ms readout, {} V".format(int(readout_time / 1e6), power)
    ax.set_title(title_text)
    print("mean 0", round(mu_0, 2))
    print("mean m", round(mu_m, 2))
    return thresh, fid, fig3, mu_0, mu_m


def plot_threshold(
    nv_sig,
    readout_dur,
    nv0_counts,
    nvm_counts,
    power,
    bins,
    fit_threshold_full_model=False,
    nd_filter=None,
    do_save=False,
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
        nv0_counts, nvm_counts, readout_dur, bins
    )
    max_x_val = max(list(x_vals_0) + list(x_vals_m)) + 10

    num_reps = len(nv0_counts)
    mean_0 = sum(occur_0 * x_vals_0) / num_reps
    mean_m = sum(occur_m * x_vals_m) / num_reps

    if fit_threshold_full_model:
        dur_us = readout_dur / 1e3
        nv0_counts_list = [
            np.count_nonzero(np.array(rep) < dur_us) for rep in nv0_counts
        ]
        nvm_counts_list = [
            np.count_nonzero(np.array(rep) < dur_us) for rep in nvm_counts
        ]
        threshold, fidelity, mu_0, mu_m, fig = calculate_threshold_with_model(
            readout_dur, nv0_counts_list, nvm_counts_list, max_x_val, power, nd_filter
        )
    else:
        threshold, fidelity, fig, mu_0, mu_m = calculate_threshold_no_model(
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

    if do_save:
        file_path = tool_belt.get_file_path(
            __file__, timestamp, nv_sig["name"] + "-threshold"
        )
        tool_belt.save_figure(fig, file_path)
    return mu_0, mu_m


def determine_opti_readout_dur(nv0, nvm, max_readout_dur, bins):
    if max_readout_dur <= 100e6:
        readout_dur_linspace = np.arange(1e6, max_readout_dur, 1e6)
    else:
        readout_dur_linspace = np.arange(10e6, max_readout_dur, 10e6)

    # Round to nearest ms
    readout_dur_linspace = [int(1e6 * round(val / 1e6)) for val in readout_dur_linspace]

    sensitivities = []
    num_reps = len(nv0)

    for dur in readout_dur_linspace:
        occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(nv0, nvm, dur, bins)
        separation = calc_separation(occur_0, x_vals_0, occur_m, x_vals_m, num_reps)
        sensitivities.append(separation * np.sqrt(dur * 10**9))

    max_sensitivity = max(sensitivities)
    opti_readout_dur_ind = sensitivities.index(max_sensitivity)
    opti_readout_dur = readout_dur_linspace[opti_readout_dur_ind]

    return opti_readout_dur


def plot_histogram(
    nv_sig,
    nv0,
    nvm,
    dur,
    power,
    # total_seq_time_sec,
    bins,
    nd_filter=None,
    do_save=True,
    report_averages=False,
):
    num_reps = len(nv0)
    occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(nv0, nvm, dur, bins)
    # overlap = calc_overlap(occur_0, x_vals_0, occur_m, x_vals_m, num_reps)
    # print("fractional overlap: {}".format(overlap))
    separation = calc_separation(
        occur_0, x_vals_0, occur_m, x_vals_m, num_reps, report_averages
    )
    sensitivity = separation * np.sqrt(dur * 10**9)
    print(f"Normalized separation / sqrt(Hz): {sensitivity}")

    fig_hist, ax = plt.subplots(1, 1)
    ax.plot(x_vals_0, occur_0, "r-o", label="Initial red pulse")
    ax.plot(x_vals_m, occur_m, "g-o", label="Initial green pulse")
    ax.set_xlim(0)
    ax.set_xlabel("Counts")
    ax.set_ylabel("Occur.")
    # ax.set_title("{} ms readout, {} V".format(int(dur / 1e6), power))
    if nd_filter:
        title_text = "{} ms readout, {} V, {}".format(int(dur / 1e6), power, nd_filter)
    else:
        title_text = "{} ms readout, {} V".format(int(dur / 1e6), power)
    ax.set_title(title_text)
    ax.legend()

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
        rep_processed_timetags = [val - open_timetag for val in rep_processed_timetags]
        processed_timetags.append(rep_processed_timetags)

    return processed_timetags


def measure_histograms_sub(
    cxn,
    nv_sig,
    opti_nv_sig,
    x_readout_step,
    y_readout_step,
    seq_file,
    seq_args,
    apd_indices,
    num_reps,
):
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(seq_file, seq_args_string)
    period = ret_vals[0]
    period_sec = period / 10**9

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
        opti_coords = targeting.main_with_cxn(cxn, opti_nv_sig, apd_indices)
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
        # xy_server.load_arb_scan_xy(x_points, y_points, int(10e7)) #CF

        cxn.apd_tagger.start_tag_stream(apd_indices)
        cxn.apd_tagger.clear_buffer()

        # Run the sequence
        if num_reps_remaining > num_reps_per_cycle:
            num_reps_to_run = num_reps_per_cycle
        else:
            num_reps_to_run = num_reps_remaining

        ##############this is what I mainly added
        xy_server = tool_belt.get_xy_server(cxn)  # CF

        init_pulse_x = adjusted_nv_coords[0]
        init_pulse_y = adjusted_nv_coords[1]
        readout_pulse_x = init_pulse_x + x_readout_step
        readout_pulse_y = init_pulse_y + y_readout_step
        x_points = [init_pulse_x] + [readout_pulse_x, init_pulse_x] * num_reps_to_run
        y_points = [init_pulse_y] + [readout_pulse_y, init_pulse_y] * num_reps_to_run

        xy_server.load_arb_scan_xy(x_points, y_points, int(period))  # CF
        ################

        cxn.pulse_streamer.stream_immediate(seq_file, num_reps_to_run, seq_args_string)

        ret_vals = cxn.apd_tagger.read_tag_stream(num_reps_to_run * 2)
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

        cxn.apd_tagger.stop_tag_stream()

        num_reps_remaining -= num_reps_per_cycle

    return timetags, channels, period_sec


# Apply a gren or red pulse, then measure the counts under yellow illumination.
# Repeat num_reps number of times and returns the list of counts after red illumination, then green illumination
# Use with DM on red and green
def measure_histograms(
    nv_sig, opti_nv_sig, x_readout_step, y_readout_step, apd_indices, num_reps
):
    with labrad.connect() as cxn:
        nv0, nvm, total_seq_time_sec = measure_histograms_with_cxn(
            cxn,
            nv_sig,
            opti_nv_sig,
            x_readout_step,
            y_readout_step,
            apd_indices,
            num_reps,
        )

    return nv0, nvm, total_seq_time_sec


def measure_histograms_with_cxn(
    cxn, nv_sig, opti_nv_sig, x_readout_step, y_readout_step, apd_indices, num_reps
):
    # Only support a single APD for now
    apd_index = apd_indices[0]

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv-_prep_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv0_prep_laser")

    readout_laser_power = tool_belt.set_laser_power(cxn, nv_sig, "charge_readout_laser")

    readout_pulse_time = nv_sig["charge_readout_dur"]

    # Pulse sequence to do a single pulse followed by readout
    readout_on_2nd_pulse = 2
    seq_file = "simple_readout_two_pulse_moving_target.py"  # CF
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

    timetags, channels, period_sec = measure_histograms_sub(
        cxn,
        nv_sig,
        opti_nv_sig,
        x_readout_step,
        y_readout_step,
        seq_file,
        seq_args,
        apd_indices,
        num_reps,
    )
    nvm = process_timetags(apd_gate_channel, timetags, channels)

    # Red measurement
    seq_args = gen_seq_args("nv0_prep_laser")
    timetags, channels, period_sec = measure_histograms_sub(
        cxn,
        nv_sig,
        opti_nv_sig,
        x_readout_step,
        y_readout_step,
        seq_file,
        seq_args,
        apd_indices,
        num_reps,
    )
    nv0 = process_timetags(apd_gate_channel, timetags, channels)

    tool_belt.reset_cfm(cxn)

    return nv0, nvm, period_sec * 2


def main(
    nv_sig,
    opti_nv_sig,
    x_steps,
    y_steps,
    apd_indices,
    num_reps=500,
    max_readout_dur=1e9,
    bins=None,
    readout_powers=None,
    plot_readout_durs=None,
    fit_threshold_full_model=False,
    NIR_test=False,
):
    xs, ys = [], []
    mu_0_NIR_list, mu_0_list, mu_m_NIR_list, mu_m_list = [], [], [], []
    for xstep in x_steps:
        for ystep in y_steps:
            try:
                mu_0, mu_m = determine_readout_dur_power(
                    nv_sig,
                    opti_nv_sig,
                    xstep,
                    ystep,
                    apd_indices,
                    num_reps,
                    max_readout_dur,
                    bins,
                    readout_powers,
                    plot_readout_durs,
                    fit_threshold_full_model,
                )

                mu_0_list.append(mu_0)
                mu_m_list.append(mu_m)
            except:
                mu_0_list.append(
                    numpy.nan
                )  # if it fails, it just appends None so that when plotted it is am empty space
                mu_m_list.append(numpy.nan)

    if NIR_test:
        with labrad.connect() as cxn:
            power_supply = cxn.power_supply_mp710087
            power_supply.output_on()
            power_supply.set_voltage(1.3)
        time.sleep(1)
        for xstep in x_steps:
            for ystep in y_steps:
                try:
                    mu_0_NIR, mu_m_NIR = determine_readout_dur_power(
                        nv_sig,
                        opti_nv_sig,
                        xstep,
                        ystep,
                        apd_indices,
                        num_reps,
                        max_readout_dur,
                        bins,
                        readout_powers,
                        plot_readout_durs,
                        fit_threshold_full_model,
                        NIR_run="On",
                    )
                    mu_0_NIR_list.append(mu_0_NIR)
                    mu_m_NIR_list.append(mu_m_NIR)
                except:
                    mu_0_NIR_list.append(numpy.nan)
                    mu_m_NIR_list.append(numpy.nan)

        with labrad.connect() as cxn:
            power_supply = cxn.power_supply_mp710087
            power_supply.output_off()
        time.sleep(1)

    plot_and_save_means(
        nv_sig,
        num_reps,
        readout_powers,
        max_readout_dur,
        x_steps,
        y_steps,
        mu_0_list,
        mu_m_list,
        mu_0_NIR_list,
        mu_m_NIR_list,
    )


def plot_and_save_means(
    nv_sig,
    num_reps,
    readout_pow,
    max_readout_dur,
    x_steps,
    y_steps,
    mu_0_list,
    mu_m_list,
    mu_0_NIR_list=None,
    mu_m_NIR_list=None,
):
    figm0, ax0 = plt.subplots(1, 1)
    figm1, ax1 = plt.subplots(1, 1)
    figm2, ax2 = plt.subplots(1, 1)
    figm3, ax3 = plt.subplots(1, 1)

    if len(x_steps) > 1:
        domain = x_steps
        ax0.set_xlabel("x displacement (V)")
        ax1.set_xlabel("x displacement (V)")
        ax2.set_xlabel("x displacement (V)")
        ax3.set_xlabel("x displacement (V)")
    elif len(y_steps) > 1:
        domain = y_steps
        ax0.set_xlabel("y displacement (V)")
        ax1.set_xlabel("y displacement (V)")
        ax2.set_xlabel("y displacement (V)")
        ax3.set_xlabel("y displacement (V)")

    ax0.set_ylabel(r"$\mu_m$")
    ax1.set_ylabel(r"$\mu_0$")
    ax0.plot(domain, mu_m_list, label="no NIR")
    ax1.plot(domain, mu_0_list, label="no NIR")
    if mu_0_NIR_list != None:
        ax0.plot(domain, mu_m_NIR_list, label="NIR")
        ax1.plot(domain, mu_0_NIR_list, label="NIR")

        normed_mu_m = (np.asarray(mu_m_NIR_list) - np.asarray(mu_m_list)) / np.asarray(
            mu_m_list
        )
        normed_mu_0 = (np.asarray(mu_0_NIR_list) - np.asarray(mu_0_list)) / np.asarray(
            mu_0_list
        )
        ax2.plot(domain, normed_mu_m)
        ax3.plot(domain, normed_mu_0)
        ax2.set_ylabel(r"($\mu_{mNIR}$ - $\mu_m$)/$\mu_m$")
        ax3.set_ylabel(r"($\mu_{0NIR}$ - $\mu_0$)/$\mu_0$")
    ax0.legend()
    ax1.legend()

    plt.show()

    timestamp = tool_belt.get_time_stamp()

    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "num_reps": num_reps,
        "readout_power": readout_pow,
        "readout_duration": max_readout_dur,
        "x_steps": x_steps.tolist(),
        "y_steps": y_steps.tolist(),
        "mu_0_no_NIR": mu_0_list,
        "mu_m_no_NIR": mu_m_list,
        "mu_0_NIR": mu_0_NIR_list,
        "mu_m_NIR": mu_m_NIR_list,
    }

    name = nv_sig["name"]
    filePath = tool_belt.get_file_path(__file__, timestamp, name)
    filePath0 = tool_belt.get_file_path(__file__, timestamp, name + "mu_m")
    filePath1 = tool_belt.get_file_path(__file__, timestamp, name + "mu_0")
    filePath2 = tool_belt.get_file_path(__file__, timestamp, name + "mu_m_norm")
    filePath3 = tool_belt.get_file_path(__file__, timestamp, name + "mu_0_norm")
    tool_belt.save_figure(figm0, filePath0)
    tool_belt.save_figure(figm1, filePath1)
    tool_belt.save_figure(figm2, filePath2)
    tool_belt.save_figure(figm3, filePath3)
    tool_belt.save_raw_data(rawData, filePath)


def determine_readout_dur_power(
    nv_sig,
    opti_nv_sig,
    x_readout_step,
    y_readout_step,
    apd_indices,
    num_reps=500,
    max_readout_dur=1e9,
    bins=None,
    readout_powers=None,
    plot_readout_durs=None,
    fit_threshold_full_model=False,
    NIR_run="off",
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
            nv_sig_copy,
            opti_nv_sig,
            x_readout_step,
            y_readout_step,
            apd_indices,
            num_reps,
        )
        nv0_power.append(nv0)
        nvm_power.append(nvm)

        timestamp = tool_belt.get_time_stamp()
        file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
        raw_data = {
            "timestamp": timestamp,
            "nv_sig": nv_sig_copy,
            "nv_sig-units": tool_belt.get_nv_sig_units(),
            "x_step": x_readout_step,
            "y_step": y_readout_step,
            "NIR": NIR_run,
            "num_reps": num_reps,
            "nv0": nv0,
            "nv0-units": "list(list(us))",
            "nvm": nvm,
            "nvm-units": "list(list(us))",
        }

        tool_belt.save_raw_data(raw_data, file_path)

        if plot_readout_durs is not None:
            for dur in plot_readout_durs:
                filter_key = "charge_readout_laser_filter"
                if filter_key in nv_sig:
                    nd_filter = nv_sig[filter_key]
                else:
                    nd_filter = None

                plot_histogram(nv_sig, nv0, nvm, dur, p, bins, nd_filter=nd_filter)

                if fit_threshold_full_model:
                    print(
                        "Calculating threshold values...\nMay take up to a few minutes..."
                    )
                mu_0, mu_m = plot_threshold(
                    nv_sig,
                    dur,
                    nv0,
                    nvm,
                    p,
                    bins,
                    fit_threshold_full_model,
                    nd_filter=nd_filter,
                    do_save=True,
                )

        print("data collected!")

    return mu_0, mu_m


# %%


if __name__ == "__main__":
    ############ Replots ############

    file_name = "2022_08_18-03_44_11-hopper-search"
    data = tool_belt.get_raw_data(file_name)
    nv_sig = data["nv_sig"]

    x_steps = np.array(data["x_steps"])
    y_steps = np.array(data["y_steps"])
    mu_0_list = np.array(data["mu_0_no_NIR"])
    mu_m_list = np.array(data["mu_m_no_NIR"])
    mu_0_NIR_list = np.array(data["mu_0_NIR"])
    mu_m_NIR_list = np.array(data["mu_m_NIR"])

    figm0, ax0 = plt.subplots(1, 1)
    figm1, ax1 = plt.subplots(1, 1)
    figm2, ax2 = plt.subplots(1, 1)
    figm3, ax3 = plt.subplots(1, 1)

    if len(x_steps) > 1:
        domain = x_steps
        ax0.set_xlabel("x displacement (V)")
        ax1.set_xlabel("x displacement (V)")
        ax2.set_xlabel("x displacement (V)")
        ax3.set_xlabel("x displacement (V)")
    elif len(y_steps) > 1:
        domain = y_steps
        ax0.set_xlabel("y displacement (V)")
        ax1.set_xlabel("y displacement (V)")
        ax2.set_xlabel("y displacement (V)")
        ax3.set_xlabel("y displacement (V)")

    ax0.set_ylabel(r"$\mu_m$")
    ax1.set_ylabel(r"$\mu_0$")
    ax0.plot(
        domain[np.where(np.isnan(mu_m_list) == False)],
        mu_m_list[np.where(np.isnan(mu_m_list) == False)],
        "o-",
        markersize=3,
        label="no NIR",
    )
    ax1.plot(
        domain[np.where(np.isnan(mu_0_list) == False)],
        mu_0_list[np.where(np.isnan(mu_0_list) == False)],
        "o-",
        markersize=3,
        label="no NIR",
    )
    if len(mu_0_NIR_list) != 0:
        ax0.plot(
            domain[np.where(np.isnan(mu_m_NIR_list) == False)],
            mu_m_NIR_list[np.where(np.isnan(mu_m_NIR_list) == False)],
            "o-",
            markersize=3,
            label="NIR",
        )
        ax1.plot(
            domain[np.where(np.isnan(mu_0_NIR_list) == False)],
            mu_0_NIR_list[np.where(np.isnan(mu_0_NIR_list) == False)],
            "o-",
            markersize=3,
            label="NIR",
        )

        normed_mu_m = (mu_m_NIR_list - mu_m_list) / mu_m_list
        normed_mu_0 = (mu_0_NIR_list - mu_0_list) / mu_0_list
        ax2.plot(
            domain[np.where(np.isnan(normed_mu_m) == False)],
            normed_mu_m[np.where(np.isnan(normed_mu_m) == False)],
            "o-",
            markersize=3,
        )
        ax3.plot(
            domain[np.where(np.isnan(normed_mu_0) == False)],
            normed_mu_0[np.where(np.isnan(normed_mu_0) == False)],
            "o-",
            markersize=3,
        )
        ax2.set_ylabel(r"($\mu_{mNIR}$ - $\mu_m$)/$\mu_m$")
        ax3.set_ylabel(r"($\mu_{0NIR}$ - $\mu_0$)/$\mu_0$")
    ax0.legend()
    ax1.legend()
    plt.show()
