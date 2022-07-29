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
import matplotlib.pyplot as plt
import labrad
import time
import sys

import utils.tool_belt as tool_belt

# import majorroutines.optimize_digital as optimize
import majorroutines.optimize as optimize


# %%


def calc_histogram(nv0, nvm, dur):

    # Counts are in us, readout is in ns
    dur_us = dur / 1e3
    nv0_counts = [np.count_nonzero(np.array(rep) < dur_us) for rep in nv0]
    nvm_counts = [np.count_nonzero(np.array(rep) < dur_us) for rep in nvm]

    max_0 = max(nv0_counts)
    max_m = max(nvm_counts)
    occur_0, bin_edges_0 = np.histogram(
        nv0_counts, np.linspace(0, max_0, 200)#max_0 + 1)
    )
    occur_m, bin_edge_m = np.histogram(
        nvm_counts, np.linspace(0, max_m, 200)#max_m + 1)
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


def determine_opti_readout_dur(nv0, nvm, max_readout_dur):

    if max_readout_dur <= 100e6:
        readout_dur_linspace = np.arange(1e6, max_readout_dur, 1e6)
    else:
        readout_dur_linspace = np.arange(10e6, max_readout_dur, 10e6)

    # Round to nearest ms
    readout_dur_linspace = [
        int(1e6 * round(val / 1e6)) for val in readout_dur_linspace
    ]

    separations = []
    num_reps = len(nv0)

    for dur in readout_dur_linspace:
        occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(nv0, nvm, dur)
        separation = calc_separation(
            occur_0, x_vals_0, occur_m, x_vals_m, num_reps
        )
        separations.append(separation)

    max_separation = max(separations)
    opti_readout_dur_ind = separations.index(max_separation)
    opti_readout_dur = readout_dur_linspace[opti_readout_dur_ind]

    return opti_readout_dur


def plot_histogram(
    nv_sig, nv0, nvm, dur, power, total_seq_time_sec, do_save=True, report_averages=False
):

    num_reps = len(nv0)
    occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(nv0, nvm, dur)
    # overlap = calc_overlap(occur_0, x_vals_0, occur_m, x_vals_m, num_reps)
    # print("fractional overlap: {}".format(overlap))
    separation = calc_separation(
        occur_0, x_vals_0, occur_m, x_vals_m, num_reps, report_averages
    )
    print("Normalized separation: {}".format(separation))
    
    fig_of_merit = separation/np.sqrt(total_seq_time_sec) 

    fig_hist, ax = plt.subplots(1, 1)
    ax.plot(x_vals_0, occur_0, "r-o", label="Initial red pulse")
    ax.plot(x_vals_m, occur_m, "g-o", label="Initial green pulse")
    ax.set_xlabel("Counts")
    ax.set_ylabel("Occur.")
    # ax.set_title("{} ms readout, {} V".format(int(dur / 1e6), power))
    ax.set_title("{} ms readout, {} V, {} sep/sqrt(time)".format(int(dur / 1e6), power,round(fig_of_merit,2)))
    ax.legend()

    if do_save:
        timestamp = tool_belt.get_time_stamp()
        file_path = tool_belt.get_file_path(
            __file__, timestamp, nv_sig["name"]+"_histogram"
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

    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(seq_file, seq_args_string)
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
        cxn.apd_tagger.start_tag_stream(apd_indices)
        cxn.apd_tagger.clear_buffer()

        # Run the sequence
        if num_reps_remaining > num_reps_per_cycle:
            num_reps_to_run = num_reps_per_cycle
        else:
            num_reps_to_run = num_reps_remaining
        cxn.pulse_streamer.stream_immediate(
            seq_file, num_reps_to_run, seq_args_string
        )
        # print(num_reps_to_run)

        ret_vals = cxn.apd_tagger.read_tag_stream(num_reps_to_run)
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
def measure_histograms(nv_sig, opti_nv_sig, apd_indices, num_reps):

    with labrad.connect() as cxn:
        nv0, nvm, total_seq_time_sec = measure_histograms_with_cxn(
            cxn, nv_sig, opti_nv_sig, apd_indices, num_reps
        )

    return nv0, nvm, total_seq_time_sec


def measure_histograms_with_cxn(
    cxn, nv_sig, opti_nv_sig, apd_indices, num_reps
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
    timetags, channels, period_sec = measure_histograms_sub(
        cxn, nv_sig, opti_nv_sig, seq_file, seq_args, apd_indices, num_reps
    )
    nvm = process_timetags(apd_gate_channel, timetags, channels)

    # Red measurement
    seq_args = gen_seq_args("nv0_prep_laser")
    timetags, channels, period_sec = measure_histograms_sub(
        cxn, nv_sig, opti_nv_sig, seq_file, seq_args, apd_indices, num_reps
    )
    nv0 = process_timetags(apd_gate_channel, timetags, channels)

    tool_belt.reset_cfm(cxn)

    return nv0, nvm, period_sec*2


def determine_readout_dur_power(
    nv_sig,
    opti_nv_sig,
    apd_indices,
    num_reps=500,
    max_readout_dur=1e9,
    readout_powers=None,
    plot_readout_durs=None,
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
            nv_sig_copy, opti_nv_sig, apd_indices, num_reps
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
                plot_histogram(nv_sig, nv0, nvm, dur, p,total_seq_time_sec)

        print("data collected!")

    return


#%%


if __name__ == "__main__":

    ############ Replots ############

    if False:
    # if True:
        tool_belt.init_matplotlib()
        # file_name = "2022_02_14-03_32_40-wu-nv1_2022_02_10"
        file_name = "2022_04_14-16_31_30-wu-nv3_2022_04_14"
        data = tool_belt.get_raw_data(file_name)
        nv_sig = data["nv_sig"]
        nv0 = data["nv0"]
        nvm = data["nvm"]
        readout_power = nv_sig["charge_readout_laser_power"]
        max_readout_dur = nv_sig["charge_readout_dur"]

        opti_readout_dur = determine_opti_readout_dur(
            nv0, nvm, max_readout_dur
        )
        opti_readout_dur = 80e6
        # do_save = True
        do_save = False
        plot_histogram(
            nv_sig,
            nv0,
            nvm,
            opti_readout_dur,
            readout_power,
            do_save=do_save,
            report_averages=True,
        )

        # plot_histogram(nv_sig, nv0, nvm, 700e6, readout_power)

        # readout_durs = [10e6, 25e6, 50e6, 100e6, 200e6]
        # for dur in readout_durs:
        #     plot_histogram(nv_sig, nv0, nvm, dur, readout_power)

        # plt.show(block=True)
        sys.exit()

    ########################

    # Rabi
    apd_indices = [1]
    sample_name = "johnson"

    green_laser = "integrated_520"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    nv_sig = {
        "coords": [-0.748, -0.180, 6.17],
        "name": "{}-nv1".format(sample_name),
        "disable_opt": False,
        "disable_z_opt": False,
        "expected_count_rate": 32,
        # 'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e7,
        # 'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0", 'imaging_readout_dur': 1e8,
        "imaging_laser": green_laser,
        "imaging_laser_filter": "nd_0.5",
        "imaging_readout_dur": 1e7,
        # 'imaging_laser': green_laser, 'imaging_laser_filter': "nd_0.5", 'imaging_readout_dur': 1e8,
        # 'imaging_laser': yellow_laser, 'imaging_laser_power': 1.0, 'imaging_readout_dur': 1e8,
        # 'imaging_laser': red_laser, 'imaging_readout_dur': 1e7,
        # 'spin_laser': green_laser, 'spin_laser_filter': 'nd_0.5', 'spin_pol_dur': 1E5, 'spin_readout_dur': 350,
        "spin_laser": green_laser,
        "spin_laser_filter": "nd_0.5",
        "spin_pol_dur": 1e4,
        "spin_readout_dur": 350,
        # 'spin_laser': green_laser, 'spin_laser_filter': 'nd_0', 'spin_pol_dur': 1E4, 'spin_readout_dur': 300,
        "nv-_reionization_laser": green_laser,
        "nv-_reionization_dur": 1e6,
        "nv-_reionization_laser_filter": "nd_1.0",
        # 'nv-_reionization_laser': green_laser, 'nv-_reionization_dur': 1E5, 'nv-_reionization_laser_filter': 'nd_0.5',
        "nv-_prep_laser": green_laser,
        "nv-_prep_laser_dur": 1e6,
        "nv-_prep_laser_filter": None,# "nd_1.0",
        "nv0_ionization_laser": red_laser,
        "nv0_ionization_dur": 100,
        "nv0_prep_laser": red_laser,
        "nv0_prep_laser-power": 0.69,
        "nv0_prep_laser_dur": 1E6,
        "spin_shelf_laser": yellow_laser,
        "spin_shelf_dur": 0,
        "spin_shelf_laser_power": 1.0,
        # 'spin_shelf_laser': green_laser, 'spin_shelf_dur': 50,
        "initialize_laser": green_laser,
        "initialize_dur": 1e4,
        # "charge_readout_laser": yellow_laser, "charge_readout_dur": 1000e6, "charge_readout_laser_power": 1.0,
        "charge_readout_laser": yellow_laser,
        "charge_readout_dur": 1840e6,
        "charge_readout_laser_power": 1.0,
        "collection_filter": "715_sp+630_lp",
        "magnet_angle": None,
        "resonance_LOW": 2.8073,
        "rabi_LOW": 173.2,
        "uwave_power_LOW": 16.5,
        # 'resonance_LOW': 2.8451, 'rabi_LOW': 176.4, 'uwave_power_LOW': 16.5,
        "resonance_HIGH": 2.9489,
        "rabi_HIGH": 234.6,
        "uwave_power_HIGH": 16.5,
    }

    # readout_durs = [10*10**3, 50*10**3, 100*10**3, 500*10**3,
    #                 1*10**6, 2*10**6, 3*10**6, 4*10**6, 5*10**6,
    #                 6*10**6, 7*10**6, 8*10**6, 9*10**6, 1*10**7,
    #                 2*10**7, 3*10**7, 4*10**7, 5*10**7]
    # readout_durs = numpy.linspace(10e6, 50e6, 5)
    # readout_durs = [10e6, 25e6, 50e6, 100e6, 200e6, 400e6, 700e6, 1e9, 2e9]
    # readout_durs = [10e6, 25e6, 50e6, 100e6, 200e6, 400e6, 1e9]
    readout_durs = [5e6, 10e6, 20e6]
    # readout_durs = numpy.linspace(700e6, 1e9, 7)
    # readout_durs = [50e6, 100e6, 200e6, 400e6, 1e9]
    # readout_durs = [2e9]
    readout_durs = [int(el) for el in readout_durs]
    max_readout_dur = max(readout_durs)

    # readout_powers = np.linspace(0.6, 1.0, 9)
    # readout_powers = np.arange(0.75, 1.05, 0.05)
    # readout_powers = np.arange(0.68, 1.04, 0.04)
    # readout_powers = np.linspace(0.9, 1.0, 3)
    readout_powers = [0.2, 0.3, 0.4, 0.5]

    # num_reps = 2000
    # num_reps = 1000
    num_reps = 500

    try:
        determine_readout_dur_power(
            nv_sig,
            nv_sig,
            apd_indices,
            num_reps,
            max_readout_dur=max_readout_dur,
            readout_powers=readout_powers,
            plot_readout_durs=readout_durs,
        )
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
