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

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize


# %%


def plot_histogram(nv_sig, nv0, nvm, readout):

    # Counts are in us, readout is in ns
    readout_us = readout / 1e3
    nv0_counts = np.count_nonzero(nv0 < readout)
    nvm_counts = np.count_nonzero(nvm < readout)

    fig_hist, ax = plt.subplots(1, 1)
    max_0 = max(nv0)
    max_m = max(nvm)
    occur_0, x_vals_0 = np.histogram(nv0, np.linspace(0, max_0, max_0 + 1))
    occur_m, x_vals_m = np.histogram(nvm, np.linspace(0, max_m, max_m + 1))
    ax.plot(x_vals_0[:-1], occur_0, "r-o", label="Initial red pulse")
    ax.plot(x_vals_m[:-1], occur_m, "g-o", label="Initial green pulse")
    ax.set_xlabel("Counts")
    ax.set_ylabel("Occur.")
    ax.set_title("{} ms readout, {} V".format(t / 10 ** 6, p))
    ax.legend()

    timestamp = tool_belt.get_time_stamp()
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_figure(fig_hist, file_path + "_histogram")


def process_timetags(apd_gate_channel, timetags, channels):

    processed_timetags = []

    gate_open_channel = apd_gate_channel
    gate_close_channel = -gate_open_channel

    gate_open_inds = np.where(channels == gate_open_channel)[0]
    gate_close_inds = np.where(channels == gate_close_channel)[0]

    for ind in range(len(gate_open_inds)):
        open_ind = gate_open_inds[ind]
        close_ind = gate_close_inds[ind]
        open_timetag = timetags[open_ind]
        rep_processed_timetags = timetags[open_ind + 1 : close_ind].tolist()
        rep_processed_timetags = [
            val - open_timetag for val in rep_processed_timetags
        ]
        processed_timetags.append(rep_processed_timetags)

    return processed_timetags


def measure_histograms_sub(
    cxn, nv_sig, opti_nv_sig, seq_file, seq_args, num_reps
):

    seq_args_string = tool_belt.encode_seq_args(seq_args)
    period = cxn.pulse_streamer.stream_load(seq_file, seq_args_string)
    period_sec = period * 10 ** 9

    # Some initial parameters
    opti_period = 2.5 * 60
    num_reps_per_cycle = opti_period // period_sec
    num_reps_remaining = num_reps
    timetags = []
    channels = []

    while num_reps_remaining > 0:

        coords = nv_sig["coords"]
        opti_coords_list = []
        opti_coords = optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)
        drift = tool_belt.get_drift()
        adjusted_nv_coords = coords + np.array(drift)
        tool_belt.set_xyz(cxn, adjusted_nv_coords)

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)

        # Run the sequence
        if num_reps_remaining > num_reps_per_cycle:
            num_reps_to_run = num_reps_per_cycle
        else:
            num_reps_to_run = num_reps_remaining
        cxn.pulse_streamer.stream_immediate(
            seq_file, num_reps_to_run, seq_args_string
        )

        ret_vals = cxn.apd_tagger.read_tag_stream()
        buffer_timetags, buffer_channels = ret_vals
        # We don't care about picosecond resolution here, so just round to us
        # We also don't care about the offset value, so subtract that off
        if len(timetags) == 0:
            offset = np.int64(buffer_timetags[0])
        buffer_timetags = [
            int((val - offset) / 1e6) for val in buffer_timetags
        ]
        timetags.append(buffer_timetags)
        channels.append(buffer_channels)

        num_reps_remaining -= num_reps_per_cycle

    timetags = np.array(timetags)
    channels = np.array(channels)

    return timetags, channels


# Apply a gren or red pulse, then measure the counts under yellow illumination.
# Repeat num_reps number of times and returns the list of counts after red illumination, then green illumination
# Use with DM on red and green
def measure_histograms(nv_sig, opti_nv_sig, apd_indices, num_reps):

    with labrad.connect() as cxn:
        nv0, nvm = measure_histograms_with_cxn(
            cxn, nv_sig, opti_nv_sig, apd_indices, num_reps
        )

    return nv0, nvm


def measure_histograms_with_cxn(
    cxn, nv_sig, opti_nv_sig, apd_indices, num_reps
):
    # Only support a single APD for now
    apd_index = apd_indices[0]

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv-_prep_laser")

    readout_laser_power = tool_belt.set_laser_power(
        cxn, nv_sig, "charge_readout_laser"
    )
    nvm_laser_power = tool_belt.set_laser_power(cxn, nv_sig, "nv-_prep_laser")
    nv0_laser_power = tool_belt.set_laser_power(cxn, nv_sig, "nv0_prep_laser")

    readout_pulse_time = nv_sig["charge_readout_dur"]

    reionization_time = nv_sig["nv-_prep_laser_dur"]
    ionization_time = nv_sig["nv0_prep_laser_dur"]

    # Pulse sequence to do a single pulse followed by readout
    seq_file = "simple_readout_two_pulse.py"
    gen_seq_args = lambda init_laser: [
        reionization_time,
        readout_pulse_time,
        nv_sig[init_laser],
        nv_sig["charge_readout_laser"],
        nv0_laser_power,
        readout_laser_power,
        apd_index,
    ]

    apd_gate_channel = tool_belt.get_apd_gate_channel(cxn, apd_index)

    # Green measurement
    seq_args = gen_seq_args("nv-_prep_laser")
    timetags, channels = measure_histograms_sub(
        cxn, nv_sig, opti_nv_sig, seq_file, seq_args, num_reps
    )
    nvm = process_timetags(apd_gate_channel, timetags, channels)

    # Red measurement
    seq_args = gen_seq_args("nv0_prep_laser")
    timetags, channels = measure_histograms_sub(
        cxn, nv_sig, opti_nv_sig, seq_file, seq_args, num_reps
    )
    nv0 = process_timetags(apd_gate_channel, timetags, channels)

    tool_belt.reset_cfm(cxn)

    return nv0, nvm


def determine_readout_dur_power(
    nv_sig,
    opti_nv_sig,
    apd_indices,
    max_readout_dur=1e9,
    readout_powers=None,
):
    num_reps = 500

    if readout_powers is None:
        readout_yellow_powers = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]

    tool_belt.init_safe_stop()

    for p in readout_yellow_powers:

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        nv0_power = []
        nvm_power = []

        nv_sig_copy = copy.deepcopy(nv_sig)
        nv_sig_copy["charge_readout_dur"] = max_readout_dur
        nv_sig_copy["charge_readout_laser_power"] = p

        nv0, nvm = measure_histograms(
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
            "num_runs": num_reps,
            "nv0": nv0.tolist(),
            "nv0-units": "list(us)",
            "nvm": nvm.tolist(),
            "nvm-units": "list(us)",
        }

        tool_belt.save_raw_data(raw_data, file_path)

        print("data collected!")

    return


#%%


if __name__ == "__main__":

    # apd_indices = [0]
    apd_indices = [1]
    # apd_indices = [0,1]

    # nd = 'nd_0'
    nd = "nd_0.5"
    # nd = 'nd_1.0'
    # nd = 'nd_2.0'

    sample_name = "wu"

    green_laser = "laserglow_532"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    nv_sig = {
        "coords": [0.126, 0.297, -1],
        "name": "{}-nv1_2021_11_26".format(sample_name),
        "disable_opt": False,
        "expected_count_rate": 23,
        "imaging_laser": green_laser,
        "imaging_laser_filter": nd,
        "imaging_readout_dur": 1e7,
        # 'imaging_laser': yellow_laser, 'imaging_laser_power': 1.0, 'imaging_readout_dur': 1e8,
        # 'imaging_laser': red_laser, 'imaging_readout_dur': 1000,
        "spin_laser": green_laser,
        "spin_laser_filter": nd,
        "spin_pol_dur": 1e5,
        "spin_readout_dur": 350,
        "nv-_reionization_laser": green_laser,
        "nv-_reionization_dur": 1e5,
        "nv-_prep_laser": green_laser,
        "nv-_prep_laser_dur": 1e5,
        "nv-_prep_laser_filter": "nd_0.5",
        "nv0_ionization_laser": red_laser,
        "nv0_ionization_dur": 1000,
        "nv0_prep_laser": red_laser,
        "nv0_prep_laser_dur": 1000,
        "spin_shelf_laser": yellow_laser,
        "spin_shelf_dur": 0,
        "initialize_laser": green_laser,
        "initialize_dur": 1e4,
        "CPG_laser": red_laser,
        "CPG_laser_dur": 3e3,
        "charge_readout_laser": yellow_laser,
        "charge_readout_dur": 50e6,
        "collection_filter": None,
        "magnet_angle": None,
        "resonance_LOW": 2.8144,
        "rabi_LOW": 131.0,
        "uwave_power_LOW": 16.5,
        "resonance_HIGH": 2.9239,
        "rabi_HIGH": 183.5,
        "uwave_power_HIGH": 16.5,
    }

    # readout_times = [10*10**3, 50*10**3, 100*10**3, 500*10**3,
    #                 1*10**6, 2*10**6, 3*10**6, 4*10**6, 5*10**6,
    #                 6*10**6, 7*10**6, 8*10**6, 9*10**6, 1*10**7,
    #                 2*10**7, 3*10**7, 4*10**7, 5*10**7]
    # readout_times = numpy.linspace(10e6, 50e6, 5)
    readout_times = [10e6, 25e6, 50e6, 100e6, 200e6]  # , 400e6, 700e6, 1e9]
    # readout_times = numpy.linspace(100e6, 1e9, 10)
    # readout_times = numpy.linspace(700e6, 1e9, 7)
    # readout_times = [50e6, 100e6, 200e6, 400e6, 1e9]
    # readout_times = [2e9]
    readout_times = [int(el) for el in readout_times]
    max_readout = max(readout_times)

    # readout_powers = numpy.linspace(0.6, 1.0, 5)
    readout_powers = np.linspace(0.7, 0.9, 6)
    # readout_powers = numpy.linspace(0.76, 0.8, 5)
    # readout_powers = numpy.linspace(0.2, 1.0, 5)
    # readout_powers = [0.65]

    try:
        nv0, nvm = determine_readout_dur_power(
            nv_sig,
            nv_sig,
            apd_indices,
            max_readout=max_readout,
            readout_powers=readout_powers,
            nd_filter=None,
        )
        for dur in readout_times:
            plot_histogram(nv_sig, nv0, nvm, dur)
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print("\n\nRoutine complete. Press enter to exit.")
            tool_belt.poll_safe_stop()
