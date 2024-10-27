# -*- coding: utf-8 -*-
"""
Created on mon Apr 8 10:45:09 2020

This file will test the charge state with variable dark times between either
a green then yellow readout or a red and yellow readout.

@author: agardill
"""
import time

import labrad
import matplotlib.pyplot as plt
import numpy

import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt

# %%
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)


# %%
def plot_time_sweep(test_pulse_dur_list, sig_count_list, title, text=None):
    # turn the list into an array, so we can convert into us
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 8.5))
    ax.plot(test_pulse_dur_list / 10**9, sig_count_list, "bo")
    ax.set_xlabel("Dark time (s)")
    ax.set_ylabel("Counts (single shot)")
    ax.set_xscale("log")
    ax.set_title(title)
    #    ax.legend()

    ax.set_title(title)
    if text:
        ax.text(
            0.50,
            0.90,
            text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

    return fig


# %% Main
# Connect to labrad in this file, as opposed to control panel
def main(
    nv_sig, apd_indices, num_reps, initial_pulse_time, dark_time, init_pulse_color
):
    with labrad.connect() as cxn:
        sig_counts = main_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            num_reps,
            initial_pulse_time,
            dark_time,
            init_pulse_color,
        )

    return sig_counts


def main_with_cxn(
    cxn, nv_sig, apd_indices, num_reps, initial_pulse_time, dark_time, init_pulse_color
):
    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    readout_time = nv_sig["pulsed_SCC_readout_dur"]
    aom_ao_589_pwr = nv_sig["am_589_power"]
    nd_filter = nv_sig["nd_filter"]

    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)

    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    # delay of aoms and laser
    laser_515_delay = shared_params["515_laser_delay"]
    aom_589_delay = shared_params["589_aom_delay"]
    laser_638_delay = shared_params["638_DM_laser_delay"]

    #    wait_time = shared_params['post_polarization_wait_dur']

    if init_pulse_color == 532:
        init_pulse_delay = laser_515_delay
    elif init_pulse_color == 638:
        init_pulse_delay = laser_638_delay

    # Estimate the lenth of the sequance
    file_name = "time_resolved_readout.py"
    seq_args = [
        readout_time,
        readout_time,
        initial_pulse_time,
        dark_time,
        init_pulse_delay,
        aom_589_delay,
        aom_ao_589_pwr,
        apd_indices[0],
        init_pulse_color,
        589,
    ]
    #    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    # Collect data

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
    sample_counts = new_counts[0]

    # signal counts are even - get every second element starting from 0
    sig_counts = numpy.average(sample_counts)

    cxn.apd_tagger.stop_tag_stream()

    return sig_counts


# %% The routine if we want to moniter dark times in the second-long times.
def main_s(
    nv_sig, apd_indices, num_reps, initial_pulse_time, dark_time, init_pulse_color
):
    with labrad.connect() as cxn:
        sig_counts = main_s_with_cxn(
            cxn,
            nv_sig,
            apd_indices,
            num_reps,
            initial_pulse_time,
            dark_time,
            init_pulse_color,
        )

    return sig_counts


def main_s_with_cxn(
    cxn, nv_sig, apd_indices, num_reps, initial_pulse_time, dark_time, init_pulse_color
):
    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    readout_time = nv_sig["pulsed_SCC_readout_dur"]
    aom_ao_589_pwr = nv_sig["am_589_power"]
    nd_filter = nv_sig["nd_filter"]

    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)

    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    # delay of aoms and laser
    laser_515_delay = shared_params["515_laser_delay"]
    aom_589_delay = shared_params["589_aom_delay"]
    laser_638_delay = shared_params["638_DM_laser_delay"]

    # Create a list to store the counts
    sig_counts = []

    if init_pulse_color == 532:
        init_pulse_delay = laser_515_delay
    elif init_pulse_color == 638:
        init_pulse_delay = laser_638_delay

    # Estimate the lenth of the sequance
    for i in range(num_reps):
        # shine the initial pulse for the specified time
        file_name = "simple_pulse.py"
        seq_args = [
            init_pulse_delay,
            initial_pulse_time,
            aom_ao_589_pwr,
            init_pulse_color,
        ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(file_name, 1, seq_args_string)

        # Wait the dark time. We pass in the time in ns, so convert to s
        time.sleep(dark_time / 10**9)

        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        # Collect counts
        file_name = "simple_readout.py"
        seq_args = [aom_589_delay, readout_time, aom_ao_589_pwr, apd_indices[0], 589]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        # Clear the buffer
        cxn.apd_tagger.clear_buffer()
        # Run the sequence
        cxn.pulse_streamer.stream_immediate(file_name, 1, seq_args_string)
        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        sig_counts.append(sample_counts)

        cxn.apd_tagger.stop_tag_stream()

    sig_counts_avg = numpy.average(sig_counts)
    return sig_counts_avg


# %%


def do_dark_time_w_red(nv_sig, test_pulse_dur_list=None):
    apd_indices = [0]
    num_reps = 100
    if not test_pulse_dur_list:
        #        test_pulse_dur_list = [10**3,5*10**3, 10**4,2*10**4,5*10**4,10**5,2*10**5, 5*10**5, 10**6, 5*10**6,
        #                               10**7, 5*10**7]
        test_pulse_dur_list = [
            10**3,
            2 * 10**3,
            3 * 10**3,
            4 * 10**3,
            5 * 10**3,
            6 * 10**3,
            7 * 10**3,
            8 * 10**3,
            9 * 10**3,
            10**4,
            2 * 10**4,
            3 * 10**4,
            4 * 10**4,
            10**5,
        ]
    initial_pulse_time = 10**6

    # create some lists for data
    sig_count_list = []

    # Step through the pulse lengths for the dark time
    for test_pulse_length in test_pulse_dur_list:
        if test_pulse_length < 5 * 10**8:
            print("{} ms".format(test_pulse_length / 10**6))
            sig_count = main(
                nv_sig,
                apd_indices,
                num_reps,
                initial_pulse_time,
                test_pulse_length,
                638,
            )
            sig_count_list.append(sig_count)
        else:
            print("{} s".format(test_pulse_length / 10**9))
            sig_count = main_s(
                nv_sig,
                apd_indices,
                num_reps,
                initial_pulse_time,
                test_pulse_length,
                638,
            )
            sig_count_list.append(sig_count)
    # Plot
    title = "Sweep dark time after {} ms red pulse".format(initial_pulse_time / 10**6)
    fig = plot_time_sweep(test_pulse_dur_list, sig_count_list, title)
    # Save
    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "init_color_ind": "638 nm",
        "initial_pulse_time": initial_pulse_time,
        "initial_pulse_time-units": "ns",
        "num_reps": num_reps,
        "test_pulse_dur_list": test_pulse_dur_list,
        "test_pulse_dur_list-units": "ns",
        "sig_count_list": sig_count_list,
        "sig_count_list-units": "counts",
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_raw_data(raw_data, file_path + "-dark_time_w_638")

    tool_belt.save_figure(fig, file_path + "-dark_time_w_638")

    print(" \nRoutine complete!")
    return


# %%


def do_dark_time_w_green(nv_sig, test_pulse_dur_list=None):
    apd_indices = [0]
    num_reps = 100
    if not test_pulse_dur_list:
        test_pulse_dur_list = [
            10**3,
            5 * 10**3,
            10**4,
            2 * 10**4,
            5 * 10**4,
            10**5,
            2 * 10**5,
            5 * 10**5,
            10**6,
            5 * 10**6,
            10**7,
            5 * 10**7,
        ]
    initial_pulse_time = 10**6

    # create some lists for data
    sig_count_list = []

    # Step through the pulse lengths for the dark time
    for test_pulse_length in test_pulse_dur_list:
        if test_pulse_length < 5 * 10**8:
            print("{} ms".format(test_pulse_length / 10**6))
            sig_count = main(
                nv_sig,
                apd_indices,
                num_reps,
                initial_pulse_time,
                test_pulse_length,
                532,
            )
            sig_count_list.append(sig_count)
        else:
            print("{} s".format(test_pulse_length / 10**9))
            sig_count = main_s(
                nv_sig,
                apd_indices,
                num_reps,
                initial_pulse_time,
                test_pulse_length,
                532,
            )
            sig_count_list.append(sig_count)

    # Plot
    title = "Sweep dark time after {} ms green pulse".format(initial_pulse_time / 10**6)
    fig = plot_time_sweep(test_pulse_dur_list, sig_count_list, title)
    # Save
    timestamp = tool_belt.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "init_color_ind": "532 nm",
        "initial_pulse_time": initial_pulse_time,
        "initial_pulse_time-units": "ns",
        "num_reps": num_reps,
        "test_pulse_dur_list": test_pulse_dur_list,
        "test_pulse_dur_list-units": "ns",
        "sig_count_list": sig_count_list,
        "sig_count_list-units": "counts",
    }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_raw_data(raw_data, file_path + "-dark_time_w_532")

    tool_belt.save_figure(fig, file_path + "-dark_time_w_532")

    print(" \nRoutine complete!")
    return


# %% Run the files

if __name__ == "__main__":
    sample_name = "choy"

    nv1 = {
        "coords": [0.227, -0.275, 5.0],
        "name": "{}-nv1".format(sample_name),
        "expected_count_rate": 140,
        "nd_filter": "nd_0",
        "pulsed_readout_dur": 300,
        "pulsed_SCC_readout_dur": 1 * 10**7,
        "am_589_power": 0.25,
        "pulsed_initial_ion_dur": 25 * 10**3,
        "pulsed_shelf_dur": 200,
        "am_589_shelf_power": 0.35,
        "pulsed_ionization_dur": 500,
        "cobalt_638_power": 160,
        "pulsed_reionization_dur": 100 * 10**3,
        "cobalt_532_power": 8,
        "magnet_angle": 20,
        "resonance_LOW": 2.8181,
        "rabi_LOW": 137,
        "uwave_power_LOW": 9.0,
        "resonance_HIGH": 2.9675,
        "rabi_HIGH": 95.2,
        "uwave_power_HIGH": 10.0,
    }
    nv_sig = nv1

    #    do_dark_time_w_green(nv_sig, test_pulse_dur_list = [10**3, 5*10**3, 10**4, 5*10**4,10**5, 5*10**5, 10**6, 5*10**6,
    #                               10**7, 5*10**7, 10**8])
    do_dark_time_w_green(
        nv_sig,
        test_pulse_dur_list=[
            10**3,
            5 * 10**3,
            10**4,
            5 * 10**4,
            10**5,
            5 * 10**5,
            10**6,
            5 * 10**6,
            10**7,
            5 * 10**7,
            10**8,
            5 * 10**8,
            10**9,
            5 * 10**9,
            10**10,
        ],
    )
    do_dark_time_w_red(
        nv_sig,
        test_pulse_dur_list=[
            10**3,
            5 * 10**3,
            10**4,
            5 * 10**4,
            10**5,
            5 * 10**5,
            10**6,
            5 * 10**6,
            10**7,
            5 * 10**7,
            10**8,
            5 * 10**8,
            10**9,
            5 * 10**9,
            10**10,
        ],
    )
