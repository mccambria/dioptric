# -*- coding: utf-8 -*-
"""
Sit on the passed coordinates and take counts.

Created on Fri Apr 12 09:25:24 2019

@author: mccambria
"""


# %% Imports


import time

import labrad
import matplotlib.pyplot as plt
import numpy

import majorroutines.stationary_count as stationary_count
import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt

# %% Functions


# %% Main


def two_pulse_main(
    nv_sig,
    num_steps,
    init_color,
    read_color,
    init_time,
    readout_time,
    apd_indices,
    continuous=False,
):
    with labrad.connect() as cxn:
        average, st_dev = two_pulse_main_with_cxn(
            cxn,
            nv_sig,
            num_steps,
            init_color,
            read_color,
            init_time,
            readout_time,
            apd_indices,
            continuous,
        )

    return average, st_dev


def two_pulse_main_with_cxn(
    cxn,
    nv_sig,
    num_steps,
    init_color,
    read_color,
    init_time,
    readout_time,
    apd_indices,
    continuous=False,
):
    # %% Some initial setup

    tool_belt.reset_cfm_wout_uwaves(cxn)

    shared_parameters = tool_belt.get_shared_parameters_dict(cxn)
    #    readout = shared_parameters['continuous_readout_dur']*10
    readout_sec = readout_time / 10**9

    aom_ao_589_pwr = nv_sig["am_589_power"]

    if init_color == 532:
        init_delay = shared_parameters["515_laser_delay"]
    elif init_color == 589:
        init_delay = shared_parameters["589_aom_delay"]
    elif init_color == 638:
        init_delay = shared_parameters["638_laser_delay"]

    if read_color == 532:
        read_delay = shared_parameters["515_laser_delay"]
    elif read_color == 589:
        read_delay = shared_parameters["589_aom_delay"]
    elif read_color == 638:
        read_delay = shared_parameters["638_laser_delay"]

    # %% Optimize

    #    optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532)

    drift = numpy.array(tool_belt.get_drift())
    coords = numpy.array(nv_sig["coords"])

    coords_drift = coords + drift

    cxn.galvo.write(coords_drift[0], coords_drift[1])
    cxn.objective_piezo.write(coords_drift[2])
    #    print(coords_drift)

    # %% Load the PulseStreamer
    seq_args = [
        0,
        init_delay,
        read_delay,
        init_time,
        readout_time,
        aom_ao_589_pwr,
        apd_indices[0],
        init_color,
        read_color,
    ]

    #    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(
        "simple_readout_two_pulse.py", seq_args_string
    )
    period = ret_vals[0]

    #    total_num_samples = int(run_time / period)

    # %% Set up the APD

    cxn.apd_tagger.start_tag_stream(apd_indices)

    # %% Initialize the figure

    samples = numpy.empty(num_steps)
    samples.fill(numpy.nan)  # Only floats support NaN
    write_pos = [0]  # This is a list because we need a mutable variable

    # Set up the line plot
    x_vals = numpy.arange(num_steps) + 1
    x_vals = x_vals / (10**9) * period  # Elapsed time in s

    fig = tool_belt.create_line_plot_figure(samples, x_vals)
    #    fig = tool_belt.create_line_plot_figure(samples)

    # Set labels
    axes = fig.get_axes()
    ax = axes[0]
    ax.set_title("Stationary Counts")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("kcts/sec")

    # Maximize the window
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()

    # Args to pass to update_line_plot
    args = fig, samples, write_pos, readout_sec

    # %% Collect the data

    cxn.pulse_streamer.stream_start(num_steps)

    timeout_duration = ((period * (10**-9)) * num_steps) + 10
    timeout_inst = time.time() + timeout_duration

    num_read_so_far = 0

    tool_belt.init_safe_stop()

    while num_read_so_far < num_steps:
        if time.time() > timeout_inst:
            break

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_tagger.read_counter_simple()
        num_new_samples = len(new_samples)
        if num_new_samples > 0:
            stationary_count.update_line_plot(new_samples, num_read_so_far, *args)
            num_read_so_far += num_new_samples

    # %% Clean up and report the data

    tool_belt.reset_cfm_wout_uwaves(cxn)

    # Replace x/0=inf with 0
    try:
        average = numpy.mean(samples[0 : write_pos[0]]) / (10**3 * readout_sec)
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(average)
        # Assign to 0 based on the passed conditional array
        average[inf_mask] = 0

    try:
        st_dev = numpy.std(samples[0 : write_pos[0]]) / (10**3 * readout_sec)
    except RuntimeWarning as e:
        print(e)
        inf_mask = numpy.isinf(st_dev)
        # Assign to 0 based on the passed conditional array
        st_dev[inf_mask] = 0

    return average, st_dev
