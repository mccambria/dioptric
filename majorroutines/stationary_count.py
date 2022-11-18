# -*- coding: utf-8 -*-
"""
Sit on the passed coordinates and take counts.

Created on Fri Apr 12 09:25:24 2019

@author: mccambria
"""

import utils.tool_belt as tool_belt
import numpy as np
import matplotlib.pyplot as plt
import time
import labrad
import majorroutines.optimize as optimize

# region Functions


def update_line_plot(new_samples, num_read_so_far, *args):

    fig, samples, write_pos, readout_sec, total_num_samples = args

    num_samples = np.count_nonzero(~np.isnan(samples))
    num_new_samples = len(new_samples)

    # If we're going to overflow, just shift everything over and drop the
    # earliest samples
    overflow = (num_samples + num_new_samples) - total_num_samples
    if overflow > 0:
        num_nans = max(total_num_samples - num_samples, 0)
        samples[::] = np.append(
            samples[num_new_samples - num_nans : total_num_samples - num_nans],
            new_samples,
        )
    else:
        cur_write_pos = write_pos[0]
        new_write_pos = cur_write_pos + num_new_samples
        samples[cur_write_pos:new_write_pos] = new_samples
        write_pos[0] = new_write_pos

    # Update the figure in k counts per sec
    tool_belt.update_line_plot_figure(fig, (samples / (10 ** 3 * readout_sec)))


# endregion

# region Main


def main(
    nv_sig,
    run_time,
    apd_indices,
    disable_opt=None,
    nv_minus_initialization=False,
    nv_zero_initialization=False,
    background_subtraction=False,
    background_coords=None,
):

    with labrad.connect() as cxn:
        average, st_dev = main_with_cxn(
            cxn,
            nv_sig,
            run_time,
            apd_indices,
            disable_opt,
            nv_minus_initialization,
            nv_zero_initialization,
            background_subtraction,
            background_coords,
        )

    return average, st_dev


def main_with_cxn(
    cxn,
    nv_sig,
    run_time,
    apd_indices,
    disable_opt=None,
    nv_minus_initialization=False,
    nv_zero_initialization=False,
    background_subtraction=False,
    background_coords=None,
):

    # %% Some initial setup

    if disable_opt is not None:
        nv_sig["disable_opt"] = disable_opt

    tool_belt.reset_cfm(cxn)

    readout = int(nv_sig["imaging_readout_dur"])
    readout_sec = readout / 10 ** 9

    # %% Optimize / positioning setup

    optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    coords = nv_sig["coords"]
    drift = tool_belt.get_drift()
    adj_coords = []
    for i in range(3):
        adj_coords.append(coords[i] + drift[i])
    tool_belt.set_xyz(cxn, adj_coords)
    
    if background_subtraction:
        adj_bg_coords = []
        for i in range(3):
            adj_bg_coords.append(background_coords[i] + drift[i])
        xy_server = tool_belt.get_xy_server(cxn)
        x_voltages, y_voltages = xy_server.load_two_point_xy(adj_coords[0], adj_coords[1],
                                             adj_bg_coords[0], adj_bg_coords[1], readout)

    # %% Set up the imaging laser

    laser_key = "imaging_laser"
    readout_laser = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

    # %% Load the PulseStreamer

    if nv_minus_initialization:
        laser_key = "nv-_prep_laser"
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        init = nv_sig["{}_dur".format(laser_key)]
        init_laser = nv_sig[laser_key]
        init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        seq_args = [
            init,
            readout,
            apd_indices[0],
            init_laser,
            init_power,
            readout_laser,
            readout_power,
        ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(
            "charge_initialization-simple_readout_background_subtraction.py",
            seq_args_string,
        )
    elif nv_zero_initialization:
        laser_key = "nv0_prep_laser"
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        init = nv_sig["{}_dur".format(laser_key)]
        init_laser = nv_sig[laser_key]
        init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        seq_args = [
            init,
            readout,
            apd_indices[0],
            init_laser,
            init_power,
            readout_laser,
            readout_power,
        ]
        # print(seq_args)
        # return
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(
            "charge_initialization-simple_readout_background_subtraction.py",
            seq_args_string,
        )
    elif background_subtraction:
        # See if this setup has finely specified delay times, else just get the
        # one-size-fits-all value.
        dir_path = ['', 'Config', 'Positioning']
        cxn.registry.cd(*dir_path)
        _, keys = cxn.registry.dir()
        if 'xy_small_response_delay' in keys:
            xy_delay = tool_belt.get_registry_entry(cxn,
                                            'xy_small_response_delay', dir_path)
        else:
            xy_delay = tool_belt.get_registry_entry(cxn, 'xy_delay', dir_path)
        seq_args = [xy_delay, readout, apd_indices[0], readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(
            "simple_readout.py", seq_args_string
        )
    else:
        seq_args = [0, readout, apd_indices[0], readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(
            "simple_readout.py", seq_args_string
        )
    period = ret_vals[0]

    total_num_samples = int(run_time / period)

    # %% Initialize the figure

    samples = np.empty(total_num_samples)
    samples.fill(np.nan)  # Only floats support NaN
    write_pos = [0]  # This is a list because we need a mutable variable

    # Set up the line plot
    x_vals = np.arange(total_num_samples) + 1
    x_vals = x_vals / (10 ** 9) * period  # Elapsed time in s

    fig = tool_belt.create_line_plot_figure(samples, x_vals)
    # fig = tool_belt.create_line_plot_figure(samples)

    # Set labels
    axes = fig.get_axes()
    ax = axes[0]
    ax.set_title("Stationary Counts")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("kcps")

    # Maximize the window
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()

    # Args to pass to update_line_plot
    args = fig, samples, write_pos, readout_sec, total_num_samples

    # %% Collect the data

    cxn.apd_tagger.start_tag_stream(apd_indices)
    cxn.pulse_streamer.stream_start(-1)

    # timeout_duration = ((period*(10**-9)) * total_num_samples) + 10
    # timeout_inst = time.time() + timeout_duration

    num_read_so_far = 0

    tool_belt.init_safe_stop()

    charge_initialization = nv_minus_initialization or nv_zero_initialization
    charge_initialization = False
    # print(charge_initialization)
    leftover_sample = None
    snr = lambda nv, bg: (nv - bg) / np.sqrt(nv + bg)

    while True:

        # if time.time() > timeout_inst:
        #     break

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        # start = time.time()
        if charge_initialization:
            new_samples = cxn.apd_tagger.read_counter_modulo_gates(2)
        else:
            new_samples = cxn.apd_tagger.read_counter_simple()
        # stop = time.time()
        # print(f"Collection time: {stop - start}")
        # print(new_samples)
        
        # Read the samples and update the image
        #        print(new_samples)
        num_new_samples = len(new_samples)
        if num_new_samples > 0:

            # If we did charge initialization, subtract out the background
            if charge_initialization:
                new_samples = [
                    max(int(el[0]) - int(el[1]), 0) for el in new_samples
                ]
            if background_subtraction:
                # Make sure we have an even number of samples
                new_samples = np.array(new_samples, dtype=int)
                # print(new_samples)
                if leftover_sample is not None:
                    new_samples = np.insert(new_samples, 0, leftover_sample)
                if len(new_samples) % 2 == 0:
                    leftover_sample = None
                else:
                    leftover_sample = new_samples[-1]
                    new_samples = new_samples[:-1]
                # print(leftover_sample)
                # new_samples = abs(new_samples[::2] - new_samples[1::2])
                new_samples = [snr(new_samples[2*ind], new_samples[2*ind+1]) for ind in range(num_new_samples // 2)]

            # start = time.time()
            update_line_plot(new_samples, num_read_so_far, *args)
            # stop = time.time()
            # print(f"Plot time: {stop - start}")
            num_read_so_far += num_new_samples

    # %% Clean up and report the data

    tool_belt.reset_cfm(cxn)

    # Replace x/0=inf with 0
    try:
        average = np.mean(samples[0 : write_pos[0]]) / (
            10 ** 3 * readout_sec
        )
        print("average: {}".format(average))
    except RuntimeWarning as e:
        print(e)
        inf_mask = np.isinf(average)
        # Assign to 0 based on the passed conditional array
        average[inf_mask] = 0

    try:
        st_dev = np.std(samples[0 : write_pos[0]]) / (10 ** 3 * readout_sec)
        print("st_dev: {}".format(st_dev))
    except RuntimeWarning as e:
        print(e)
        inf_mask = np.isinf(st_dev)
        # Assign to 0 based on the passed conditional array
        st_dev[inf_mask] = 0

    return average, st_dev


# endregion
