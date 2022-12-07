# -*- coding: utf-8 -*-
"""
Sit on the passed coordinates and record counts

Created on April 12th, 2019

@author: mccambria
"""


import utils.tool_belt as tool_belt
import utils.positioning as positioning
import utils.kplotlib as kpl
import numpy
import matplotlib.pyplot as plt
import labrad
import majorroutines.optimize as optimize


def main(nv_sig, run_time, disable_opt=None, nv_minus_init=False, nv_zero_init=False):
    with labrad.connect() as cxn:
        average, st_dev = main_with_cxn(
            cxn, nv_sig, run_time, disable_opt, nv_minus_init, nv_zero_init
        )
    return average, st_dev


def main_with_cxn(
    cxn, nv_sig, run_time, disable_opt=None, nv_minus_init=False, nv_zero_init=False
):

    ### Initial setup

    if disable_opt is not None:
        nv_sig["disable_opt"] = disable_opt
    tool_belt.reset_cfm(cxn)
    readout = nv_sig["imaging_readout_dur"]
    readout_sec = readout / 10**9
    charge_init = nv_minus_init or nv_zero_init
    # optimize.main_with_cxn(cxn, nv_sig)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    counter_server = tool_belt.get_server_counter(cxn)

    # %% Optimize

    optimize.main_with_cxn(cxn, nv_sig)
    coords = nv_sig['coords']
    drift = positioning.get_drift(cxn)
    adj_coords = []
    for i in range(3):
        adj_coords.append(coords[i] + drift[i])
    positioning.set_xyz(cxn, adj_coords)

    # %% Set up the imaging laser

    laser_key = 'imaging_laser'
    readout_laser = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)

    # Charge init setup and sequence processing
    if charge_init:
        if nv_minus_init:
            laser_key = "nv-_prep_laser"
        elif nv_zero_init:
            laser_key = "nv0_prep_laser"
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        init = nv_sig["{}_dur".format(laser_key)]
        init_laser = nv_sig[laser_key]
        init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        seq_args = [init, readout, init_laser, init_power, readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        seq_name = "charge_init-simple_readout_background_subtraction.py"
    else:
        delay = 0
        seq_args = [delay, readout, readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        seq_name = "simple_readout.py"
    ret_vals = pulsegen_server.stream_load(seq_name, seq_args_string)
    period = ret_vals[0]

    total_num_samples = int(run_time / period)
    run_time_s = run_time * 1e-9

    # Figure setup
    samples = numpy.empty(total_num_samples)
    samples.fill(numpy.nan)  # NaNs don't get plotted
    write_pos = 0
    x_vals = numpy.arange(total_num_samples) + 1
    x_vals = x_vals / (10**9) * period  # Elapsed time in s
    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    kpl.plot_line(ax, x_vals, samples)
    ax.set_xlim(-0.05*run_time_s, 1.05*run_time_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count rate (kcps)")
    plt.get_current_fig_manager().window.showMaximized()  # Maximize the window

    ### Collect the data

    counter_server.start_tag_stream()
    pulsegen_server.stream_start(-1)
    tool_belt.init_safe_stop()
    # b = 0  # If this just for the OPX, please find a way to implement that does not interfere with other expts

    # Run until user says stop
    while True:
        # b = b + 1
        # if (b % 50) == 0 and (pulsegen_server == "QM_opx"):
        #     tool_belt.reset_cfm(cxn)
        #     counter_server.start_tag_stream()
        #     pulsegen_server.stream_start(-1)
        #     print("restarting")

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        if charge_init:
            new_samples = counter_server.read_counter_modulo_gates(2)
        else:
            new_samples = counter_server.read_counter_simple()

        # Read the samples and update the image
        num_new_samples = len(new_samples)
        if num_new_samples > 0:

            # If we did charge init, subtract out the non-initialized count rate
            if charge_init:
                new_samples = [max(int(el[0]) - int(el[1]), 0) for el in new_samples]

            num_samples = numpy.count_nonzero(~numpy.isnan(samples))

            # If we're going to overflow, shift everything over and drop earliest samples
            overflow = (num_samples + num_new_samples) - total_num_samples
            if overflow > 0:
                num_nans = max(total_num_samples - num_samples, 0)
                samples[::] = numpy.append(
                    samples[num_new_samples - num_nans : total_num_samples - num_nans],
                    new_samples,
                )
            else:
                cur_write_pos = write_pos
                new_write_pos = cur_write_pos + num_new_samples
                samples[cur_write_pos:new_write_pos] = new_samples
                write_pos = new_write_pos

            # Update the figure in k counts per sec
            samples_kcps = samples / (10**3 * readout_sec)
            kpl.plot_line_update(ax, x =x_vals, y =samples_kcps, relim_x=False)
            

    ### Clean up and report average and standard deviation

    tool_belt.reset_cfm(cxn)
    average = numpy.mean(samples[0:write_pos]) / (10**3 * readout_sec)
    print(f"Average: {average}")
    st_dev = numpy.std(samples[0:write_pos]) / (10**3 * readout_sec)
    print(f"Standard deviation: {st_dev}")
    return average, st_dev
