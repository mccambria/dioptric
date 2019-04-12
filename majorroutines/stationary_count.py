# -*- coding: utf-8 -*-
"""
Sit on the passed coordinates and take counts.

Created on Fri Apr 12 09:25:24 2019

@author: mccambria
"""


# %% Imports


import Utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
import time
from twisted.logger import Logger
log = Logger()


# %% Functions


def update_line_plot(new_samples, num_read_so_far, *args):

    fig, samples, write_pos, readout_sec = args

    # Write to the samples array
    cur_write_pos = write_pos[0]
    new_write_pos = cur_write_pos + len(new_samples)
    samples[cur_write_pos: new_write_pos] = new_samples
    write_pos[0] = new_write_pos

    # Update the figure in k counts per sec
    tool_belt.update_line_plot_figure(fig, samples / (10**3 * readout_sec))


# %% Main


def main(cxn, name, coords, run_time, readout, apd_index, continuous=False):

    # %% Some initial calculations

    x_center, y_center, z_center = coords

    total_num_samples = int((run_time * 10**9) / readout)

    # convert readout to seconds
    readout_sec = readout / 10**9

    # %% Load the PulseStreamer

    # We require bookends on samples so stream one extra cycle
    seq_cycles = total_num_samples + 1
    period = cxn.pulse_streamer.stream_load('simple_readout.py', seq_cycles,
                                            [0, readout, apd_index])

    # %% Set x, y, and z

    cxn.galvo.write(x_center, y_center)
    cxn.objective_piezo.write_voltage(z_center)

    # %% Set up the APD

    cxn.apd_counter.load_stream_reader(apd_index, period, total_num_samples)

    # %% Initialize the figure

    samples = numpy.empty(total_num_samples)
    samples.fill(numpy.nan)  # Only floats support NaN
    write_pos = [0]  # This is a list because we need a mutable variable

    # Set up the line plot
#    x_vals = numpy.arange(totalNumSamples) + 1
#    x_vals *= period / (10**9)  # Elapsed time in s
#    fig = tool_belt.create_line_plot_figure(samples, x_vals)
    fig = tool_belt.create_line_plot_figure(samples)

    # Set labels
    axes = fig.get_axes()
    ax = axes[0]
    ax.set_title("Stationary Counts")
    ax.set_xlabel('Elapsed time (s)')
    ax.set_ylabel('kcts/sec')

    # Maximize the window
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()

    # Args to pass to update_line_plot
    args = fig, samples, write_pos, readout_sec

    # %% Collect the data

    cxn.pulse_streamer.stream_start()

    timeout_duration = ((period*(10**-9)) * total_num_samples) + 10
    timeout_inst = time.time() + timeout_duration

    num_read_so_far = 0

    tool_belt.init_safe_stop()

    while num_read_so_far < total_num_samples:

        if time.time() > timeout_inst:
            log.failure('Timed out before all samples were collected.')
            break

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_counter.read_stream(apd_index)
        num_new_samples = len(new_samples)
        if num_new_samples > 0:
            update_line_plot(new_samples, num_read_so_far, *args)
            num_read_so_far += num_new_samples

    # %% Clean up

    # Stop the pulser
    cxn.pulse_streamer.constant_default()

    # Close tasks
    cxn.apd_counter.close_task(apd_index)

    # %% Report the data

    average = numpy.mean(samples[0:write_pos[0]]) / (10**3 * readout_sec)
    print("average: {0:d}".format(int(average)))
    st_dev = numpy.std(samples[0:write_pos[0]]) / (10**3 * readout_sec)
    print("standard deviation: {0:.3f}".format(st_dev))

    return average, st_dev
