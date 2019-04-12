# -*- coding: utf-8 -*-
"""
Scans the microwave frequency, taking counts at each point.

Created on Thu Apr 11 15:39:23 2019

@author: mccambria
"""

# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import majorroutines.optimize as optimize

# Library modules
import numpy
import os


# %% Main


def main(cxn, name, x_center, y_center, z_center, apd_index,
         freq_center, freq_range, num_steps, uwave_power):

    # %% Initial calculations and setup

    file_name = os.path.basename(__file__)
    file_name_no_ext = os.path.splitext(file_name)[0]

    # Calculate the frequencies we need to set
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    freqs = numpy.linspace(freq_low, freq_high, num_steps)

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    counts = numpy.empty(num_steps)
    counts[:] = numpy.nan

    # %% Optimize on the passed coordinates

    optimize.main(cxn, name, x_center, y_center, z_center, apd_index)

    # %% Set up the pulser

    delay = 0.1 * 10**9  # 0.1 s to switch frequencies
    readout = 10 * 10**6  # 0.01 to count
    period = delay + readout

    # The sequence library file is named the same as this file
    cxn.pulse_streamer.stream_load(file_name, 1,
                                   [period, readout, apd_index])

    # %% Load the APD task

    cxn.apd_counter.load_stream_reader(apd_index, period, num_steps)

    # %% Set up the plot

    fig = tool_belt.create_line_plot_figure(counts, freqs)

    # %% Collect and plot the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    # Provide the counter with its reference sample
    cxn.pulse_streamer.stream_start()

    # Take a sample and increment the frequency
    for ind in range(num_steps):

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        cxn.uwave_sig_gen.set_freq(freqs[ind])

        # If this is the first sample then we have to enable the signal
        if ind == 0:
            cxn.uwave_sig_gen.set_freq(uwave_power)
            cxn.uwave_sig_gen.uwave_on()

        # Start the timing stream
        cxn.pulse_streamer.stream_start()

        counts[ind] = cxn.apd_counter.read_stream(apd_index, True)[0]

        tool_belt.update_line_plot_figure(fig, counts)

    # %% Turn off the RF and save the data

    cxn.uwave_sig_gen.uwave_off()

    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'name': name,
               'xyz_centers': [x_center, y_center, z_center],
               'freq_center': freq_center,
               'freq_range': freq_range,
               'num_steps': num_steps,
               'uwave_power': uwave_power,
               'readout': readout,
               'delay': delay,
               'counts': counts.astype(int).tolist()}

    filePath = tool_belt.get_file_path(file_name_no_ext, timestamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
