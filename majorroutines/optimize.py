# -*- coding: utf-8 -*-
"""
Optimize on an NV

Created on Thu Apr 11 11:19:56 2019

@author: mccambria
"""


# %% Imports

import Utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
from twisted.logger import Logger
log = Logger()

# %% Main


def main(cxn, name, coords, apd_index,
         set_to_opti_centers=True, save_data=False, plot_data=False):

    # %% Initial set up

    x_center, y_center, z_center = coords

    readout = 10 * 10**6

    num_steps = 50

    xy_range = 0.02
    z_range = 5.0

    # List to store the optimized centers
    opti_centers = [None, None, None]

    # %% Collect the x/y counts

    # The galvo's small angle step response is 400 us
    # Let's give ourselves a buffer of 500 us (500000 ns)
    delay = int(0.5 * 10**6)

    ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',
                                              [delay, readout, apd_index])
    period = ret_vals[0]

    ret_vals = cxn.galvo.load_cross_scan(x_center, y_center, xy_range,
                                         num_steps, period)
    x_voltages, y_voltages = ret_vals

    xy_num_steps = len(x_voltages) + len(y_voltages)

    cxn.objective_piezo.write_voltage(z_center)

    cxn.apd_counter.load_stream_reader(apd_index, period, xy_num_steps)

    # Collect the data

    num_read_so_far = 0
    xy_counts = []

    timeout_duration = ((period*(10**-9)) * xy_num_steps) + 10
    timeout_inst = time.time() + timeout_duration

    tool_belt.init_safe_stop()

    cxn.pulse_streamer.stream_start(xy_num_steps)

    while num_read_so_far < xy_num_steps:

        if time.time() > timeout_inst:
            log.failure('Timed out before all samples were collected.')
            break

        if tool_belt.safe_stop():
            return opti_centers

        # Read the samples and update the image
        new_samples = cxn.apd_counter.read_stream(apd_index)
        num_new_samples = len(new_samples)
        if num_new_samples > 0:
            xy_counts.extend(new_samples)
            num_read_so_far += num_new_samples

    xy_counts = numpy.array(xy_counts, dtype=numpy.uint32)

    # %% Collect the z counts

    # If the user said stop, let's just stop
    if tool_belt.safe_stop():
        return opti_centers

    half_z_range = z_range / 2
    z_low = z_center - half_z_range
    z_high = z_center + half_z_range
    z_voltages = numpy.linspace(z_low, z_high, num_steps)

    # Base this off the piezo hysteresis and step response
    delay = int(0.5 * 10**6)  # Assume it's the same as the galvo for now

    # Set up the galvo
    cxn.galvo.write(x_center, y_center)

    # Set up the stream
    ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',
                                              [delay, readout, apd_index])
    period = ret_vals[0]

    # Set up the APD
    cxn.apd_counter.load_stream_reader(apd_index, period, num_steps)

    z_counts = numpy.zeros(num_steps, dtype=numpy.uint32)

    cxn.objective_piezo.write_voltage(z_voltages[0])
    time.sleep(0.5)

    for ind in range(num_steps):

        cxn.objective_piezo.write_voltage(z_voltages[ind])

        # Start the timing stream
        cxn.pulse_streamer.stream_start()

        z_counts[ind] = cxn.apd_counter.read_stream(apd_index, 1)[0]

    # %% Extract each dimension's counts

    # Calculate the readout in seconds
    readout_sec = readout / 10**9

    start = 0
    end = num_steps
    x_counts = xy_counts[start: end]

    start = num_steps
    end = num_steps + num_steps
    y_counts = xy_counts[start: end]

    # %% Fit Gaussians and plot the data

    # Create 3 plots in the figure, one for each axis
    if plot_data:
        fig, axesPack = plt.subplots(1, 3, figsize=(17, 8.5))
#        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Pack up
    voltages_pack = [x_voltages, y_voltages, z_voltages]
    k_counts_per_sec_pack = [(x_counts / 10**3) / (readout_sec * 10**3),
                             (y_counts / 10**3) / (readout_sec * 10**3),
                             (z_counts / 10**3) / (readout_sec * 10**3)]
    titles_pack = ['X Axis', 'Y Axis', 'Z Axis']
    init_fit_pack = [((23. / readout) * 10**6, x_center, xy_range / 3, 50.),
                     ((23. / readout) * 10**6, y_center, xy_range / 3, 50.),
                     ((23. / readout) * 10**6, z_center, z_range / 2, 0.)]

    # Loop over each dimension
    for ind in range(3):

        optimizationFailed = False

        # Unpack for the dimension
        voltages = voltages_pack[ind]
        k_counts_per_sec = k_counts_per_sec_pack[ind]
        title = titles_pack[ind]
        init_fit = init_fit_pack[ind]

        # Least squares
        try:
            optiParams, varianceArr = curve_fit(tool_belt.gaussian, voltages,
                                                k_counts_per_sec, p0=init_fit)
        except Exception:
            optimizationFailed = True

        if not optimizationFailed:
            opti_centers[ind] = optiParams[1]

        # Plot the data
        if plot_data:
            ax = axesPack[ind]
            ax.plot(voltages, k_counts_per_sec)
            ax.set_title(title)
            ax.set_xlabel('Volts (V)')
            ax.set_ylabel('kcts/sec')

            # Plot the fit
            if not optimizationFailed:
                first = voltages[0]
                last = voltages[len(voltages)-1]
                linspaceVoltages = numpy.linspace(first, last, num=1000)
                gaussianFit = tool_belt.gaussian(linspaceVoltages, *optiParams)
                ax.plot(linspaceVoltages, gaussianFit)

                # Add info to the axes
                # a: coefficient that defines the peak height
                # mu: mean, defines the center of the Gaussian
                # sigma: standard deviation, defines the width of the Gaussian
                # offset: constant y value to account for background
                text = 'a={:.3f}\n $\mu$={:.3f}\n ' \
                    '$\sigma$={:.3f}\n offset={:.3f}'.format(*optiParams)

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)

            fig.canvas.draw()
            fig.canvas.flush_events()

    # %% Save the data

    # Don't bother saving the data if we're just using this to find the
    # optimized coordinates
    if save_data:

        timestamp = tool_belt.get_time_stamp()

        rawData = {'timestamp': timestamp,
                   'name': name,
                   'xyz_centers': [x_center, y_center, z_center],
                   'xy_range': xy_range,
                   'z_range': z_range,
                   'num_steps': num_steps,
                   'readout': int(readout),
                   'counts': [x_counts.astype(int).tolist(),
                              y_counts.astype(int).tolist(),
                              z_counts.astype(int).tolist()]}

        filePath = tool_belt.get_file_path('optimize', timestamp, name)
        tool_belt.save_raw_data(rawData, filePath)
        if plot_data:
            tool_belt.save_figure(fig, filePath)

    # %% Set and return the optimized centers

    if None not in opti_centers:
        if set_to_opti_centers:
            cxn.galvo.write(opti_centers[0], opti_centers[1])
            cxn.objective_piezo.write_voltage(opti_centers[2])
        else:
            print('{:.3f}, {:.3f}, {:.3f}'.format(*opti_centers))
    else:
        # Let the user know something went wrong and reset to what was passed
        print('Centers could not be located.')
        if set_to_opti_centers:
            cxn.galvo.write(x_center, y_center)
            cxn.objective_piezo.write_voltage(z_center)
        else:
            print('{:.3f}, {:.3f}, {:.3f}'.format(*opti_centers))

    return opti_centers
