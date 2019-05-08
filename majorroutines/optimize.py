# -*- coding: utf-8 -*-
"""
Optimize on an NV

Created on Thu Apr 11 11:19:56 2019

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.stationary_count as stationary_count
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
from twisted.logger import Logger

log = Logger()


# %% Functions


def read_timed_counts(cxn, num_steps, period, apd_index):

    cxn.apd_counter.load_stream_reader(apd_index, period, num_steps)
    
    num_read_so_far = 0
    counts = []

    timeout_duration = ((period*(10**-9)) * num_steps) + 10
    timeout_inst = time.time() + timeout_duration

    cxn.pulse_streamer.stream_start(num_steps)

    while num_read_so_far < num_steps:

        if time.time() > timeout_inst:
            log.failure('Timed out before all samples were collected.')
            break
        
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_counter.read_stream(apd_index)
        num_new_samples = len(new_samples)
        if num_new_samples > 0:
            counts.extend(new_samples)
            num_read_so_far += num_new_samples

    return numpy.array(counts, dtype=int)
    
    
def do_plot_data(fig, ax, title, voltages, k_counts_per_sec, 
                 optimizationFailed, optiParams):
    ax.plot(voltages, k_counts_per_sec)
    ax.set_title(title)
    ax.set_xlabel('Volts (V)')
    ax.set_ylabel('Count rate (kcps)')

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

# %% Main


def main(cxn, coords, nd_filter, apd_index, name='untitled', prev_max_counts=None,
         set_to_opti_centers=True, save_data=False, plot_data=False):
    
    readout = 10**6
    lower_threshold = prev_max_counts * 2/3
    upper_threshold = prev_max_counts * 4/3
    
    optimization_success = False
    
    # Try to optimize twice
    for ind in range(2):
        
        opti_centers = do_optimize(cxn, coords, nd_filter, apd_index, name='untitled', prev_max_counts=None,
         set_to_opti_centers, save_data, plot_data)
        
        # If optimization succeeds, go on
        if None not in opti_centers:
            
            # If there is a threshold set, go on
            if prev_max_counts != None:
                
                # check the counts
                values = stationary_counts.main(cxn, coords, nd_filter, run_time, readout, apd_index, name)
                opti_counts = values[0]
                
                # If the counts are close to what we expect, we succeeded!
                if lower_threshold <= opti_counts and opti_counts <= upper_threshold:
                    print("optimization success and counts within threshold!")
                    optimization_success = True
                else:
                    print("optimization success, but counts outside of threshold")
                    
             # If the threshold is not set, we succeed based only on optimize       
             else:
                 print("opimization success, no threshold set")
                 optimization_success = True
        # Optimize fails    
        else:
            print("optimization failed")
 
    if optimization_success:
        if set_to_opti_centers:
            cxn.galvo.write(opti_centers[0], opti_centers[1])
            cxn.objective_piezo.write_voltage(opti_centers[2])
        else:
            print('centers: \n' + '{:.3f}, {:.3f}, {:.1f}'.format(*opti_centers))
            drift = numpy.array(opti_centers) - numpy.array(coords)
            print('drift: \n' + '{:.3f}, {:.3f}, {:.1f}'.format(*drift))
    else:
        # Let the user know something went wrong and reset to what was passed
        print('Centers could not be located.')
        if set_to_opti_centers:
            cxn.galvo.write(x_center, y_center)
            cxn.objective_piezo.write_voltage(z_center)
        else:
            center_texts = []
            for center_ind in range(len(opti_centers)):
                center = opti_centers[center_ind]
                center_text = 'None'
                if center is not None:
                    if center_ind == 3:
                        center_text = '{:.1f}'
                    else:
                        center_text = '{:.3f}'
                    center_text = center_text.format(center)
                center_texts.append(center_text)
            print(opti_centers)
            print(', '.join(center_texts))                               
    
    
    
    
    
def do_optimize(cxn, coords, nd_filter, apd_index, name='untitled', prev_max_counts=None,
         set_to_opti_centers=True, save_data=False, plot_data=False):

    # %% Initial set up

    x_center, y_center, z_center = coords

    readout = 10 * 10**6
    readout_sec = readout / 10**9  # Calculate the readout in seconds

    num_steps = 51

    xy_range = 0.02
    z_range = 5.0

    # The galvo's small angle step response is 400 us
    # Let's give ourselves a buffer of 500 us (500000 ns)
    delay = int(0.5 * 10**6)

    # List to store the optimized centers
    opti_centers = [None, None, None]
    
    # Create 3 plots in the figure, one for each axis
    if plot_data:
        fig, axes_pack = plt.subplots(1, 3, figsize=(17, 8.5))
#        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    tool_belt.init_safe_stop()

    # %% Shared x/y setup

    ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',
                                              [delay, readout, apd_index])
    period = ret_vals[0]
    
    cxn.objective_piezo.write_voltage(z_center)
    
    # %% Collect the x counts

    x_voltages = cxn.galvo.load_x_scan(x_center, y_center, xy_range,
                                       num_steps, period)
    
    # Collect the data
    x_counts = read_timed_counts(cxn, num_steps, period, apd_index)

    if tool_belt.safe_stop():
        return opti_centers

    # Fit
    k_counts_per_sec = (x_counts / 1000) / readout_sec
    init_fit = ((23. / readout) * 10**6, x_center, xy_range / 3, 50.)
    try:
        optiParams, cov_arr = curve_fit(tool_belt.gaussian, x_voltages,
                                        k_counts_per_sec, p0=init_fit)
        
        optimizationFailed = False
    except Exception:
        optimizationFailed = True

    if not optimizationFailed:
        opti_centers[0] = optiParams[1]
        
    # Plot
    if plot_data:
        do_plot_data(fig, axes_pack[0], 'X Axis', x_voltages, k_counts_per_sec, 
                     optimizationFailed, optiParams)
    
    # %% Collect the y counts

    y_voltages = cxn.galvo.load_y_scan(x_center, y_center, xy_range,
                                       num_steps, period)
    
    # Collect the data
    y_counts = read_timed_counts(cxn, num_steps, period, apd_index)

    if tool_belt.safe_stop():
        return opti_centers

    # Fit
    k_counts_per_sec = (y_counts / 1000) / readout_sec
    init_fit = ((23. / readout) * 10**6, y_center, xy_range / 3, 50.)
    try:
        optiParams, cov_arr = curve_fit(tool_belt.gaussian, y_voltages,
                                        k_counts_per_sec, p0=init_fit)
        
                
        optimizationFailed = False
    except Exception:
        optimizationFailed = True

    if not optimizationFailed:
        opti_centers[1] = optiParams[1]
    
    # Plot
    if plot_data:
        do_plot_data(fig, axes_pack[1], 'Y Axis', y_voltages, k_counts_per_sec, 
                     optimizationFailed, optiParams)

    # %% Collect the z counts

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

    z_counts = numpy.zeros(num_steps, dtype=int)

    cxn.objective_piezo.write_voltage(z_voltages[0])
    time.sleep(0.5)

    for ind in range(num_steps):
        
        if tool_belt.safe_stop():
            break

        cxn.objective_piezo.write_voltage(z_voltages[ind])

        # Start the timing stream
        cxn.pulse_streamer.stream_start()

        z_counts[ind] = int(cxn.apd_counter.read_stream(apd_index, 1)[0])

    # Fit
    k_counts_per_sec = (z_counts / 1000) / readout_sec
    init_fit = ((23. / readout) * 10**6, z_center, z_range / 2, 0.)
    try:
        optiParams, cov_arr = curve_fit(tool_belt.gaussian, z_voltages,
                                        k_counts_per_sec, p0=init_fit)
        

        
        optimizationFailed = False
    except Exception:
        optimizationFailed = True

    if not optimizationFailed:
        opti_centers[2] = optiParams[1]
    
    # Plot
    if plot_data:
        do_plot_data(fig, axes_pack[2], 'Z Axis', z_voltages, k_counts_per_sec, 
                     optimizationFailed, optiParams)
        
    # %% Read out the counts at the nv to check if we are still on it
    
    values = stationary_count.main(cxn, coords, nd_filter, 10**9, readout, apd_index) 
    live_counts = values[0] #in kcounts/ sec
    print(live_counts)

    # %% Save the data

    # Don't bother saving the data if we're just using this to find the
    # optimized coordinates
    if save_data:

        timestamp = tool_belt.get_time_stamp()

        rawData = {'timestamp': timestamp,
                   'name': name,
                   'coords': coords.tolist(),
                   'coords-units': 'V',
                   'nd_filter': nd_filter,
                   'xy_range': xy_range,
                   'xy_range-units': 'V',
                   'z_range': z_range,
                   'xy_range-units': 'V',
                   'num_steps': num_steps,
                   'readout': readout,
                   'readout-units': 'ns',
                   'x_voltages': x_voltages.tolist(),
                   'x_voltages-units': 'V',
                   'y_voltages': y_voltages.tolist(),
                   'y_voltages-units': 'V',
                   'z_voltages': z_voltages.tolist(),
                   'z_voltages-units': 'V',
                   'xyz_centers': [x_center, y_center, z_center],
                   'xyz_centers-units': 'V',
                   'x_counts': x_counts.tolist(),
                   'x_counts-units': 'counts',
                   'y_counts': y_counts.tolist(),
                   'y_counts-units': 'counts',
                   'z_counts': z_counts.tolist(),
                   'z_counts-units': 'counts'}

        filePath = tool_belt.get_file_path(__file__, timestamp, name)
        tool_belt.save_raw_data(rawData, filePath)
        if plot_data:
            tool_belt.save_figure(fig, filePath)
            
    # %% Return opticenters
    
    return opti_centers

#    # %% Set and return the optimized centers
#
#    if None not in opti_centers:
#        if set_to_opti_centers:
#            cxn.galvo.write(opti_centers[0], opti_centers[1])
#            cxn.objective_piezo.write_voltage(opti_centers[2])
#        else:
#            print('centers: \n' + '{:.3f}, {:.3f}, {:.1f}'.format(*opti_centers))
#            drift = numpy.array(opti_centers) - numpy.array(coords)
#            print('drift: \n' + '{:.3f}, {:.3f}, {:.1f}'.format(*drift))
#    else:
#        # Let the user know something went wrong and reset to what was passed
#        print('Centers could not be located.')
#        if set_to_opti_centers:
#            cxn.galvo.write(x_center, y_center)
#            cxn.objective_piezo.write_voltage(z_center)
#        else:
#            center_texts = []
#            for center_ind in range(len(opti_centers)):
#                center = opti_centers[center_ind]
#                center_text = 'None'
#                if center is not None:
#                    if center_ind == 3:
#                        center_text = '{:.1f}'
#                    else:
#                        center_text = '{:.3f}'
#                    center_text = center_text.format(center)
#                center_texts.append(center_text)
#            print(opti_centers)
#            print(', '.join(center_texts))
#
#    return 

