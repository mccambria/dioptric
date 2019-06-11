# -*- coding: utf-8 -*-
"""
Optimize on an NV

Created on Thu Apr 11 11:19:56 2019

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time


# %% Plotting functions


def create_figure():
    fig, axes_pack = plt.subplots(1, 3, figsize=(17, 8.5))
    axis_titles = ['X Axis', 'Y Axis', 'Z Axis']
    for ind in range(3):
        ax = axes_pack[ind]
        ax.set_title(axis_titles[ind])
        ax.set_xlabel('Volts (V)')
        ax.set_ylabel('Count rate (kcps)')
    fig.set_tight_layout(True)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig
    
    
def update_figure_raw_data(fig, axis_ind, voltages, count_rates):
    axes = fig.get_axes()
    ax = axes[axis_ind]
    ax.plot(voltages, count_rates)
    
    
def update_figure_fit(fig, axis_ind, voltages, opti_params):
    axes = fig.get_axes()
    ax = axes[axis_ind]
    # Plot the fit
    first = voltages[0]
    last = voltages[-1]
    linspace_voltages = numpy.linspace(first, last, num=1000)
    fit = tool_belt.gaussian(linspace_voltages, *opti_params)
    ax.plot(linspace_voltages, fit)

    # Add info to the axes
    # a: coefficient that defines the peak height
    # mu: mean, defines the center of the Gaussian
    # sigma: standard deviation, defines the width of the Gaussian
    # offset: constant y value to account for background
    text = 'a={:.3f}\n $\mu$={:.3f}\n ' \
        '$\sigma$={:.3f}\n offset={:.3f}'.format(*opti_params)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    fig.canvas.draw()
    fig.canvas.flush_events()


# %% Other functions


def read_timed_counts(cxn, num_steps, period, apd_indices):

    cxn.apd_tagger.start_tag_stream(apd_indices)
    num_read_so_far = 0
    counts = []

    timeout_duration = ((period*(10**-9)) * num_steps) + 10
    timeout_inst = time.time() + timeout_duration

    cxn.pulse_streamer.stream_start(num_steps)
    
    while num_read_so_far < num_steps:
        
        if time.time() > timeout_inst:
            break
        
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_tagger.read_counter_simple()
        num_new_samples = len(new_samples)
        if num_new_samples > 0:
            counts.extend(new_samples)
            num_read_so_far += num_new_samples

    cxn.apd_tagger.stop_tag_stream()
    
    return numpy.array(counts, dtype=int)

    
def stationary_count_lite(cxn, coords, shared_params, apd_indices):
    
    readout = shared_params['continuous_readout_ns']
    
    #  Some initial calculations
    x_center, y_center, z_center = coords
    readout = readout // 2

    # Load the PulseStreamer
    cxn.pulse_streamer.stream_load('simple_readout.py',
                                   [0, readout, apd_indices[0]])
    total_num_samples = 2

    # Set x, y, and z
    cxn.galvo.write(x_center, y_center)
    cxn.objective_piezo.write_voltage(z_center)

    # Set up the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    
    # Collect the data
    cxn.pulse_streamer.stream_start(total_num_samples)
    new_samples = cxn.apd_tagger.read_counter_simple(total_num_samples)
    new_samples_avg = numpy.average(new_samples)
    cxn.apd_tagger.stop_tag_stream()
    counts_kcps = (new_samples_avg / 1000) / (readout / 10**9)
    
    return counts_kcps
    
    
def optimize_on_axis(cxn, nv_sig, axis_ind, shared_params,
                     apd_indices, fig=None):
    
    seq_file_name = 'simple_readout.py'
    
    axis_center = nv_sig[axis_ind]
    x_center, y_center, z_center = nv_sig[0: 3]
    
    scan_range_nm = 3 * shared_params['airy_radius_nm']
    readout = shared_params['continuous_readout_ns']
    
    num_steps = 51
    
    tool_belt.init_safe_stop()
    
    # x/y
    if axis_ind in [0, 1]:
        
        scan_range = scan_range_nm / shared_params['galvo_nm_per_volt']
        
        seq_params = [shared_params['galvo_delay_ns'],
                      readout,
                      apd_indices[0]]
        ret_vals = cxn.pulse_streamer.stream_load(seq_file_name, seq_params)
        period = ret_vals[0]
        
        # Fix the piezo
        cxn.objective_piezo.write_voltage(z_center)
        
        # Get the proper scan function
        if axis_ind == 0:
            scan_func = cxn.galvo.load_x_scan
        elif axis_ind == 1:
            scan_func = cxn.galvo.load_y_scan
            
        voltages = scan_func(x_center, y_center, scan_range, num_steps, period)
        counts = read_timed_counts(cxn, num_steps, period, apd_indices)
        
    # z
    elif axis_ind == 2:
        
        scan_range = scan_range_nm / shared_params['piezo_nm_per_volt']
        half_scan_range = scan_range / 2
        low_voltage = axis_center - half_scan_range
        high_voltage = axis_center + half_scan_range
        voltages = numpy.linspace(low_voltage, high_voltage, num_steps)
    
        # Fix the galvo
        cxn.galvo.write(x_center, y_center)
    
        # Set up the stream
        seq_params = [shared_params['piezo_delay_ns'],
                      readout,
                      apd_indices[0]]
        ret_vals = cxn.pulse_streamer.stream_load(seq_file_name, seq_params)
        period = ret_vals[0]
    
        # Set up the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
    
        counts = numpy.zeros(num_steps, dtype=int)
    
        for ind in range(num_steps):
            
            if tool_belt.safe_stop():
                break
    
            cxn.objective_piezo.write_voltage(voltages[ind])
    
            # Start the timing stream
            cxn.pulse_streamer.stream_start()
    
            counts[ind] = int(cxn.apd_tagger.read_counter_simple(1)[0])
    
        cxn.apd_tagger.stop_tag_stream()
        
    count_rates = (counts / 1000) / (readout / 10**9)
    
    if fig is not None:
        update_figure_raw_data(fig, axis_ind, voltages, count_rates)
        
    return fit_gaussian(nv_sig, voltages, count_rates, axis_ind, fig)
    
    
def fit_gaussian(nv_sig, voltages, count_rates, axis_ind, fig=None):
        
    # The order of parameters is 
    # 0: coefficient that defines the peak height
    # 1: mean, defines the center of the Gaussian
    # 2: standard deviation, defines the width of the Gaussian
    # 3: constant y value to account for background
    expected_counts = nv_sig[3]
    background_counts = nv_sig[4]
    scan_range = voltages[-1] - voltages[0]
    init_fit = (expected_counts - background_counts,
                nv_sig[axis_ind],
                scan_range / 3,
                background_counts)
    try:
        opti_params, cov_arr = curve_fit(tool_belt.gaussian, voltages,
                                        count_rates, p0=init_fit)
    except Exception:
        print('Optimization failed for axis {}'.format(axis_ind))
        opti_params = None
        
    # Plot
    if fig is not None:
        update_figure_fit(fig, axis_ind, voltages, opti_params)
        
    return opti_params
    

# %% Main


def main(cxn, nv_sig, nd_filter, apd_indices, name='untitled', 
         set_to_opti_centers=True, save_data=False, plot_data=False):
    
    # Get the shared parameters from the registry
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    
    # Create 3 plots in the figure, one for each axis
    fig = None
    if plot_data:
        fig = create_figure()
    
    x_center, y_center, z_center = nv_sig[0:3]
    expected_counts = nv_sig[3]
    background_counts = nv_sig[4]
    
    opti_succeeded = False
    
    # Try to optimize twice
    for ind in range(2):
        
        # Optimize on each axis
        opti_params = []
        for axis_ind in range(3):
            opti_params.append(optimize_on_axis(cxn, nv_sig, axis_ind,
                                                shared_params, apd_indices, fig))
        
        # Get just the coordinates out of the returned parameters
        opti_coords = [None, None, None]
        for axis_ind in range(3):
            params = opti_params[axis_ind]
            if params is not None:
                opti_coords[axis_ind] = params[1]
            
        # If there is a threshold set, go on
        if expected_counts != None:
            
            lower_threshold = expected_counts * 3/4
            upper_threshold = expected_counts * 5/4
            
            # check the counts
            opti_counts = stationary_count_lite(cxn, opti_coords,
                                                shared_params, apd_indices)
            print('Counts from optimization: {}'.format(opti_counts)) 
            print('Expected counts: {}'.format(expected_counts))  
            print(' ')
            
            # If the counts are close to what we expect, we succeeded!
            if lower_threshold <= opti_counts and opti_counts <= upper_threshold:
                print("Optimization success and counts within threshold! \n ")
                optimization_success = True
                break
            else:
                print("Optimization success, but counts outside of threshold \n ")
                
        # If the threshold is not set, we succeed based only on optimize       
        else:
            print("Opimization success, no threshold set \n ")
            optimization_success = True
            break
            
    if optimization_success == True:
        if set_to_opti_centers:
            cxn.galvo.write(opti_coords[0], opti_coords[1])
            cxn.objective_piezo.write_voltage(opti_coords[2])
        else:
            print('centers: \n' + '{:.3f}, {:.3f}, {:.1f}'.format(*opti_coords))
            drift = numpy.array(opti_coords) - numpy.array(coords)
            print('drift: \n' + '{:.3f}, {:.3f}, {:.1f}'.format(*drift))
            
        return opti_coords, optimization_success
    else:
        # Let the user know something went wrong and reset to what was passed
        print('Centers could not be located.')
        if set_to_opti_centers:
            tool_belt.set_xyz_on_nv(cxn, nv_sig)
        else:
            center_texts = []
            for center_ind in range(len(opti_coords)):
                center = opti_coords[center_ind]
                center_text = 'None'
                if center is not None:
                    if center_ind == 3:
                        center_text = '{:.1f}'
                    else:
                        center_text = '{:.3f}'
                    center_text = center_text.format(center)
                center_texts.append(center_text)
            print(opti_coords)
            print(', '.join(center_texts))
                               
        return coords, optimization_success
        
    # %% Save the data

    # Don't bother saving the data if we're just using this to find the
    # optimized coordinates
    if save_data:

        timestamp = tool_belt.get_time_stamp()

        rawData = {'timestamp': timestamp,
                   'name': name,
                   'coords': coords,
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

