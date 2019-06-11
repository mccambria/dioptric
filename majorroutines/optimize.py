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


# %% Define a few parameters

num_steps = 51


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
    
    
def update_figure(fig, axis_ind, voltages, count_rates, text=None):
    axes = fig.get_axes()
    ax = axes[axis_ind]
    ax.plot(voltages, count_rates)

    if text is not None:
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
        update_figure(fig, axis_ind, voltages, count_rates)
        
    opti_coord = fit_gaussian(nv_sig, voltages, count_rates, axis_ind, fig)
        
    return opti_coord, voltages, counts, 
    
    
def fit_gaussian(nv_sig, voltages, count_rates, axis_ind, fig=None):
        
    # The order of parameters is 
    # 0: coefficient that defines the peak height
    # 1: mean, defines the center of the Gaussian
    # 2: standard deviation, defines the width of the Gaussian
    # 3: constant y value to account for background
    expected_counts = nv_sig[3]
    background_counts = nv_sig[4]
    first_voltage = voltages[0]
    last_voltage = voltages[-1]
    scan_range = last_voltage - first_voltage
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
    if (fig is not None) and (opti_params is not None):
        # Plot the fit
        linspace_voltages = numpy.linspace(first_voltage, last_voltage,
                                           num=1000)
        fit_count_rates = tool_belt.gaussian(linspace_voltages, *opti_params)
        # Add info to the axes
        # a: coefficient that defines the peak height
        # mu: mean, defines the center of the Gaussian
        # sigma: standard deviation, defines the width of the Gaussian
        # offset: constant y value to account for background
        text = 'a={:.3f}\n $\mu$={:.3f}\n ' \
            '$\sigma$={:.3f}\n offset={:.3f}'.format(*opti_params)
        update_figure(fig, axis_ind, linspace_voltages,
                      fit_count_rates, text)
    
    center = None
    if opti_params is not None:
        center = opti_params[1]
        
    return center


# %% User functions
    

def optimize_list(cxn, nv_sig_list, apd_indices):
    opti_nv_sig_list = []
    for nv_sig in nv_sig_list:
        opti_coords = main(cxn, nv_sig, apd_indices, set_to_opti_coords=False)
        opti_nv_sig_list.append([*opti_coords, *nv_sig[3: ]])
    
    for nv_sig in opti_nv_sig_list:
        print('[{:.3f}, {:.3f}, {:.1f}, {}, {}],'.format(*nv_sig))
    

# %% Main


def main(cxn, nv_sig, nd_filter, apd_indices, name='untitled', 
         set_to_opti_coords=True, save_data=False, plot_data=False):
    
    # Get the shared parameters from the registry
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    
    x_center, y_center, z_center = nv_sig[0:3]
    expected_count_rate = nv_sig[3]
    background_count_rate = nv_sig[4]
    
    opti_succeeded = False
    
    # %% Try to optimize
    
    for ind in range(2):
        
        # Create 3 plots in the figure, one for each axis
        fig = None
        if plot_data:
            fig = create_figure()
        
        # Optimize on each axis
        opti_coords = []
        voltages_by_axis = []
        counts_by_axis = []
        for axis_ind in range(3):
            ret_vals = optimize_on_axis(cxn, nv_sig, axis_ind, shared_params,
                                        apd_indices, fig)
            opti_coords.append(ret_vals[0])
            voltages_by_axis.append(ret_vals[1])
            counts_by_axis.append(ret_vals[2])
            
            
        # Verify that our optimization found a reasonable spot by checking
        # the count rate at the center against the expected count rate
        if (expected_count_rate is not None) and (None not in opti_coords):
            
            lower_threshold = expected_count_rate * 3/4
            upper_threshold = expected_count_rate * 5/4
            
            # check the counts
            opti_count_rate = stationary_count_lite(cxn, opti_coords,
                                                    shared_params, apd_indices)
            print('Counts from optimization: {}'.format(opti_count_rate))
            print('Expected counts: {}'.format(background_count_rate))
            
            # If the count rate close to what we expect, we succeeded!
            if lower_threshold <= opti_count_rate <= upper_threshold:
                print('Optimization succeeded!')
                opti_succeeded = True
            else:
                print('Count rate at optimized coordinates out of bounds. ' \
                      'Trying again.')
                
        # If the threshold is not set, we succeed based only on optimize       
        else:
            print('Optimization succeeded! (No expected count rate passed.)')
            opti_succeeded = True
        # Break out of the loop if optimization succeeded
        if opti_succeeded:
            break
    
    # %% Set to the optimized coordinates, or just tell the user what they are
            
    if set_to_opti_coords:
        if opti_succeeded:
            tool_belt.set_xyz(cxn, opti_coords)
        else:
            # Let the user know something went wrong
            print('Centers could not be located. Resetting to coordinates ' \
                  'about which we attempted to optimize.')
            tool_belt.set_xyz(cxn, nv_sig[0:3])
    else:
        if opti_succeeded:
            print('centers: ')
            print('{:.3f}, {:.3f}, {:.1f}'.format(*opti_coords))
            diff = numpy.array(opti_coords) - numpy.array(nv_sig[0:3])
            print('difference: ')
            print('{:.3f}, {:.3f}, {:.1f}'.format(*diff))
        else:
            print('Optimization failed.')
                               
    # %% Save the data

    # Don't bother saving the data if we're just using this to find the
    # optimized coordinates
    if save_data:

        timestamp = tool_belt.get_time_stamp()

        rawData = {'timestamp': timestamp,
                   'name': name,
                   'nv_sig': nv_sig,
                   'nv_sig-units': '[V, V, V, kcps, kcps]',
                   'nd_filter': nd_filter,
                   'num_steps': num_steps,
                   'readout': shared_params['continuous_readout_ns'],
                   'readout-units': 'ns',
                   'opti_coords': opti_coords,
                   'opti_coords-units': 'V',
                   'x_voltages': voltages_by_axis[0].tolist(),
                   'x_voltages-units': 'V',
                   'y_voltages': voltages_by_axis[1].tolist(),
                   'y_voltages-units': 'V',
                   'z_voltages': voltages_by_axis[2].tolist(),
                   'z_voltages-units': 'V',
                   'x_counts': counts_by_axis[0].tolist(),
                   'x_counts-units': 'number',
                   'y_counts': counts_by_axis[1].tolist(),
                   'y_counts-units': 'number',
                   'z_counts': counts_by_axis[2].tolist(),
                   'z_counts-units': 'number'}

        filePath = tool_belt.get_file_path(__file__, timestamp, name)
        tool_belt.save_raw_data(rawData, filePath)
        
        if fig is not None:
            tool_belt.save_figure(fig, filePath)
            
    # %% Return the optimized coordinates we found
    
    return opti_coords
