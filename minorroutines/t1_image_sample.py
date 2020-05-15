# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:18:44 2020

This program will use t1 measured at one spot to image a sample.

@author: agardilll
"""

import majorroutines.t1_double_quantum as t1_double_quantum
from majorroutines.image_sample import on_click_image
from majorroutines.image_sample import populate_img_array
import utils.tool_belt as tool_belt
from utils.tool_belt import States
import numpy
import labrad
import time
import matplotlib.pyplot as plt

# %%

def main(nv_sig, scan_range, num_steps, relaxation_time_point, apd_indices):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, scan_range, num_steps, 
                      relaxation_time_point, apd_indices)
        

def main_with_cxn(cxn, nv_sig, scan_range, num_steps, 
                  relaxation_time_point, apd_indices):
    
    tool_belt.reset_cfm(cxn)

    adj_coords = (numpy.array(nv_sig['coords']) + \
                  numpy.array(tool_belt.get_drift())).tolist()
    x_center, y_center, z_center = adj_coords

    total_num_samples = num_steps**2
    
    relaxation_time_range = [relaxation_time_point, relaxation_time_point]
    t1_num_steps = 1
    t1_num_reps = 10**3
    t1_num_runs = 100
    init_read_list = [States.ZERO, States.ZERO]

    # %% Initialize at the passed coordinates

    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])

    # %% Set up the galvo

    x_voltages, y_voltages = tool_belt.x_y_image_grid(x_center, y_center,
                                                       scan_range, scan_range,
                                                       num_steps)

    x_num_steps = len(x_voltages)
    x_low = x_voltages[0]
    x_high = x_voltages[x_num_steps-1]
    y_num_steps = len(y_voltages)
    y_low = y_voltages[0]
    y_high = y_voltages[y_num_steps-1]

    pixel_size = x_voltages[1] - x_voltages[0]
    
    # %% Set up our raw data objects

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = numpy.empty((x_num_steps, y_num_steps))
    img_array[:] = numpy.nan
    img_write_pos = []  
 
    # %% Create image
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
    title = 'Confocal T1 scan'
    fig = tool_belt.create_image_figure(img_array, img_extent,
                                        clickHandler=on_click_image,
                                        title = title)   
    
    for i in range(total_num_samples):
        tool_belt.set_xyz(cxn, [x_voltages[i], y_voltages[i], z_center])
        time.sleep(0.1)
        t1_signal = t1_double_quantum.main(nv_sig, apd_indices, relaxation_time_range,
                       t1_num_steps, t1_num_reps, t1_num_runs, init_read_list)
        populate_img_array(t1_signal[0], img_array, img_write_pos)
        tool_belt.update_image_figure(fig, img_array)
        
    # %% Clean up

    tool_belt.reset_cfm(cxn)

    # Return to center
    cxn.galvo.write(x_center, y_center)
    
    # %% Save the data

    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'scan_range': scan_range,
               'scan_range-units': 'V',
               'num_steps': num_steps,
               't1_num_steps': t1_num_steps,
               't1_num_reps': t1_num_reps,
               't1_num_runs': t1_num_runs,
               'relaxation_time_point': relaxation_time_point,
               'relaxation_time_point-units': 'ns',
               'x_voltages': x_voltages.tolist(),
               'x_voltages-units': 'V',
               'y_voltages': y_voltages.tolist(),
               'y_voltages-units': 'V',
               'img_array': img_array.astype(int).tolist(),
               'img_array-units': 'counts'}

    if save_data:

        filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
        tool_belt.save_raw_data(rawData, filePath)

        if plot_data:

            tool_belt.save_figure(fig, filePath)
    
    