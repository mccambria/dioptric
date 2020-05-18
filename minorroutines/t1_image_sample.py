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
    t1_num_reps = 2*10**2
    t1_num_runs = 1000
    init_read_list = [States.ZERO, States.ZERO]

    # %% Initialize at the passed coordinates

    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])

    # %% Set up the galvo

    x_voltages, y_voltages = tool_belt.x_y_image_grid(x_center, y_center,
                                                       scan_range, scan_range,
                                                       num_steps)
#    print(x_voltages)
#    print(y_voltages)

    x_num_steps = num_steps
    x_low = x_voltages[0]
    x_high = x_voltages[x_num_steps-1]
    y_num_steps = num_steps
    y_low = y_voltages[0]
    y_high = y_voltages[-1]

    pixel_size = x_voltages[1] - x_voltages[0]
    
    # %% Set up our raw data objects

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    norm_img_array = numpy.empty((x_num_steps, y_num_steps))
    norm_img_array[:] = numpy.nan
    sig_img_array = numpy.copy(norm_img_array)
    ref_img_array = numpy.copy(norm_img_array)
#    print(img_array)
    norm_img_write_pos = []
    sig_img_write_pos = []
    ref_img_write_pos = []
 
    # %% Create image
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
    title = 'Confocal T1 scan'
    norm_fig = tool_belt.create_image_figure(norm_img_array, img_extent,
                                        clickHandler=on_click_image,
                                        title = title,
                                        color_bar_label = 'T1 normalized signal after {} ms'.format(relaxation_time_point/10**6))   

    sig_fig = tool_belt.create_image_figure(sig_img_array, img_extent,
                                        clickHandler=on_click_image,
                                        title = title,
                                        color_bar_label = 'T1 signal after {} ms'.format(relaxation_time_point/10**6))   

    ref_fig = tool_belt.create_image_figure(ref_img_array, img_extent,
                                        clickHandler=on_click_image,
                                        title = title,
                                        color_bar_label = 'T1 reference after {} ms'.format(relaxation_time_point/10**6))   

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()    
    for i in range(total_num_samples):
        if tool_belt.safe_stop():
            break
#        print(img_write_pos)
        tool_belt.set_xyz(cxn, [x_voltages[i], y_voltages[i], z_center])
        time.sleep(0.1)
        sig, ref, norm = t1_double_quantum.main(nv_sig, apd_indices, relaxation_time_range,
                       t1_num_steps, t1_num_reps, t1_num_runs, init_read_list,
                       plot_data = False, save_data = True)
#        print(t1_signal)
        populate_img_array(norm, norm_img_array, norm_img_write_pos)
        tool_belt.update_image_figure(norm_fig, norm_img_array)
        
        populate_img_array(sig, sig_img_array, sig_img_write_pos)
        tool_belt.update_image_figure(sig_fig, sig_img_array)
        
        populate_img_array(ref, ref_img_array, ref_img_write_pos)
        tool_belt.update_image_figure(ref_fig, ref_img_array)
        
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
               'norm_img_array': norm_img_array.astype(int).tolist(),
               'norm_img_array-units': 'counts',
               'sig_img_array': sig_img_array.astype(int).tolist(),
               'sig_img_array-units': 'counts',
               'ref_img_array': ref_img_array.astype(int).tolist(),
               'ref_img_array-units': 'counts',
               }

 

    filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(rawData, filePath)

 

    tool_belt.save_figure(norm_fig, filePath + '-norm')
    tool_belt.save_figure(sig_fig, filePath + '-sig')
    tool_belt.save_figure(ref_fig, filePath + '-ref')
    
    