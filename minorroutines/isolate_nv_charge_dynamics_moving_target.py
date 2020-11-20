# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:33:57 2020

A routine to take one NV to readout the charge state, after pulsing a laser
at a distance from this readout NV.

@author: agardill
"""

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
import labrad
import majorroutines.image_sample as image_sample
import copy
import scipy.stats as stats


def moving_target(nv_sig, x_range, num_steps, num_runs, init_color, pulse_color):
    with labrad.connect() as cxn:
        moving_target_with_cxn(cxn, nv_sig, x_range, num_steps,num_runs,  init_color, pulse_color)
    return
        
def moving_target_with_cxn(cxn, nv_sig, x_range, num_steps,num_runs, init_color, pulse_color):
    tool_belt.reset_cfm(cxn)
        
    # Define paramters
    apd_indices = [0]
    readout_color = 532
    
    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    aom_589_delay = shared_params['589_aom_delay']
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']

    am_589_power = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
    if init_color == 532:
        initialization_time = nv_sig['pulsed_reionization_dur']
    elif init_color == 638:
        initialization_time = nv_sig['pulsed_ionization_dur']
    if pulse_color == 532:
        pulse_time = nv_sig['pulsed_reionization_dur']
    elif pulse_color == 638:
        pulse_time = nv_sig['pulsed_ionization_dur']
    
    opti_coords_list = []
    
    drift = numpy.array(tool_belt.get_drift())
    # make a list of the coords that we'll send to the glavo
    # we want this list to have the pattern [[target], [readout], [readout], ...]
    coords_to_step_thru = []
    
    # get the readout coords with drift
    readout_coords = numpy.array(nv_sig['coords'])
    readout_coords_drift = readout_coords + drift
    
    # calculate the x values we want to step thru
    start_x_value = readout_coords_drift[0]
    end_x_value = start_x_value + x_range
    target_x_values = numpy.linspace(start_x_value, end_x_value, num_steps)
    
    # now create a list of all the coords we want to feed to the galvo
    for x in target_x_values:
        target_coords_drift  = [x, readout_coords_drift[1], readout_coords_drift[2]]
        coords_to_step_thru.append(target_coords_drift)
        coords_to_step_thru.append(readout_coords_drift)
        coords_to_step_thru.append(readout_coords_drift)       
        # might need to add one last coord for it to work, we'll see
        
    
    # define the sequence paramters
    file_name = 'isolate_nv_charge_dynamics_moving_target.py'
    seq_args = [ initialization_time, pulse_time, readout_pulse_time, 
        laser_515_delay, aom_589_delay, laser_638_delay, 
        galvo_delay, am_589_power, apd_indices[0], init_color, pulse_color, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name,
                                              seq_args_string)
    period = ret_vals[0]
    
#    # Just try to do the simple pulse..    
#    file_name = 'simple_readout.py'
#    seq_args = [ laser_515_delay, readout_pulse_time, 0,  apd_indices[0], 532]
#    seq_args_string = tool_belt.encode_seq_args(seq_args)
#    ret_vals = cxn.pulse_streamer.stream_load(file_name,
#                                              seq_args_string)
#    
#    period = ret_vals[0]
    
    
    for r in range(num_runs):    
        # optimize on the readout NV
#        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)
#        opti_coords_list.append(opti_coords)
            
        # start on the readout NV
        tool_belt.set_xyz(cxn, readout_coords_drift)
               
        
        #  Set up the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
         

        # Clear the tagger buffer of any excess counts
#        cxn.apd_tagger.clear_buffer()
        
#        # stream the sequence, the number of reps are euqal to the number of steps for the target
#        cxn.pulse_streamer.stream_immediate(file_name, num_steps, seq_args_string)
#        
#        
#        # Taken from image_sample
##        cxn.pulse_streamer.stream_start(num_steps)
#        new_counts = cxn.apd_tagger.read_counter_simple()
#        print(new_counts)
        
        cxn.pulse_streamer.stream_start(num_steps)
        total_num_steps = 3*num_steps

        timeout_duration = ((period*(10**-9)) * num_steps) + 10
        timeout_inst = time.time() + timeout_duration
    
        num_read_so_far = 0
    
        tool_belt.init_safe_stop()
    
        while num_read_so_far < total_num_steps:
    
            if time.time() > timeout_inst:
                break
    
            if tool_belt.safe_stop():
                break
    
            # Read the samples and update the image
            new_samples = cxn.apd_tagger.read_counter_separate_gates(3)
            num_new_samples = len(new_samples)
            if num_new_samples > 0:
                print(new_samples)
                num_read_so_far += num_new_samples
        
        
        

#        timeout_duration = ((period*(10**-9)) * num_steps) + 10
#        timeout_inst = time.time() + timeout_duration
#
#        num_read_so_far = 0
#
#        tool_belt.init_safe_stop()
#        
#        # there are 3 clock pulses in the sequence, so we need to be prepared to
#        # read 3x the num of steps
#        total_num_steps = 3*num_steps
#
#        while num_read_so_far < total_num_steps:
#
#            if time.time() > timeout_inst:
#                break
#    
#            if tool_belt.safe_stop():
#                break
#    
#            # Read the samples and update the image
#            new_samples = cxn.apd_tagger.read_counter_separate_gates(1)
#            
#            # I imagine the counts will look somewhat like this: [[1], [0], [20] ...]
#            # because they will be broken up by each clock pulse
#            # All we need to do is keep every 3rd element
#            
#            
#            print(new_samples)
##            num_new_samples = len(new_samples)
##            if num_new_samples > 0:           
##                populate_img_array(new_samples, img_array, img_write_pos)
##                # This is a horribly inefficient way of getting kcps, but it
##                # is easy and readable and probably fine up to some resolution
##                if plot_data:
##    #                img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
##                    tool_belt.update_image_figure(fig, img_array)
##                num_read_so_far += num_new_samples
        
   

    
    return

# %% Run the files

if __name__ == '__main__':
    sample_name= 'johnson'
    nv_sig  = { 'coords': [0.047, 0.030, 5.22],
                'name': '{}-nv1_2020_11_10'.format(sample_name),
                'expected_count_rate': 60, 'nd_filter': 'nd_0',
                'pulsed_SCC_readout_dur': 4*10**6, 'am_589_power': 0.2,
                'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120,
                'pulsed_reionization_dur': 10**5, 'cobalt_532_power':20,
                'magnet_angle': 0}
    
    num_steps = 3
    init_color = 638
    pulse_color = 532
    num_runs = 1
    x_range = 0.05
    
    
    moving_target(nv_sig, x_range, num_steps, num_runs, init_color, pulse_color)
    
    