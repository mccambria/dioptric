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
#import majorroutines.image_sample as image_sample
import copy
import scipy.stats as stats
def build_voltages_from_list(start_coords_drift, coords_list_drift):

    # calculate the x values we want to step thru
    start_x_value = start_coords_drift[0]
    start_y_value = start_coords_drift[1]
    
    num_samples = len(coords_list_drift)
    
    # we want this list to have the pattern [[readout], [target], [readout], [readout], 
    #                                                   [target], [readout], [readout],...]
    # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]
    
    # now create a list of all the coords we want to feed to the galvo
    for i in range(num_samples):
        x_points.append(coords_list_drift[i][0])
        x_points.append(start_x_value)
        x_points.append(start_x_value) 
        
        y_points.append(coords_list_drift[i][1])
        y_points.append(start_y_value)
        y_points.append(start_y_value) 
        
    return x_points, y_points

def build_voltages_start_end(start_coords_drift, end_coords_drift, num_steps):

    # calculate the x values we want to step thru
    start_x_value = start_coords_drift[0]
    end_x_value = end_coords_drift[0]
    target_x_values = numpy.linspace(start_x_value, end_x_value, num_steps)
    
    # calculate the y values we want to step thru
    start_y_value = start_coords_drift[1]
    end_y_value = end_coords_drift[1]
    
#    ## Change this code to be more general later:##
#    mid_y_value = start_y_value + 0.3
#    
#    dense_target_y_values = numpy.linspace(start_y_value, mid_y_value, 101)
#    sparse_target_y_values = numpy.linspace(mid_y_value+0.06, end_y_value, 20)
#    target_y_values = numpy.concatenate((dense_target_y_values, sparse_target_y_values))
    
    
    target_y_values = numpy.linspace(start_y_value, end_y_value, num_steps)
    
    # make a list of the coords that we'll send to the glavo
    # we want this list to have the pattern [[readout], [target], [readout], [readout], 
    #                                                   [target], [readout], [readout],...]
    # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]
    
    # now create a list of all the coords we want to feed to the galvo
    for x in target_x_values:
        x_points.append(x)
        x_points.append(start_x_value)
        x_points.append(start_x_value) 
        

    # now create a list of all the coords we want to feed to the galvo
    for y in target_y_values:
        y_points.append(y)
        y_points.append(start_y_value)
        y_points.append(start_y_value)   
        
    return x_points, y_points
       # %%
def target_list(nv_sig, start_coords, coords_list, num_runs, init_color, pulse_color):
    with labrad.connect() as cxn:
        ret_vals = target_list_with_cxn(cxn, nv_sig, start_coords, coords_list, num_runs, init_color, pulse_color)
    
    readout_counts_avg, readout_counts_ste, target_counts_avg, target_counts_ste = ret_vals
                        
    return readout_counts_avg, readout_counts_ste, target_counts_avg, target_counts_ste
        
def target_list_with_cxn(cxn, nv_sig, start_coords, coords_list, num_runs, init_color, pulse_color):
    tool_belt.reset_cfm(cxn)
        
    # Define paramters
    apd_indices = [0]
    readout_color = 589
    
    num_samples = len(coords_list)
    
    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = 1*shared_params['large_angle_galvo_delay']
    
    # copy the start coords onto the nv_sig
    start_nv_sig = copy.deepcopy(nv_sig)
    start_nv_sig['coords'] = start_coords

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
    readout_counts_array = []
    target_counts_array = []

    startFunctionTime = time.time()
    
    # define the sequence paramters
    file_name = 'isolate_nv_charge_dynamics_moving_target.py'
    seq_args = [ initialization_time, pulse_time, readout_pulse_time, 
        laser_515_delay, aom_589_delay, laser_638_delay, 
        galvo_delay, am_589_power, apd_indices[0], init_color, pulse_color, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    
    period = ret_vals[0]
    period_s = period/10**9
    period_s_total = (period_s*num_samples + 1)*num_runs
    print('Expected total run time: {:.0f} s'.format(period_s_total))
#    return

    # Optimize at the start of the routine
    opti_coords = optimize.main_with_cxn(cxn, start_nv_sig, apd_indices, 532, disable=False)
    opti_coords_list.append(opti_coords)
    
    start_timestamp = tool_belt.get_time_stamp()
    
    # record the time starting at the beginning of the runs
    run_start_time = time.time()
    
    for r in range(num_runs):  
        print( 'run {}'.format(r))
        #optimize every 5 min or so
        # So first check the time. If the time that has passed since the last
        # optimize is longer that 5 min, optimize again
        current_time = time.time()
        if current_time - run_start_time >= 5*60:
            opti_coords = optimize.main_with_cxn(cxn, start_nv_sig, apd_indices, 532, disable=False)
            opti_coords_list.append(opti_coords)
            run_start_time = current_time
        
        drift = numpy.array(tool_belt.get_drift())
        
        # get the readout coords with drift
        start_coords_drift = start_coords + drift
        coords_list_drift = numpy.array(coords_list) + drift
#        end_coords_drift = end_coords + drift
                    
        # start on the readout NV
        tool_belt.set_xyz(cxn, start_coords_drift)
        
               
        # load the sequence
        ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
        
        # Load the galvo
        x_voltages, y_voltages = build_voltages_from_list(start_coords_drift, coords_list_drift)
#        x_voltages.insert(0,start_coords_drift[0])
        cxn.galvo.load_arb_points_scan(x_voltages, y_voltages, int(period))
        
        #  Set up the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
         
        
        cxn.pulse_streamer.stream_start(num_samples)
        total_num_samples = 3*num_samples
    
        tool_belt.init_safe_stop()
        
        new_samples = cxn.apd_tagger.read_counter_simple(total_num_samples)
        # The last of the triplet of readout windows is the counts we are interested in
        readout_counts = new_samples[2::3]
        readout_counts = [int(el) for el in readout_counts]
        target_counts = new_samples[1::3]
        target_counts = [int(el) for el in target_counts]
        
        readout_counts_array.append(readout_counts)
        target_counts_array.append(target_counts)
        
        cxn.apd_tagger.stop_tag_stream()
        
        # save incrimentally 
        raw_data = {'start_timestamp': start_timestamp,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_colo': readout_color,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'num_runs':num_runs,
            'opti_coords_list': opti_coords_list,
            'readout_counts_array': readout_counts_array,
            'readout_counts_array-units': 'counts',
            'target_counts_array': target_counts_array,
            'target_counts_array-units': 'counts',
            }
                
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
    
    # calculate the radial distances from the readout NV to the target points
    # the target coords are the first of triplet of coords we pass.
    x_voltages.pop(0)
    y_voltages.pop(0)
    x_target_coords = numpy.array(x_voltages[0::3])
    y_target_coords = numpy.array(y_voltages[0::3])
    x_diffs = (x_target_coords - start_coords_drift[0])
    y_diffs = (y_target_coords- start_coords_drift[1])
    rad_dist = numpy.sqrt(x_diffs**2 + y_diffs**2)
    
    # Statistics
    readout_counts_avg = numpy.average(readout_counts_array, axis=0)
    readout_counts_ste = stats.sem(readout_counts_array, axis=0)
    target_counts_avg = numpy.average(target_counts_array, axis=0)
    target_counts_ste = stats.sem(target_counts_array, axis=0)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])
    
#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    ax.plot(rad_dist*35,readout_counts_avg, label = 'counts on readout after each pulse on target spot')
##    ax.plot(rad_dist*35,target_counts_avg, label = 'counts on target spot' )
#    ax.set_xlabel('Distance from readout NV (um)')
#    ax.set_ylabel('Average counts')
#    ax.set_title('Stationary readout NV, moving target ({} init, {} ms {} pulse)'.\
#                                    format(init_color, pulse_time/10**6, pulse_color))
#    ax.legend()
#    
#    fig_temp, ax = plt.subplots(1, 1, figsize=(10, 10))
#    ax.plot(rad_dist*35,readout_counts_avg, label = 'counts on readout after each pulse on target spot')
#    ax.plot(rad_dist*35,target_counts_avg, label = 'counts on target spot' )
#    ax.set_xlabel('Distance from readout NV (um)')
#    ax.set_ylabel('Average counts')
#    ax.set_title('Stationary readout NV, moving target ({} init, {} ms {} pulse)'.\
#                                    format(init_color, pulse_time/10**6, pulse_color))
#    ax.legend()
    
    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_colo': readout_color,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'green_optical_power_pd': green_optical_power_pd,
            'green_optical_power_pd-units': 'V',
            'green_optical_power_mW': green_optical_power_mW,
            'green_optical_power_mW-units': 'mW',
            'red_optical_power_pd': red_optical_power_pd,
            'red_optical_power_pd-units': 'V',
            'red_optical_power_mW': red_optical_power_mW,
            'red_optical_power_mW-units': 'mW',
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'num_runs':num_runs,
            'opti_coords_list': opti_coords_list,
            'rad_dist': rad_dist.tolist(),
            'rad_dist-units': 'V',
            'readout_counts_array': readout_counts_array,
            'readout_counts_array-units': 'counts',
            'target_counts_array': target_counts_array,
            'target_counts_array-units': 'counts',

            'readout_counts_avg': readout_counts_avg.tolist(),
            'readout_counts_avg-units': 'counts',
            'target_counts_avg': target_counts_avg.tolist(),
            'target_counts_avg-units': 'counts',

            'readout_counts_ste': readout_counts_ste.tolist(),
            'readout_counts_ste-units': 'counts',
            'target_counts_ste': target_counts_ste.tolist(),
            'target_counts_ste-units': 'counts',
            }
    
    file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
#    tool_belt.save_figure(fig, file_path)
#    tool_belt.save_figure(fig_temp, file_path + '-path_compare')
    
    return readout_counts_avg, readout_counts_ste, target_counts_avg, \
                        target_counts_ste
     
# %%
def moving_target(nv_sig, start_coords, end_point, num_steps, num_runs, init_color, pulse_color, point_list = None):
    with labrad.connect() as cxn:
        moving_target_with_cxn(cxn, nv_sig,start_coords,  end_point, num_steps,num_runs,  init_color, pulse_color, point_list)
    return
        
def moving_target_with_cxn(cxn, nv_sig,start_coords,  end_coords, num_steps,num_runs, init_color, pulse_color, point_list = None):
    tool_belt.reset_cfm(cxn)
        
    # Define paramters
    apd_indices = [0]
    readout_color = 589
    
    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = 1*shared_params['large_angle_galvo_delay']
    
    # copy the start coords onto the nv_sig
    start_nv_sig = copy.deepcopy(nv_sig)
    start_nv_sig['coords'] = start_coords

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
    readout_counts_array = []
    target_counts_array = []

    startFunctionTime = time.time()
    
    # define the sequence paramters
    file_name = 'isolate_nv_charge_dynamics_moving_target.py'
    seq_args = [ initialization_time, pulse_time, readout_pulse_time, 
        laser_515_delay, aom_589_delay, laser_638_delay, 
        galvo_delay, am_589_power, apd_indices[0], init_color, pulse_color, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    
    period = ret_vals[0]
    period_s = period/10**9
    period_s_total = (period_s*num_steps + 5)*num_runs
    print('Expected total run time: {:.0f} s'.format(period_s_total))
#    return

    # Optimize at the start of the routine
    opti_coords = optimize.main_with_cxn(cxn, start_nv_sig, apd_indices, 532, disable=False)
    opti_coords_list.append(opti_coords)
    
    start_timestamp = tool_belt.get_time_stamp()
    
    # record the time starting at the beginning of the runs
    run_start_time = time.time()
    
    for r in range(num_runs):  
        print( 'run {}'.format(r))
        #optimize every 5 min or so
        # So first check the time. If the time that has passed since the last
        # optimize is longer that 5 min, optimize again
        current_time = time.time()
        if current_time - run_start_time >= 5*60:
            opti_coords = optimize.main_with_cxn(cxn, start_nv_sig, apd_indices, 532, disable=False)
            opti_coords_list.append(opti_coords)
            run_start_time = current_time
            
        drift = numpy.array(tool_belt.get_drift())
        
        # get the readout coords with drift
        start_coords_drift = start_coords + drift
        end_coords_drift = end_coords + drift
                    
        # start on the readout NV
        tool_belt.set_xyz(cxn, start_coords_drift)
        
               
        # load the sequence
        ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
        
        # Load the galvo
        x_voltages, y_voltages = build_voltages_start_end(start_coords_drift, end_coords_drift, num_steps)
#        x_voltages.insert(0,start_coords_drift[0])
        cxn.galvo.load_arb_points_scan(x_voltages, y_voltages, int(period))
        
        #  Set up the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
         
        
        cxn.pulse_streamer.stream_start(num_steps)
        total_num_steps = 3*num_steps
    
        tool_belt.init_safe_stop()
        
        new_samples = cxn.apd_tagger.read_counter_simple(total_num_steps)
        # The last of the triplet of readout windows is the counts we are interested in
        readout_counts = new_samples[2::3]
        readout_counts = [int(el) for el in readout_counts]
        target_counts = new_samples[1::3]
        target_counts = [int(el) for el in target_counts]
        
        readout_counts_array.append(readout_counts)
        target_counts_array.append(target_counts)
        
        cxn.apd_tagger.stop_tag_stream()
        
        # save incrimentally 
        raw_data = {'start_timestamp': start_timestamp,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_colo': readout_color,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'num_runs':num_runs,
            'opti_coords_list': opti_coords_list,
            'readout_counts_array': readout_counts_array,
            'readout_counts_array-units': 'counts',
            'target_counts_array': target_counts_array,
            'target_counts_array-units': 'counts',
            }
                
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
    
    # calculate the radial distances from the readout NV to the target points
    # the target coords are the first of triplet of coords we pass.
    x_voltages.pop(0)
    y_voltages.pop(0)
    x_target_coords = numpy.array(x_voltages[0::3])
    y_target_coords = numpy.array(y_voltages[0::3])
    x_diffs = (x_target_coords - start_coords_drift[0])
    y_diffs = (y_target_coords- start_coords_drift[1])
    rad_dist = numpy.sqrt(x_diffs**2 + y_diffs**2)
    
    # Statistics
    readout_counts_avg = numpy.average(readout_counts_array, axis=0)
    readout_counts_ste = stats.sem(readout_counts_array, axis=0)
    target_counts_avg = numpy.average(target_counts_array, axis=0)
    target_counts_ste = stats.sem(target_counts_array, axis=0)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(rad_dist*35,readout_counts_avg, label = 'counts on readout after each pulse on target spot')
#    ax.plot(rad_dist*35,target_counts_avg, label = 'counts on target spot' )
    ax.set_xlabel('Distance from readout NV (um)')
    ax.set_ylabel('Average counts')
    ax.set_title('Stationary readout NV, moving target ({} init, {} ms {} pulse)'.\
                                    format(init_color, pulse_time/10**6, pulse_color))
    ax.legend()
    
    fig_temp, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(rad_dist*35,readout_counts_avg, label = 'counts on readout after each pulse on target spot')
    ax.plot(rad_dist*35,target_counts_avg, label = 'counts on target spot' )
    ax.set_xlabel('Distance from readout NV (um)')
    ax.set_ylabel('Average counts')
    ax.set_title('Stationary readout NV, moving target ({} init, {} ms {} pulse)'.\
                                    format(init_color, pulse_time/10**6, pulse_color))
    ax.legend()
    
    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_colo': readout_color,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'green_optical_power_pd': green_optical_power_pd,
            'green_optical_power_pd-units': 'V',
            'green_optical_power_mW': green_optical_power_mW,
            'green_optical_power_mW-units': 'mW',
            'red_optical_power_pd': red_optical_power_pd,
            'red_optical_power_pd-units': 'V',
            'red_optical_power_mW': red_optical_power_mW,
            'red_optical_power_mW-units': 'mW',
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'num_runs':num_runs,
            'opti_coords_list': opti_coords_list,
            'rad_dist': rad_dist.tolist(),
            'rad_dist-units': 'V',
            'readout_counts_array': readout_counts_array,
            'readout_counts_array-units': 'counts',
            'target_counts_array': target_counts_array,
            'target_counts_array-units': 'counts',

            'readout_counts_avg': readout_counts_avg.tolist(),
            'readout_counts_avg-units': 'counts',
            'target_counts_avg': target_counts_avg.tolist(),
            'target_counts_avg-units': 'counts',

            'readout_counts_ste': readout_counts_ste.tolist(),
            'readout_counts_ste-units': 'counts',
            'target_counts_ste': target_counts_ste.tolist(),
            'target_counts_ste-units': 'counts',
            }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_figure(fig_temp, file_path + '-path_compare')
    
    return

def plot_times_on_off_nv(nv_sig, readout_coords,  target_nv_coords, dark_coords, num_runs, init_color, pulse_color):
    NV_avg_list = []
    NV_ste_list = []
    dark_avg_list = []
    dark_ste_list = []
    
    time_list = numpy.array([10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9])
#    time_list = numpy.array([10**3])
    coords_list = [target_nv_coords, dark_coords]
    # run various times#    
    for t in time_list:    
        nv_sig_copy = copy.deepcopy(nv_sig)
        if pulse_color == 532:
            nv_sig_copy['pulsed_reionization_dur'] = int(t)
        if pulse_color == 638:
            nv_sig_copy['pulsed_ionization_dur'] = int(t) 
            
        ret_vals = target_list(nv_sig_copy, readout_coords, coords_list, num_runs, init_color, pulse_color)
        readout_counts_avg, readout_counts_ste, target_counts_avg, target_counts_ste = ret_vals
        print(readout_counts_avg)
                        
        # sort out the counts after pulse on NV vs off NV
        NV_avg_list.append(readout_counts_avg[0])
        NV_ste_list.append(readout_counts_ste[0])
        dark_avg_list.append(readout_counts_avg[1])
        dark_ste_list.append(readout_counts_ste[1])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.errorbar(time_list/10**6,NV_avg_list, yerr = NV_ste_list, fmt = 'o', label = 'Readout counts after NV target')
    ax.errorbar(time_list/10**6,dark_avg_list, yerr = dark_ste_list,fmt = 'o', label = 'Readout counts after off NV target')
    ax.set_xlabel('Pulse time (ms)')
    ax.set_ylabel('Average counts')
    ax.set_title('Stationary readout NV, moving target on/off NV ({} init, {} pulse)'.\
                                    format(init_color, pulse_color))
    ax.legend()
    ax.set_xscale('log')
    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'init_color': init_color,
                'pulse_color': pulse_color,
            'target_nv_coords': target_nv_coords,
            'dark_coords': dark_coords,
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'num_runs':num_runs,
            'time_list': time_list.tolist(),
            'time_list-units': 'ns',
            'NV_avg_list': NV_avg_list,
            'NV_avg_list-units': 'counts',
            'dark_avg_list': dark_avg_list,
            'dark_avg_list-units': 'counts',

            'NV_ste_list': NV_ste_list,
            'NV_ste_list-units': 'counts',
            'dark_ste_list': dark_ste_list,
            'dark_ste_list-units': 'counts',

            }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)

    return

# %% Run the files

if __name__ == '__main__':
    sample_name= 'johnson'
    nv18_2020_11_10 = { 'coords':None, 
            'name': '{}-nv18_2020_11_10'.format(sample_name),
            'expected_count_rate': 30, 'nd_filter': 'nd_1.0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 20*10**6, 'am_589_power': 0.7, 
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120, 
            'pulsed_reionization_dur': 10**7,
            'cobalt_532_power':20, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}  


    
    start_coords = [0.179, 0.247, 5.26]
    dy = 0.2
    end_coords=[0.179, 0.247 + dy, 5.26]
    num_steps = 101
    init_color = 638
    pulse_color = 532
    num_runs = 200
    
    target_nv_coords= [0.224 - 0.041, 0.285 - 0.009, 5.26]
    dark_coords = [0.193 - 0.041, 0.246 - 0.009, 5.26]
 
#    moving_target(nv18_2020_11_10,start_coords,  end_coords, num_steps, num_runs, init_color, pulse_color)    
       
    plot_times_on_off_nv(nv18_2020_11_10, start_coords,  target_nv_coords, dark_coords, num_runs, init_color, pulse_color)
    
                    
         
    
#    for t in [10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]:    
#        init_color = 638
#        pulse_color = 532
#        nv_sig = copy.deepcopy(nv18_2020_11_10)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t    
#        moving_target(nv_sig,start_coords,  end_coords, num_steps, num_runs, init_color, pulse_color)
#        
#    for t in [10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]:    
#        init_color = 638
#        pulse_color = 638
#        nv_sig = copy.deepcopy(nv18_2020_11_10)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t    
#        moving_target(nv_sig,start_coords,  end_coords, num_steps, num_runs, init_color, pulse_color)
#        
#    for t in [10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]:    
#        init_color = 532
#        pulse_color = 532
#        nv_sig = copy.deepcopy(nv18_2020_11_10)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t    
#        moving_target(nv_sig,start_coords,  end_coords, num_steps, num_runs, init_color, pulse_color)
#        
#    for t in [10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]:    
#        init_color = 532
#        pulse_color = 638
#        nv_sig = copy.deepcopy(nv18_2020_11_10)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t    
#        moving_target(nv_sig,start_coords,  end_coords, num_steps, num_runs, init_color, pulse_color)

# %%
#    file = '2020_11_23-16_57_06-johnson-nv18_2020_11_10'
#    sub_folder = 'isolate_nv_charge_dynamics_moving_target/branch_Spin_to_charge/2020_11'
#    
#    data = tool_belt.get_raw_data(sub_folder, file)
#    
#    rad_dist = data['rad_dist']
#    readout_counts_avg = data['readout_counts_avg']
#    target_counts_avg = data['target_counts_avg']
#    init_color = data['init_color']
#    pulse_color = data['pulse_color']
#    nv_sig = data['nv_sig']
#    
#    if init_color == 532:
#        initialization_time = nv_sig['pulsed_reionization_dur']
#    elif init_color == 638:
#        initialization_time = nv_sig['pulsed_ionization_dur']
#    if pulse_color == 532:
#        pulse_time = nv_sig['pulsed_reionization_dur']
#    elif pulse_color == 638:
#        pulse_time = nv_sig['pulsed_ionization_dur']
#    
#    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#    ax.plot(numpy.array(rad_dist)*35,readout_counts_avg, label = 'counts on readout after each pulse on target spot')
#    ax.plot(numpy.array(rad_dist)*35,numpy.array(target_counts_avg)/100, label = 'counts on target spot')
#    ax.set_xlabel('Distance from readout NV (um)')
#    ax.set_ylabel('Average counts')
#    ax.set_title('Stationary readout NV, moving target ({} init, {} ms {} pulse)'.\
#                                    format(init_color, pulse_time/10**6, pulse_color))
#    ax.legend()
#    ax.set_xlim([-0.5,7])
    
    
    