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
from random import shuffle
import majorroutines.image_sample as image_sample
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

def build_voltages_main(start_coords_drift, target_x_values, target_y_values):
    ''' This function does the basic building of the voltages we need in 
    this program, in the form of 
    [[readout], [target], [readout], [readout], 
                [target], [readout], [readout],...]
    '''
    
    start_x_value = start_coords_drift[0]
    start_y_value = start_coords_drift[1]
    
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

def build_voltages_image(start_coords, img_range, num_steps):
    x_center = start_coords[0]
    y_center = start_coords[1]

    x_num_steps = num_steps
    y_num_steps = num_steps

    # Force the scan to have square pixels by only applying num_steps
    # to the shorter axis
    half_x_range = img_range / 2
    half_y_range = img_range / 2

    x_low = x_center - half_x_range
    x_high = x_center + half_x_range
    y_low = y_center - half_y_range
    y_high = y_center + half_y_range

    # Apply scale and offset to get the voltages we'll apply to the galvo
    # Note that the polar/azimuthal angles, not the actual x/y positions
    # are linear in these voltages. For a small range, however, we don't
    # really care.
    x_voltages_1d = numpy.linspace(x_low, x_high, num_steps)
    y_voltages_1d = numpy.linspace(y_low, y_high, num_steps)

    # Winding cartesian product
    # The x values are repeated and the y values are mirrored and tiled
    # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

    # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
    x_inter = numpy.concatenate((x_voltages_1d,
                                 numpy.flipud(x_voltages_1d)))
    # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
    if y_num_steps % 2 == 0:  # Even x size
        target_x_values = numpy.tile(x_inter, int(y_num_steps/2))
    else:  # Odd x size
        target_x_values = numpy.tile(x_inter, int(numpy.floor(y_num_steps/2)))
        target_x_values = numpy.concatenate((target_x_values, x_voltages_1d))

    # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
    target_y_values = numpy.repeat(y_voltages_1d, x_num_steps)
        
    return target_x_values, target_y_values, x_voltages_1d, y_voltages_1d

def populate_img_array(valsToAdd, imgArray, run_num):
    """
    We scan the sample in a winding pattern. This function takes a chunk
    of the 1D list returned by this process and places each value appropriately
    in the 2D image array. This allows for real time imaging of the sample's
    fluorescence.

    Note that this function could probably be much faster. At least in this
    context, we don't care if it's fast. The implementation below was
    written for simplicity.

    Params:
        valsToAdd: numpy.ndarray
            The increment of raw data to add to the image array
        imgArray: numpy.ndarray
            The xDim x yDim array of fluorescence counts
        writePos: tuple(int)
            The last x, y write position on the image array. [] will default
            to the bottom right corner. Third index is the run number
    """
    yDim = imgArray.shape[0]
    xDim = imgArray.shape[1]
    
    # Start with the write position at the start
    writePos = [xDim, yDim - 1, run_num]

    xPos = writePos[0]
    yPos = writePos[1]

    # Figure out what direction we're heading
    headingLeft = ((yDim - 1 - yPos) % 2 == 0)

    for val in valsToAdd:
        if headingLeft:
            # Determine if we're at the left x edge
            if (xPos == 0):
                yPos = yPos - 1
                imgArray[yPos, xPos, run_num] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos - 1
                imgArray[yPos, xPos, run_num] = val
        else:
            # Determine if we're at the right x edge
            if (xPos == xDim - 1):
                yPos = yPos - 1
                imgArray[yPos, xPos, run_num] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos + 1
                imgArray[yPos, xPos, run_num] = val
    return
    
       # %%
def target_list(nv_sig, start_coords, coords_list, num_runs, init_color, pulse_color):
    with labrad.connect() as cxn:
        ret_vals = target_list_with_cxn(cxn, nv_sig, start_coords, coords_list, num_runs, init_color, pulse_color)
    
    readout_counts_avg, readout_counts_ste, target_counts_avg, target_counts_ste, readout_counts_array, target_counts_array, rad_dist= ret_vals
                        
    return readout_counts_avg, readout_counts_ste, target_counts_avg, target_counts_ste, readout_counts_array, target_counts_array, rad_dist
        
def target_list_with_cxn(cxn, nv_sig, start_coords, coords_list, num_runs, init_color, pulse_color):
    tool_belt.reset_cfm(cxn)
        
    # Define paramters
    apd_indices = [0]
    readout_color = 589
#    print(coords_list)
    
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
    color_filter = nv_sig['color_filter']
    cxn.filter_slider_ell9k_color.set_filter(color_filter)  
    
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
    if init_color == 532:
        initialization_time = 10**5
    elif init_color == 638:
        initialization_time = 10**3
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
    print('{} ms pulse time'.format(pulse_time/10**6))
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
                'color_filter': color_filter,
                'readout_color': readout_color,
            'start_coords': start_coords,
            'coords_list': coords_list,
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'num_runs':num_runs,
            'initialization_time': initialization_time,
            'initialization_time-units': 'ns',
            'pulse_time': pulse_time,
            'pulse_time-units': 'ns',
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
    
#    print(readout_counts_array)
#    print(target_counts_array)
    # Statistics
    readout_counts_avg = numpy.average(readout_counts_array, axis=0)
    readout_counts_ste = stats.sem(readout_counts_array, axis=0)
    target_counts_avg = numpy.average(target_counts_array, axis=0)
    target_counts_ste = stats.sem(target_counts_array, axis=0)
#    print(readout_counts_avg)
#    print(target_counts_avg)
    
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
                'readout_color': readout_color,
                'color_filter': color_filter,
            'start_coords': start_coords,
            'coords_list': coords_list,
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
            'initialization_time': initialization_time,
            'initialization_time-units': 'ns',
            'pulse_time': pulse_time,
            'pulse_time-units': 'ns',
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
#    tool_belt.save_figure(fig, file_path)
#    tool_belt.save_figure(fig_temp, file_path + '-path_compare')
    
    return readout_counts_avg, readout_counts_ste, target_counts_avg, target_counts_ste, readout_counts_array, target_counts_array, rad_dist
                        
   # %% UNTESTED
def moving_target_image(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color):
    with labrad.connect() as cxn:
        moving_target_image_with_cxn(cxn, nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
    return
        
def moving_target_image_with_cxn(cxn, nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color):
    tool_belt.reset_cfm(cxn)
        
    # Define paramters
    apd_indices = [0]
    readout_color = 589
    
    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']
    
    # copy the start coords onto the nv_sig
    start_nv_sig = copy.deepcopy(nv_sig)
    start_nv_sig['coords'] = start_coords

    am_589_power = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
    if init_color == 532:
        initialization_time = 10**5
    elif init_color == 638:
        initialization_time = 10**3
    if pulse_color == 532:
        pulse_time = nv_sig['pulsed_reionization_dur']
    elif pulse_color == 638:
        pulse_time = nv_sig['pulsed_ionization_dur']
    
    opti_coords_list = []
    readout_image_array = numpy.empty([num_steps, num_steps, num_runs])
    readout_image_array[:] = numpy.nan
    target_image_array = readout_image_array

    startFunctionTime = time.time()
    
    # The total number of steps we will take swinding throguh the 2D image
    total_num_steps = num_steps*num_steps
    
    # define the sequence paramters
    file_name = 'isolate_nv_charge_dynamics_moving_target.py'
    seq_args = [ initialization_time, pulse_time, readout_pulse_time, 
        laser_515_delay, aom_589_delay, laser_638_delay, 
        galvo_delay, am_589_power, apd_indices[0], init_color, pulse_color, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    
    period = ret_vals[0]
    period_s = period/10**9
    period_s_total = (period_s*total_num_steps + 5)*num_runs
    print('Expected total run time: {:.0f} s'.format(period_s_total))
#    return

    # Optimize at the start of the routine
    opti_coords = optimize.main_with_cxn(cxn, start_nv_sig, apd_indices, 532, disable=False)
    opti_coords_list.append(opti_coords)
    
    # build the lists of voltages
    ret_vals = build_voltages_image(start_coords, img_range, num_steps)
    target_x_values, target_y_values, x_voltages_1d, y_voltages_1d  = ret_vals
    
    x_voltages, y_voltages = build_voltages_main(start_coords, target_x_values, target_y_values)
    
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
                    
        # start on the readout NV
        tool_belt.set_xyz(cxn, start_coords_drift)
        
               
        # load the sequence
        ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
        
        # Adjust the coords for the drift
        x_voltages = x_voltages + drift[0]
        y_voltages = y_voltages + drift[1]
        
        # Load the galvo
        cxn.galvo.load_arb_points_scan(x_voltages, y_voltages, int(period))
        
        #  Set up the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
         
        
        cxn.pulse_streamer.stream_start(target_image_array)
        # The ottal number of samples the APD will look for
        total_num_samples = 3*total_num_steps
    
        tool_belt.init_safe_stop()
        
        new_samples = cxn.apd_tagger.read_counter_simple(total_num_samples)
        # The last of the triplet of readout windows is the counts we are interested in
        readout_counts = new_samples[2::3]
        readout_counts = [int(el) for el in readout_counts]
        target_counts = new_samples[1::3]
        target_counts = [int(el) for el in target_counts]
        
        # sort the readout counts into the image array
        populate_img_array(readout_counts, readout_image_array, r)
        populate_img_array(target_counts, target_image_array, r)
        
        cxn.apd_tagger.stop_tag_stream()
        
        # save incrimentally 
        raw_data = {'start_timestamp': start_timestamp,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_color': readout_color,
            'start_coords': start_coords,
            'img_range': img_range,
            'img_range-units': 'V',
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'num_runs':num_runs,
            'opti_coords_list': opti_coords_list,
            'readout_image_array': readout_image_array.tolist(),
            'readout_image_array-units': 'counts',
            'target_image_array': target_image_array.tolist(),
            'target_image_array-units': 'counts',
            }
                
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
    

    
    # Statistics
    readout_img_avg = numpy.average(readout_image_array, axis=2)
    readout_img_ste = stats.sem(readout_image_array, axis=2)
    target_img_avg = numpy.average(target_image_array, axis=2)
    target_img_ste = stats.sem(target_image_array, axis=2)   
    
    # image dimensions, convert to um
    x_low = x_voltages_1d[0]*35
    x_high = x_voltages_1d[num_steps-1]*35
    y_low = y_voltages_1d[0]*35
    y_high = y_voltages_1d[num_steps-1]*35

    pixel_size = (x_voltages_1d[1] - x_voltages_1d[0])*35
    
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
    title = 'Counts on readout NV from moving target {} nm {} ms pulse'.format(pulse_color, pulse_time)
    fig_readout = tool_belt.create_image_figure(readout_img_avg, img_extent,
                                                title = title)
    title = 'Counts on target {} nm {} ms pulse'.format(pulse_color, pulse_time)
    fig_target = tool_belt.create_image_figure(target_img_avg, img_extent,
                                                title = title)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_colo': readout_color,
            'start_coords': start_coords,
            'img_range': img_range,
            'img_range-units': 'V',
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
            # 'rad_dist': rad_dist.tolist(),
            # 'rad_dist-units': 'V',
            'readout_image_array': readout_image_array.tolist(),
            'readout_image_array-units': 'counts',
            'target_image_array': target_image_array.tolist(),
            'target_image_array-units': 'counts',

            'readout_img_avg': readout_img_avg.tolist(),
            'readout_img_avg-units': 'counts',
            'target_img_avg': target_img_avg.tolist(),
            'target_img_avg-units': 'counts',

            'readout_img_ste': readout_img_ste.tolist(),
            'readout_img_ste-units': 'counts',
            'target_img_ste': target_img_ste.tolist(),
            'target_img_ste-units': 'counts',
            }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig_readout, file_path)
    tool_belt.save_figure(fig_target, file_path + '-pulsed_counts')
    
    return
  
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
                'readout_color': readout_color,
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
#    tool_belt.save_figure(fig_temp, file_path + '-path_compare')
    
    return
# %%
def moving_target_long_t_line(nv_sig, start_coords, end_point, num_steps, num_runs, init_color, pulse_color, point_list = None):
    with labrad.connect() as cxn:
        moving_target_long_t_line_with_cxn(cxn, nv_sig,start_coords,  end_point, num_steps,num_runs,  init_color, pulse_color, point_list)
    return
                
def moving_target_long_t_line_with_cxn(cxn, nv_sig,start_coords,  end_coords, num_steps,num_runs, init_color, pulse_color, point_list = None):
    tool_belt.reset_cfm(cxn)
        
    # Define paramters
    apd_indices = [0]
    readout_color = 589
    
    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
#    galvo_delay = 1*shared_params['large_angle_galvo_delay']
    
    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    pulser_wiring_green = wiring['do_532_aom']
#    pulser_wiring_yellow = wiring['ao_589_aom']
    pulser_wiring_red = wiring['do_638_laser']
    
    # copy the start coords onto the nv_sig
    start_nv_sig = copy.deepcopy(nv_sig)
    start_nv_sig['coords'] = start_coords

    am_589_power = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
    
    if init_color == 532:
        initialization_time = 10**5
        init_laser_delay = laser_515_delay
    elif init_color == 638:
        initialization_time = 10**3
        init_laser_delay = laser_638_delay
        
    if pulse_color == 532:
        pulse_time = nv_sig['pulsed_reionization_dur']
        direct_wiring = pulser_wiring_green
#        laser_delay = laser_515_delay
    elif pulse_color == 638:
        pulse_time = nv_sig['pulsed_ionization_dur']
        direct_wiring = pulser_wiring_red
#        laser_delay = laser_638_delay
    
    opti_coords_list = []
    readout_counts_array = numpy.empty([num_steps, num_runs])
    
    period_s = pulse_time/10**9
    period_s_total = (period_s*num_steps + 5)*num_runs*num_steps
    print('Expected total run time: {:.0f} min'.format(period_s_total/60))

    startFunctionTime = time.time()


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
        
        x_voltages = numpy.linspace(start_coords_drift[0], end_coords_drift[0], num_steps)
        y_voltages = numpy.linspace(start_coords_drift[1], end_coords_drift[1], num_steps)
        
        for i in range(num_steps):
            # start on the readout NV
            tool_belt.set_xyz(cxn, start_coords_drift)
            
            # pulse the laser at the readout
            seq_args = [init_laser_delay, int(initialization_time), 0.0, init_color]           
            seq_args_string = tool_belt.encode_seq_args(seq_args)            
            cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string) 
            
            # move to next spot
            tool_belt.set_xyz(cxn, [x_voltages[i], y_voltages[i], start_coords_drift[2]])
            
            # pulse laser for X time with time.sleep
            
            if pulse_color == 532 or pulse_color==638:
                cxn.pulse_streamer.constant([direct_wiring], 0.0, 0.0)
                time.sleep(pulse_time/10**9)
                cxn.pulse_streamer.constant([], 0.0, 0.0)
            else:
                print('Please use green or red pulses only!')
            
            # Move galvo to center NV
            tool_belt.set_xyz(cxn, start_coords_drift)
            
            # measure the counts
            seq_args = [aom_589_delay, readout_pulse_time, am_589_power, apd_indices[0], readout_color]       
            seq_args_string = tool_belt.encode_seq_args(seq_args)  
            cxn.pulse_streamer.stream_load('simple_readout.py', seq_args_string)  
                   
            cxn.apd_tagger.start_tag_stream(apd_indices) 
            cxn.pulse_streamer.stream_immediate('simple_readout.py', 1, seq_args_string) 
             
            new_samples = cxn.apd_tagger.read_counter_simple(1)
            # The last of the triplet of readout windows is the counts we are interested in
            counts = new_samples[0]
            
            readout_counts_array[i][r] = counts
            
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
            'readout_counts_array': readout_counts_array.tolist(),
            'readout_counts_array-units': 'counts',
            }
                
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
    
    # calculate the radial distances from the readout NV to the target points
    x_target_coords = numpy.array(x_voltages)
    y_target_coords = numpy.array(y_voltages)
    x_diffs = (x_target_coords - start_coords_drift[0])
    y_diffs = (y_target_coords- start_coords_drift[1])
    rad_dist = numpy.sqrt(x_diffs**2 + y_diffs**2)
    
    # Statistics
    readout_counts_avg = numpy.average(readout_counts_array, axis=1)
    readout_counts_ste = stats.sem(readout_counts_array, axis=1)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(rad_dist*35,readout_counts_avg, label = 'counts on readout after each pulse on target spot')
    ax.set_xlabel('Distance from readout NV (um)')
    ax.set_ylabel('Average counts')
    ax.set_title('Stationary readout NV, moving target ({} init, {} s {} pulse)'.\
                                    format(init_color, pulse_time/10**9, pulse_color))
    ax.legend()
    
    
    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_color': readout_color,
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
            'readout_counts_array': readout_counts_array.tolist(),
            'readout_counts_array-units': 'counts',

            'readout_counts_avg': readout_counts_avg.tolist(),
            'readout_counts_avg-units': 'counts',

            'readout_counts_ste': readout_counts_ste.tolist(),
            'readout_counts_ste-units': 'counts',
            }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)
#    tool_belt.save_figure(fig_temp, file_path + '-path_compare')
    
    return
# %%
def plot_times_on_off_nv(nv_sig, readout_coords,  target_nv_coords, dark_coords, num_runs, init_color, pulse_color):
    NV_avg_list = []
    NV_ste_list = []
    dark_avg_list = []
    dark_ste_list = []
    
#    time_list = numpy.array([10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9])
    time_list = numpy.array([10**3, 5*10**3, 10**4, 2*10**4, 5*10**4, 7*10**4, 10**5, 2*10**5, 5*10**5, 7*10**5, 10**6, 10**7, 10**8, 10**9])
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
# %%
def image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color):
    color_filter = nv_sig['color_filter']
    # calculate the list of x and y voltages we'll need to step through
    ret_vals= build_voltages_image(start_coords, img_range, num_steps)
    x_voltages, y_voltages, x_voltages_1d, y_voltages_1d  = ret_vals
    
    z_voltages =  [start_coords[2] for el in x_voltages]
    
    # Combine the x and y voltages together into pairs
    coords_voltages = list(zip(x_voltages, y_voltages, z_voltages))
    
    # Create some empty data lists
    readout_avg_list = numpy.empty(len(coords_voltages))
    readout_ste_list =  numpy.empty(len(coords_voltages))
    target_avg_list =  numpy.empty(len(coords_voltages))
    target_ste_list =  numpy.empty(len(coords_voltages))
    
    readout_image_array = numpy.empty([num_steps, num_steps])
    readout_image_array[:] = numpy.nan
    target_image_array = numpy.empty([num_steps, num_steps])
    target_image_array[:] = numpy.nan
    
    # shuffle the voltages that we're stepping thru
    ind_list = list(range(len(coords_voltages)))
    shuffle(ind_list)
    
    # shuffle the voltages to run
    coords_voltages_shuffle = []
    for i in ind_list:
        coords_voltages_shuffle.append(coords_voltages[i])
    
    coords_voltages_shuffle_list = [list(el) for el in coords_voltages_shuffle]
    
    ret_vals = target_list(nv_sig, start_coords, coords_voltages_shuffle_list, 
                               num_runs, init_color, pulse_color)
    readout_counts_avg, readout_counts_ste, target_counts_avg, target_counts_ste, readout_counts_array, target_counts_array, rad_dist = ret_vals

    # unshuffle the averaged data
    list_ind = 0
    for f in ind_list:
        readout_avg_list[f] = readout_counts_avg[list_ind]
        readout_ste_list[f] = readout_counts_ste[list_ind]
        target_avg_list[f] = target_counts_avg[list_ind]
        target_ste_list[f] = target_counts_ste[list_ind]
        list_ind += 1
        
    # Unshuffle the raw data
    # create unshuffled arrays to fill with the data
    readout_counts_array_unsh = numpy.empty([num_steps*num_steps, num_runs])
    target_counts_array_unsh = numpy.empty([num_steps*num_steps, num_runs])
    rad_dist_unsh = numpy.empty(len(rad_dist))
    
    # transpose the data so that the elements correspond to the coordinate, not the run
    readout_counts_array_sh = copy.deepcopy(readout_counts_array)
    readout_counts_array_sh = numpy.transpose(readout_counts_array_sh)
    target_counts_array_sh = copy.deepcopy(target_counts_array)
    target_counts_array_sh = numpy.transpose(target_counts_array_sh)
    
    # unshuffle the data
    list_ind = 0
    for f in ind_list:
        readout_counts_array_unsh[f] = readout_counts_array_sh[list_ind]
        target_counts_array_unsh[f] = target_counts_array_sh[list_ind]
        rad_dist_unsh[f] = rad_dist[list_ind]
        list_ind += 1
    
    # flip the matrix again so that the elemetns correspond to the run
    readout_counts_array_unsh = numpy.transpose(readout_counts_array_unsh)
    target_counts_array_unsh = numpy.transpose(target_counts_array_unsh)
#    count = 0
#    
#    # step thru each index, run the measurement and then place the counts in the corresponding index in the data list
#    for i in ind_list:
#        print(count)
#        # I technically could just pass the list of shuffled voltages into this function and get the data through that... 
#        ret_vals = target_list(nv_sig, start_coords, [list(coords_voltages[i])], 
#                               num_runs, init_color, pulse_color)
#        readout_counts_avg, readout_counts_ste, target_counts_avg, target_counts_ste = ret_vals
#        
#        readout_avg_list[i] = readout_counts_avg[0]
#        readout_ste_list[i] = readout_counts_ste[0]
#        target_avg_list[i] = target_counts_avg[0]
#        target_ste_list[i] = target_counts_ste[0]
#        
#        count += 1
        
    # create the img arrays
    writePos = []
    readout_image_array = image_sample.populate_img_array(readout_avg_list, readout_image_array, writePos)
    writePos = []
    target_image_array = image_sample.populate_img_array(target_avg_list, target_image_array, writePos)
    
    # image extent
    x_low = x_voltages_1d[0]
    x_high = x_voltages_1d[num_steps-1]
    y_low = y_voltages_1d[0]
    y_high = y_voltages_1d[num_steps-1]

    pixel_size = (x_voltages_1d[1] - x_voltages_1d[0])
    
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
    
    if pulse_color == 532:
        pulse_time = nv_sig['pulsed_reionization_dur']
    elif pulse_color == 638:
        pulse_time = nv_sig['pulsed_ionization_dur']
    
    title = 'Counts on readout NV from moving target {} nm {} ms pulse'.format(pulse_color, pulse_time/10**6)
    fig_readout = tool_belt.create_image_figure(readout_image_array, img_extent,
                                                title = title)
    title = 'Counts on target {} nm {} ms pulse'.format(pulse_color, pulse_time)
    fig_target = tool_belt.create_image_figure(target_image_array, img_extent,
                                                title = title)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    time.sleep(2)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'init_color': init_color,
                'pulse_color': pulse_color,
            'start_coords': start_coords,
            'img_range': img_range,
            'img_range-units': 'V',
            'num_steps': num_steps,
            'color_filter': color_filter, 
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
            'coords_voltages': coords_voltages,
            'coords_voltages-units': '[V, V]',
            'ind_list': ind_list,
            'x_voltages_1d': x_voltages_1d.tolist(),
            'y_voltages_1d': y_voltages_1d.tolist(),
            'rad_dist': rad_dist,
            'rad_dist-units': 'V',
            
            'img_extent': img_extent,
            'img_extent-units': 'V',
            
            'readout_image_array': readout_image_array.tolist(),
            'readout_image_array-units': 'counts',
            'target_image_array': target_image_array.tolist(),
            'target_image_array-units': 'counts',
                    
            'readout_counts_array_unsh': readout_counts_array_unsh,
            'readout_counts_array_unsh-units': 'counts',
            'target_counts_array_unsh': target_counts_array_unsh,
            'target_counts_array_unsh-units': 'counts',

            'readout_avg_list': readout_avg_list.tolist(),
            'readout_avg_list-units': 'counts',
            'target_avg_list': target_avg_list.tolist(),
            'target_avg_list-units': 'counts',

            'readout_ste_list': readout_ste_list.tolist(),
            'readout_ste_list-units': 'counts',
            'target_ste_list': target_ste_list.tolist(),
            'target_ste_list-units': 'counts',
            }
        
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-img')
    tool_belt.save_figure(fig_readout, file_path + '-img')
    tool_belt.save_figure(fig_target, file_path + '-img_pulsed_counts')
    
    return
    
# %% Run the files

if __name__ == '__main__':
    sample_name= 'goeppert-mayer'

#    nv1_2020_12_02 = { 'coords':[0.225, 0.242, 5.20], 
#            'name': '{}-nv1_2020_12_02'.format(sample_name),
#            'expected_count_rate': 50, 'nd_filter': 'nd_1.0',
#            'color_filter': '635-715 bp',
#            'pulsed_readout_dur': 300,
#            'pulsed_SCC_readout_dur': 20000000, 'am_589_power': 0.7, 
#            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120, 
#            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':20, 
#            'magnet_angle': 0,
#            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
#            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}    

    nv2_2020_12_10 = { 'coords':[0.216,0.196,5.15], 
            'name': '{}-nv2_2020_12_10'.format(sample_name),
            'expected_count_rate': 55, 'nd_filter': 'nd_1.0',
            'color_filter': '635-715 bp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 20000000, 'am_589_power': 0.7, 
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':20, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}     
    
    start_coords =[0.216,0.196,5.15]
    end_point = numpy.array(start_coords) + [1,0,0]
    end_point = end_point.tolist()
    num_steps = 20
    num_runs =  25
#    img_range = 0.35
    
    
    for t in [10**9]:        
        init_color = 532
        pulse_color = 532
        nv_sig = copy.deepcopy(nv2_2020_12_10)
        if pulse_color == 532:
            nv_sig['pulsed_reionization_dur'] = t
        if pulse_color == 638:
            nv_sig['pulsed_ionization_dur'] = t 
        nv_sig['color_filter'] = '635-715 bp'
        moving_target_long_t_line(nv_sig, start_coords, end_point, num_steps, num_runs,
                      init_color, pulse_color)
#        
#        init_color = 638
#        pulse_color = 532
#        nv_sig = copy.deepcopy(nv2_2020_12_10)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t 
#        nv_sig['color_filter'] = '635-715 bp'
#        moving_target_long_t_line(nv_sig, start_coords, end_point, num_steps, num_runs,
#                      init_color, pulse_color)
        
 
#    for t in [10**6]:
#        init_color = 638
#        pulse_color = 532
#        nv_sig = copy.deepcopy(nv1_2020_12_02)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t 
#        nv_sig['color_filter'] = '635-715 bp'
#        image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
#        nv_sig['color_filter'] = '715 lp'
#        image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
        
#        init_color = 532
#        pulse_color = 638
#        nv_sig = copy.deepcopy(nv1_2020_12_02)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t 
#        nv_sig['color_filter'] = '635-715 bp'
#        image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
#        nv_sig['color_filter'] = '715 lp'
#        image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
#        
#        init_color = 532
#        pulse_color = 532
#        nv_sig = copy.deepcopy(nv1_2020_12_02)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t 
#        nv_sig['color_filter'] = '635-715 bp'
#        image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
#        nv_sig['color_filter'] = '715 lp'
#        image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
#        
#        init_color = 638
#        pulse_color = 638
#        nv_sig = copy.deepcopy(nv1_2020_12_10)
#        if pulse_color == 532:
#            nv_sig['pulsed_reionization_dur'] = t
#        if pulse_color == 638:
#            nv_sig['pulsed_ionization_dur'] = t 
##        nv_sig['color_filter'] = '635-715 bp'
##        image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
#        nv_sig['color_filter'] = '715 lp'
#        image_indiv_points(nv_sig, start_coords, img_range, num_steps, num_runs, init_color, pulse_color)
        

        
 
    
#    moving_target(nv18_2020_11_10,start_coords,  end_coords, num_steps, num_runs, init_color, pulse_color)    
       
    # plot_times_on_off_nv(nv18_2020_11_10, start_coords,  target_nv_coords, dark_coords, num_runs, init_color, pulse_color)
  
    
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
    
    
    