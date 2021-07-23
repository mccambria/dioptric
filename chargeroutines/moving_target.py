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

#reset_range = 0.3
#num_steps_reset = int(75 * reset_range)
#reset_time = 10**7

#green_reset_power = 0.6085
#green_pulse_power = 0.65
#green_image_power = 0.65

# %%
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

# def build_voltages_main(start_coords_drift, target_x_values, target_y_values):
#     ''' This function does the basic building of the voltages we need in 
#     this program, in the form of 
#     [[readout], [target], [readout], [readout], 
#                 [target], [readout], [readout],...]
#     '''
    
#     start_x_value = start_coords_drift[0]
#     start_y_value = start_coords_drift[1]
    
#     x_points = [start_x_value]
#     y_points = [start_y_value]
    
#     # now create a list of all the coords we want to feed to the galvo
#     for x in target_x_values:
#         x_points.append(x)
#         x_points.append(start_x_value)
#         x_points.append(start_x_value) 
        

#     # now create a list of all the coords we want to feed to the galvo
#     for y in target_y_values:
#         y_points.append(y)
#         y_points.append(start_y_value)
#         y_points.append(start_y_value)   
        
#     return x_points, y_points

# def build_voltages_start_end(start_coords_drift, end_coords_drift, num_steps):

#     # calculate the x values we want to step thru
#     start_x_value = start_coords_drift[0]
#     end_x_value = end_coords_drift[0]
#     target_x_values = numpy.linspace(start_x_value, end_x_value, num_steps)
    
#     # calculate the y values we want to step thru
#     start_y_value = start_coords_drift[1]
#     end_y_value = end_coords_drift[1]
    
# #    ## Change this code to be more general later:##
# #    mid_y_value = start_y_value + 0.3
# #    
# #    dense_target_y_values = numpy.linspace(start_y_value, mid_y_value, 101)
# #    sparse_target_y_values = numpy.linspace(mid_y_value+0.06, end_y_value, 20)
# #    target_y_values = numpy.concatenate((dense_target_y_values, sparse_target_y_values))
    
    
#     target_y_values = numpy.linspace(start_y_value, end_y_value, num_steps)
    
#     # make a list of the coords that we'll send to the glavo
#     # we want this list to have the pattern [[readout], [target], [readout], [readout], 
#     #                                                   [target], [readout], [readout],...]
#     # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
#     x_points = [start_x_value]
#     y_points = [start_y_value]
    
#     # now create a list of all the coords we want to feed to the galvo
#     for x in target_x_values:
#         x_points.append(x)
#         x_points.append(start_x_value)
#         x_points.append(start_x_value) 
        

#     # now create a list of all the coords we want to feed to the galvo
#     for y in target_y_values:
#         y_points.append(y)
#         y_points.append(start_y_value)
#         y_points.append(start_y_value)   
        
#     return x_points, y_points

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
# %%
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
def main_data_collection(nv_sig, coords_list):
    with labrad.connect() as cxn:
        ret_vals = main_data_collection_with_cxn(cxn, nv_sig, coords_list)
    
    readout_counts_array, opti_coords_list = ret_vals
                        
    return readout_counts_array,  opti_coords_list
        
def main_data_collection_with_cxn(cxn, nv_sig, coords_list):
    '''
    Runs a measurement where an initial pulse is pulsed on the start coords, 
    then a pulse is set on the first point in the coords list, then the 
    counts are recorded on the start coords. The routine steps through 
    the coords list
    
    Two different methods are used for stepping throguh the coords list. If the
    intended pusle is less than 1 s, then one sequence is used which tells the 
    galvo to advance to each coord
    If the pulse is 1 s or longer, then we manually pulse the laser using 
    time.sleep()

    Parameters
    ----------
    cxn : 
        labrad connection. See other our other python functions.
    nv_sig : dict
        dictionary containing onformation about the pulse lengths, pusle powers,
        expected count rate, nd filter, color filter, etc
    coords_list : 2D list (float)
        A list of each coordinate that we will pulse the laser at.

    Returns
    -------
    readout_counts_array : numpy.array
        2D array with the raw counts from each run for each target coordinate 
        measured on the start coord.
        The first index refers to the coordinate, the secon index refers to the 
        run.
    opti_coords_list : list(float)
        A list of the optimized coordinates recorded during the measurement.
        In the form of [[x,y,z],...]

    '''
    tool_belt.reset_cfm(cxn)
        
    # Define paramters
    apd_indices = [0]
    # green_2mw_power = 0.655
    
    num_samples = len(coords_list)
    #delay of aoms and laser, parameters, etc
    # shared_params = tool_belt.get_shared_parameters_dict(cxn)
    # laser_515_DM_delay = shared_params['532_DM_laser_delay']
    # laser_515_AM_delay = shared_params['515_AM_laser_delay']
    # aom_589_delay = shared_params['589_aom_delay']
    # laser_638_delay = shared_params['638_DM_laser_delay']
    # galvo_delay = shared_params['large_angle_galvo_delay']
      
    # Get the pulser wiring
    # wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    # pulser_wiring_green = wiring['do_532_aom']
    # pulser_wiring_red = wiring['do_638_laser']
    
    # get the ionization green power:
    # green_pulse_power = nv_sig['ao_515_pwr']
    
    # copy the start coords onto the nv_sig
    master_nv_sig = copy.deepcopy(nv_sig)
    master_nv_sig['coords'] = nv_sig['coords']
    
    # opti_nv_sig = copy.deepcopy(nv_sig)
    # opti_nv_sig['coords'] = optimize_coords
    # opti_nv_sig['ao_515_pwr'] = green_2mw_power

    # am_589_power = nv_sig['am_589_power']
    # ao_515_pwr = nv_sig['ao_515_pwr']
    nd_filter = nv_sig['nd_filter']
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    
    color_filter = nv_sig['color_filter']
    cxn.filter_slider_ell9k_color.set_filter(color_filter)  
    
    # define some times for the routine
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
#    if init_color == 532 or pulse_color == 532:
        
        
#    if init_color == 532:
#        initialization_time = nv_sig['pulsed_reionization_dur']
#        green_laser_delay = laser_515_DM_delay
#        init_laser_delay = green_laser_delay
#    if init_color == '515a':
    initialization_time = nv_sig['pulsed_reionization_dur']
    green_laser_delay = laser_515_AM_delay
    init_laser_delay = green_laser_delay
#    elif init_color == 638:
    initialization_time = nv_sig['pulsed_ionization_dur']
    init_laser_delay = laser_638_delay
        
#    if pulse_color == 532:
#        direct_wiring = pulser_wiring_green
#    elif pulse_color == 638:
    direct_wiring = pulser_wiring_red
    
    opti_coords_list = []
    # Readout array will be a list in this case. This will be a list with 
    # dimensions [num_samples].
    readout_counts_list = []
    ### different building of readout array and estimating measurement time
    if pulse_time < 10**9:
        
        # define the sequence paramters
        file_name = 'isolate_nv_charge_dynamics_moving_target.py'
        seq_args = [ initialization_time, pulse_time, readout_pulse_time, 
            green_laser_delay, aom_589_delay, laser_638_delay, galvo_delay, 
            am_589_power, 
            green_2mw_power, green_pulse_power, green_2mw_power, 
            apd_indices[0],
            init_color, pulse_color, readout_color]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
#        print(seq_args)
#        return
        
        # print the expected run time
        period = ret_vals[0]
        period_s = period/10**9
        period_s_total = (period_s*num_samples + 1)
        print('{} ms pulse time'.format(pulse_time/10**6))
        print('Expected total run time: {:.1f} m'.format(period_s_total/60))
    
    else:
        period_s = pulse_time/10**9
        period_s_total = (period_s*num_samples + 5)
        print('Expected total run time: {:.0f} min'.format(period_s_total/60))
    
    ### Backto the same
    # Optimize at the start of the routine
    opti_coords = optimize.main_with_cxn(cxn, opti_nv_sig, apd_indices, '515a', disable = False)
    opti_coords_list.append(opti_coords)
              
    drift = numpy.array(tool_belt.get_drift())
            
    # get the readout coords with drift
    start_coords_drift = start_coords + drift
    coords_list_drift = numpy.array(coords_list) + [drift[0], drift[1]]
                                  
    # Different ways of stepping thru coords and recording data                   
    if pulse_time < 10**9:
        # start on the readout NV
        tool_belt.set_xyz(cxn, start_coords_drift)
        
        # load the sequence
        ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
        
        # Build the list to step through the coords on readout NV and targets
        x_voltages, y_voltages = build_voltages_from_list(start_coords_drift, coords_list_drift)
        # Load the galvo
        cxn.galvo.load_multi_point_xy_scan(x_voltages, y_voltages, int(period))
        
        #  Set up the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        
        cxn.pulse_streamer.stream_start(num_samples)
        
        # We'll be lookign for three samples each repetition with how I have
        # the sequence set up
        total_num_samples = 3*num_samples
        
        total_samples_list = []
        num_read_so_far = 0     
        tool_belt.init_safe_stop()
    
        while num_read_so_far < total_num_samples:

            if tool_belt.safe_stop():
                break
    
            # Read the samples and update the image
            new_samples = cxn.apd_tagger.read_counter_simple()
            num_new_samples = len(new_samples)
            
            if num_new_samples > 0:
                for el in new_samples:
                    total_samples_list.append(int(el))
                num_read_so_far += num_new_samples
            
        # The last of the triplet of readout windows is the counts we are interested in
        readout_counts = total_samples_list[2::3]
        readout_counts_list = [int(el) for el in readout_counts]
            
    ### Different way of measuring counts if s long pulses
    else:
        # Get the x voltages and y voltages to step thru the target coords
        x_voltages, y_voltages = list(zip(*coords_list_drift))
        
        
        # Step throguh each coordinate in the coords list
        for i in range(num_samples):
            # start on the readout NV
            tool_belt.set_xyz(cxn, start_coords_drift)
                
            # pulse the laser at the readout
            seq_args = [init_laser_delay, int(initialization_time), 0.0, ao_515_pwr, init_color]           
            seq_args_string = tool_belt.encode_seq_args(seq_args)            
            cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_string) 
                
            # move to next target spot
            tool_belt.set_xyz(cxn, [x_voltages[i], y_voltages[i], start_coords_drift[2]])
            
            # pulse laser for X time with time.sleep
            if pulse_color == '515a':
                cxn.pulse_streamer.constant([], ao_515_pwr, 0.0)
            else:
                cxn.pulse_streamer.constant([direct_wiring], 0.0, 0.0)
            time.sleep(pulse_time/10**9)
            cxn.pulse_streamer.constant([], 0.0, 0.0)
            
            # Move galvo to start coords
            tool_belt.set_xyz(cxn, start_coords_drift)
                
            # measure the counts
            seq_args = [aom_589_delay, readout_pulse_time, am_589_power,ao_515_pwr,  apd_indices[0], readout_color]       
            seq_args_string = tool_belt.encode_seq_args(seq_args)  
            cxn.pulse_streamer.stream_load('simple_readout.py', seq_args_string)  
                   
            cxn.apd_tagger.start_tag_stream(apd_indices) 
            cxn.pulse_streamer.stream_immediate('simple_readout.py', 1, seq_args_string) 
             
            new_samples = cxn.apd_tagger.read_counter_simple(1)
            count = new_samples[0]
            
            readout_counts_list.append(count)
            
            cxn.apd_tagger.stop_tag_stream()
  
    return readout_counts_list, opti_coords_list

# %% 
def main(nv_sig, img_range, 
                              num_steps, num_runs,
                              measurement_type):
    '''
    A measurements to initialize on a single point, then pulse a laser off that
    point, and then read out the charge state on the single point.
    
    Parameters
    ----------
    nv_sig : dict
        dictionary specific to the nv, which contains parameters liek the
        scc readout length, the yellow readout power, the expected count rate...\
    img_range: float
        the range that the 2D area will cover, in both x and y. NOTE: for 1D
        measurement, this is divided in half and used as the farthest point
        in the +x direction
    num_steps: int
        number of steps in 1 direction. For 2D image, this is squared for the 
        total number of points
    num_runs: int
        the number of repetitions of the same measurement, which will be averaged over
    measurement_type: '1D' or '2D'
        either run a 2D oving target or 1D moving target measurement.    
    '''
    # Record start time of the measurement
    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()
            
    if measurement_type == '1D':
        dx = img_range / 2
        end_coords = numpy.array(start_coords) + [dx, 0, 0]
        # calculate the x and y values for linearly spaced points between start and end
        x_voltages = numpy.linspace(start_coords[0], 
                                    end_coords[0], num_steps)
        y_voltages = numpy.linspace(start_coords[1], 
                                    end_coords[1], num_steps)
        # Zip the two list together
        coords_voltages = list(zip(x_voltages, y_voltages))
                    
        # calculate the radial distances from the readout NV to the target points
        x_diffs = (x_voltages - start_coords[0])
        y_diffs = (y_voltages- start_coords[1])
        rad_dist = numpy.sqrt(x_diffs**2 + y_diffs**2)
           
        
    elif measurement_type == '2D':
        # calculate the list of x and y voltages we'll need to step through
        ret_vals= build_voltages_image(start_coords, img_range, num_steps)
        x_voltages, y_voltages, x_voltages_1d, y_voltages_1d  = ret_vals
        
        # Combine the x and y voltages together into pairs
        coords_voltages = list(zip(x_voltages, y_voltages))
        num_samples = len(coords_voltages)
        
        # prep for the figure
        x_low = x_voltages_1d[0]
        x_high = x_voltages_1d[num_steps-1]
        y_low = y_voltages_1d[0]
        y_high = y_voltages_1d[num_steps-1]
    
        pixel_size = (x_voltages_1d[1] - x_voltages_1d[0])
        
        half_pixel_size = pixel_size / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]
        
        
        # Create some empty data lists
        
        readout_image_array = numpy.empty([num_steps, num_steps])
        readout_image_array[:] = numpy.nan
    
        # Create the figure
        init_color1 = eval(tool_belt.get_registry_entry_no_cxn('wavelength',
                          ['Config', 'Optics', nv_sig['initialize_laser']]))
        pulse_color1 = eval(tool_belt.get_registry_entry_no_cxn('wavelength',
                          ['Config', 'Optics', nv_sig['CPG_laser']]))
        pulse_time1 = nv_sig['CPG_laser_dur']
        title = 'Counts on readout NV from moving target {} nm init pulse \n{} nm {} ms pulse'.format(init_color1, pulse_color1, pulse_time1/10**6)
        fig_2D = tool_belt.create_image_figure(readout_image_array, 
                                               numpy.array(img_extent)*35,
                                                title = title, um_scaled = True)

    opti_coords_master = []
    num_samples = len(coords_voltages)
    readout_counts_array = numpy.empty([num_samples, num_runs])
    
    for n in range(num_runs):
        print('Run {}'.format(n))
        # shuffle the voltages that we're stepping thru
        ind_list = list(range(num_samples))
        shuffle(ind_list)
        
        # shuffle the voltages to run
        coords_voltages_shuffle = []
        for i in ind_list:
            coords_voltages_shuffle.append(coords_voltages[i])
        coords_voltages_shuffle_list = [list(el) for el in coords_voltages_shuffle]

        #========================== Run the data collection====================
        ret_vals = main_data_collection(nv_sig, coords_voltages_shuffle_list)
        
        readout_counts_list_shfl, opti_coords_list = ret_vals
        opti_coords_master.append(opti_coords_list[0])
        readout_counts_list_shfl = numpy.array(readout_counts_list_shfl)
        
        # unshuffle the raw data
        list_ind = 0
        for f in ind_list:
            readout_counts_array[f][n] = readout_counts_list_shfl[list_ind]
            list_ind += 1
        
        # Take the average and ste. Need to rotate the matrix, to then only 
        # average the runs that have been completed
        readout_counts_array_rot = numpy.rot90(readout_counts_array)
        readout_counts_avg = numpy.average(readout_counts_array_rot[-(n+1):], axis = 0)
        readout_counts_ste = stats.sem(readout_counts_array_rot[-(n+1):], axis = 0)
        #Save incrementally
        raw_data = {'timestamp': start_timestamp,
                'img_range': img_range,
                'img_range-units': 'V',
                'num_steps': num_steps,
                'num_runs':num_runs,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'coords_voltages': coords_voltages,
                'coords_voltages-units': '[V, V]',
                 'ind_list': ind_list,
                 'opti_coords_list': opti_coords_master,
                'readout_counts_array': readout_counts_array.tolist(),
                'readout_counts_array-units': 'counts',
                'readout_counts_avg': readout_counts_avg.tolist(),
                'readout_counts_avg-units': 'counts',
                'readout_counts_ste': readout_counts_ste.tolist(),
                'readout_counts_ste-units': 'counts',
                }
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')
        
        if measurement_type == '2D':
            # create the img arrays
            writePos = []
            readout_image_array = image_sample.populate_img_array(readout_counts_avg, readout_image_array, writePos)
    
            tool_belt.update_image_figure(fig_2D, readout_image_array)
            
            raw_data['x_voltages_1d'] = x_voltages_1d.tolist()
            raw_data['y_voltages_1d'] = y_voltages_1d.tolist()
            raw_data['img_extent'] = img_extent
            raw_data['img_extent-units'] = 'V'
            raw_data['readout_image_array'] = readout_image_array.tolist()
            raw_data['readout_counts_array-units'] = 'counts'
            
            
            tool_belt.save_figure(fig_2D, file_path)
            
        tool_belt.save_raw_data(raw_data, file_path)
        
    # Save at the end of the whole measurement
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    endFunctionTime = time.time()
    
    # Save
    timeElapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'timeElapsed': timeElapsed,
            'img_range': img_range,
            'img_range-units': 'V',
            'num_steps': num_steps,
            'num_runs':num_runs,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            # 'green_optical_power_pd': green_optical_power_pd,
            # 'green_optical_power_pd-units': 'V',
            # 'green_optical_power_mW': green_optical_power_mW,
            # 'green_optical_power_mW-units': 'mW',
            # 'red_optical_power_pd': red_optical_power_pd,
            # 'red_optical_power_pd-units': 'V',
            # 'red_optical_power_mW': red_optical_power_mW,
            # 'red_optical_power_mW-units': 'mW',
            # 'yellow_optical_power_pd': yellow_optical_power_pd,
            # 'yellow_optical_power_pd-units': 'V',
            # 'yellow_optical_power_mW': yellow_optical_power_mW,
            # 'yellow_optical_power_mW-units': 'mW',
            'coords_voltages': coords_voltages,
            'coords_voltages-units': '[V, V]',
             'ind_list': ind_list,
             'opti_coords_list': opti_coords_master,
            'readout_counts_array': readout_counts_array.tolist(),
            'readout_counts_array-units': 'counts',
            'readout_counts_avg': readout_counts_avg.tolist(),
            'readout_counts_avg-units': 'counts',
            'readout_counts_ste': readout_counts_ste.tolist(),
            'readout_counts_ste-units': 'counts',
            }
        
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    
    if measurement_type == '1D':    
        fig_1D, ax_1D = plt.subplots(1, 1, figsize=(10, 10))
        ax_1D.plot(rad_dist*35,readout_counts_avg, label = nv_sig['name'])
        ax_1D.set_xlabel('Distance from readout NV (um)')
        ax_1D.set_ylabel('Average counts')
        ax_1D.set_title('Stationary readout NV, moving target ({} init, {} s {} pulse)'.\
                                        format(init_color1, pulse_time1/10**9, pulse_color1))
        ax_1D.legend()
        tool_belt.save_figure(fig_1D, file_path)
        
    if measurement_type == '2D':
            raw_data['x_voltages_1d'] = x_voltages_1d.tolist()
            raw_data['y_voltages_1d'] = y_voltages_1d.tolist()
            raw_data['img_extent'] = img_extent
            raw_data['img_extent-units'] = 'V'
            raw_data['readout_image_array'] = readout_image_array.tolist()
            raw_data['readout_counts_array-units'] = 'counts'
            
            
            tool_belt.save_figure(fig_2D, file_path)
         
    tool_belt.save_raw_data(raw_data, file_path)
    
    return
 
 
# %% Run the files

if __name__ == '__main__':
    sample_name= 'goeppert-mayer'



    base_sig = { 'coords':[], 
            'name': '{}-2021_04_15'.format(sample_name),
            'expected_count_rate': None,'nd_filter': 'nd_1.0',
            'color_filter': '635-715 bp', 
#            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 10*10**7,  'am_589_power': 0.15, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 130, 
            'pulsed_reionization_dur': 1*10**3, 'cobalt_532_power':10, 
            'ao_515_pwr': 0.64,
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}   
    expected_count_list = [40, 45, 65, 64, 55, 35,  40, 45  ] # 4/13/21 ###
    start_coords_list = [
[-0.037, 0.119, 5.14],
[-0.090, 0.066, 5.04],
[-0.110, 0.042, 5.13],
[0.051, -0.115, 5.08],
[-0.110, 0.042, 5.06],

[0.063, 0.269, 5.09], 
[0.243, 0.184, 5.12],
[0.086, 0.220, 5.03],
]

#    end_coords = end_coords.tolist()
    num_steps = 41
#    num_runs = 50
#    img_range = 0.45
#    optimize_nv_ind = 3
#    optimize_coords = start_coords_list[optimize_nv_ind]
  
    for s in [5]:
             optimize_nv_ind = s
             optimize_coords = start_coords_list[optimize_nv_ind]
             start_coords = start_coords_list[s]
             nv_sig = copy.deepcopy(base_sig)
             # Set up for current NV
             nv_sig['name']= 'goeppert-mayer-nv{}_2021_04_15'.format(s)
             nv_sig['expected_count_rate'] = expected_count_list[optimize_nv_ind]
             #########
             num_runs = 20       
             # Set up for NV band
             t =5*10**6
             init_color = '515a'
             pulse_color = '515a' 
#             main(nv_sig, start_coords,optimize_coords,  0.4, t, num_steps, num_runs, 
#                                       init_color, pulse_color, measurement_type = '2D' ) 
   
             # Set up for SiV band
#             nv_sig['color_filter'] = '715 lp'
#             nv_sig['nd_filter'] = 'nd_0'
#             nv_sig['am_589_power'] = 0.3
#             nv_sig['pulsed_SCC_readout_dur'] = 4*10**7

             

