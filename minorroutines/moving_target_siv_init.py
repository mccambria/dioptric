# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:33:57 2020

A routine to take one NV to readout the charge state, after pulsing a laser
at a distance from this readout NV. This routine builds off of moving_target 
by doing a raster scan over the whole area to initialize SiV into either the 
dark or bright state


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


green_reset_power = 0.6075
green_pulse_power = 0.65
green_image_power = 0.65

bright_reset_range = 0.5
bright_reset_steps = 35
bright_reset_power = 0.6358
bright_reset_time = 10**7

dark_reset_range = 0.5
dark_reset_steps = 21
dark_reset_power = 0.6085
dark_reset_time = 10**7

# %%
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
def main_data_collection(nv_sig, start_coords, opti_coords, pulse_coords_list, pulse_time,  
                         num_runs, init_color, pulse_color, readout_color, siv_init, index_list = []):
    with labrad.connect() as cxn:
        ret_vals = main_data_collection_with_cxn(cxn, nv_sig, 
                        start_coords, opti_coords, pulse_coords_list, pulse_time,  
                        num_runs, init_color, pulse_color, readout_color, siv_init, index_list)
    
    readout_counts_array, target_counts_array, opti_coords_list = ret_vals
                        
    return readout_counts_array, target_counts_array, opti_coords_list
        
def main_data_collection_with_cxn(cxn, nv_sig, start_coords, opti_coords,
                                  pulse_coords_list, pulse_time, 
                                  num_runs, init_color, pulse_color, readout_color, siv_init, index_list = []):
    '''
    Runs a measurement where an initial scan is performed in the reset_range
    to put the SiV in a certain state. Then a pulse is set on the first point 
    in the coords list, then the counts are recorded on the start coords. 
    The routine steps through the coords list
    

    Parameters
    ----------
    cxn : 
        labrad connection. See other our other python functions.
    nv_sig : dict
        dictionary containing onformation about the pulse lengths, pusle powers,
        expected count rate, nd filter, color filter, etc
    start_coords : list (float)
        The coordinates that will be read out from. Note that the routine takes
        this coord from this input not the nv_sig. [x,y,z]
    opti_coords : list (float)
        The coordinates that will be oiptimized from. Seperate from the readout
        NV coords since we don't want to mess uop the SiV states [x,y,z]
    pulse_coords_list : 2D list (float)
        A list of each coordinate that we will pulse the laser at.
    reset_range : int
        The range in x and y that wil be reset with a scan.
    pulse_time: int
        The duration of the pulse on the target coords
    reset_time: int
        The duration of the dwell time at each coordinate in the reset scan
    num_runs : int
        Number of repetitions we will run through. These are normally averaged 
        outside of this main function
    init_color : str
        Either '532' or '638'. This is the color that will initialize the 
        starting coords
    pulse_color : str
        Either '532' or '638'. This is the color that will pulse on the target
        coords
    readout_color : str
        Preferably '589'. This is the color that we will readout with on 
        start coord
    index_list: list( int)
        A list of the indexing of the voltages passed. Used for the 2D image. 
        In the case of a crash during the measuremnt, the incrimental data will 
        have the indexing and we can replot the partial data

    Returns
    -------
    readout_counts_array : numpy.array
        2D array with the raw counts from each run for each target coordinate 
        measured on the start coord.
        The first index refers to the coordinate, the secon index refers to the 
        run.        
    target_counts_array : numpy.array
        2D array with the raw counts from each run for each target coordinate 
        measured on the target coordinates.
        The first index refers to the coordinate, the secon index refers to the 
        run.        
        If the pusle length is 1 s or longer, then this is just an empty array
        
    opti_coords_list : list(float)
        A list of the optimized coordinates recorded during the measurement.
        In the form of [[x,y,z],...]

    '''
    tool_belt.reset_cfm_wout_uwaves(cxn)
    disable_boo = False # can disable the optimize function here.
        
    # Define paramters
    apd_indices = [0]
    
    num_samples = len(pulse_coords_list)
#    reset_num_steps =  int(75 * reset_range)
    
    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']
    
    # copy the start coords onto the nv_sig
    start_nv_sig = copy.deepcopy(nv_sig)
    start_nv_sig['coords'] = start_coords
    start_nv_sig['ao_515_pwr'] = green_reset_power
    
    opti_nv_sig = copy.deepcopy(nv_sig)
    opti_nv_sig['coords'] = opti_coords
    opti_nv_sig['ao_515_pwr'] = green_image_power

    am_589_power = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    color_filter = nv_sig['color_filter']
    cxn.filter_slider_ell9k_color.set_filter(color_filter)  
    
    # define some times for the routine
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
    
    opti_coords_list = []
    
    # Readout array will be a list in this case. This will be a matrix with 
    # dimensions [num_samples][num_runs].
    readout_counts_array = numpy.empty([num_samples, num_runs])
    target_counts_array = numpy.empty([num_samples, num_runs])
    
    # define the sequence paramters
    file_name = 'moving_target_siv_init.py'
    seq_args = [pulse_time, readout_pulse_time, 
        laser_515_delay, aom_589_delay, laser_638_delay, galvo_delay, 
        am_589_power, green_pulse_power, green_image_power, apd_indices[0], 
        pulse_color, readout_color]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    
    # print the expected run time
    print('{} ms pulse time'.format(pulse_time/10**6))
    period = ret_vals[0]
    period_s = period/10**9
    print(siv_init)
    if siv_init == 'bright':
        scan_time = bright_reset_steps**2*(bright_reset_time/10**9 + 0.002)
        period_s_total = ((scan_time + period_s)*num_samples + 1)*num_runs
        print('Expected total run time: {:.1f} hr'.format(period_s_total/60/60))
    elif siv_init == 'dark':
        scan_time = dark_reset_steps**2*(dark_reset_time/10**9 + 0.002)
        period_s_total = ((scan_time + period_s)*num_samples + 1)*num_runs
        print('Expected total run time: {:.1f} hr'.format(period_s_total/60/60))
    else:
        period_s_total = ((period_s)*num_samples + 1)*num_runs
        print('Expected total run time: {:.0f} min'.format(period_s_total/60))
#    return
    # Optimize at the start of the routine
    opti_coords_measured = optimize.main_xy_with_cxn(cxn, opti_nv_sig, apd_indices, '515a', disable=disable_boo)
    opti_coords_list.append(opti_coords_measured)
        
    # record the time starting at the beginning of the runs
    start_timestamp = tool_belt.get_time_stamp()
    run_start_time = time.time()
    
    for r in range(num_runs):  
        print( 'run {}'.format(r))
        
        
        for n in range(num_samples):
            #optimize every 5 min or so
            # So first check the time. If the time that has passed since the last
            # optimize is longer that 5 min, optimize again
            current_time = time.time()
            if current_time - run_start_time >= 5*60:
                opti_coords_measured = optimize.main_xy_with_cxn(cxn, opti_nv_sig, apd_indices, '515a', disable=disable_boo)
                opti_coords_list.append(opti_coords_measured) 
                run_start_time = current_time
                
            drift = numpy.array(tool_belt.get_drift())
                
            # get the readout coords with drift
            start_coords_drift = start_coords + drift
            coords_list_drift = numpy.array(pulse_coords_list) + [drift[0], drift[1]]
                    
            # unzip coords list so we have X and Y lsit of coords
            unzipped_coords_list_drift = list(zip(*coords_list_drift))
            x_pulse_coords = unzipped_coords_list_drift[0]
            y_pulse_coords = unzipped_coords_list_drift[1]

            #############            
            if siv_init == 'bright':
                reset_sig = copy.deepcopy(nv_sig)
                reset_sig['coords'] = start_coords
                reset_sig['ao_515_pwr'] = bright_reset_power
                _,_,_ = image_sample.main(reset_sig, bright_reset_range, bright_reset_range, 
                                          bright_reset_steps, 
                                  apd_indices, init_color,readout = bright_reset_time,  
                                  save_data=False, plot_data=False) 
            elif siv_init == 'dark':
                reset_sig = copy.deepcopy(nv_sig)
                reset_sig['coords'] = start_coords
                reset_sig['ao_515_pwr'] = dark_reset_power
                _,_,_ = image_sample.main(reset_sig, dark_reset_range, dark_reset_range, 
                                          dark_reset_steps, 
                                  apd_indices, init_color,readout = dark_reset_time,  
                                  save_data=False, plot_data=False) 
                

            # pulse laser for 100 us on readout NV regardless
            tool_belt.set_xyz(cxn, start_coords_drift)
            seq_args_pulse = [140, 10**5,am_589_power, green_pulse_power, '515a']  
            seq_args_pulse_string = tool_belt.encode_seq_args(seq_args_pulse)            
            cxn.pulse_streamer.stream_immediate('simple_pulse.py', 1, seq_args_pulse_string) 
            
#            seq_args = [pulse_time, readout_pulse_time, 
#            laser_515_delay, aom_589_delay, laser_638_delay, galvo_delay, 
#            am_589_power, green_pulse_power, green_image_power, apd_indices[0], 
#            pulse_color, readout_color]
#            seq_args_string = tool_belt.encode_seq_args(seq_args)

            #############
            
            
            # start on the coordinate where pulse will be
            tool_belt.set_xyz(cxn,[x_pulse_coords[n], y_pulse_coords[n], start_coords_drift[2]] )
        
            # load the sequence
            ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
        
            # Build a tuple of voltages to pass the galvo.
            # First do pulse coord, then readout coord, then readout coords
            galvo_coords_x = [x_pulse_coords[n], start_coords_drift[0], start_coords_drift[0]]
            galvo_coords_y = [y_pulse_coords[n], start_coords_drift[1], start_coords_drift[1]]
            
            cxn.galvo.load_arb_points_scan(galvo_coords_x, galvo_coords_y, int(period))
        
            #  Set up the APD
            cxn.apd_tagger.start_tag_stream(apd_indices)
        
            cxn.pulse_streamer.stream_start(1)
        
            # We're looking for 2 samples, since there will be two clock pulses
            total_num_samples = 2 
        
#            total_samples_list = []
#            num_read_so_far = 0     
            tool_belt.init_safe_stop()

#            while num_read_so_far < total_num_samples:
#    
#                if tool_belt.safe_stop():
#                    break
#        
#                # Read the samples and update the image
#                new_samples = cxn.apd_tagger.read_counter_simple()
#                num_new_samples = len(new_samples)
#                
#                if num_new_samples > 0:
#                    for el in new_samples:
#                        total_samples_list.append(int(el))
#                    num_read_so_far += num_new_samples
##                    
##                # Read the counts
            new_samples = cxn.apd_tagger.read_counter_simple(total_num_samples)
#            print(x_pulse_coords[n])
#            print(new_samples)
            # The last of the triplet of readout windows is the counts we are interested in
            readout_counts = int(new_samples[1])
            target_counts =int( new_samples[0])
            
            readout_counts_array[n][r] = readout_counts
            target_counts_array[n][r] = target_counts
            
            cxn.apd_tagger.stop_tag_stream()
            

            
        # save incrimentally 
        raw_data = {'start_timestamp': start_timestamp,
            'init_color': init_color,
            'pulse_color': pulse_color,
            'readout_color': readout_color,
            'start_coords': start_coords,
            'opti_coords': opti_coords,
            'siv_init': siv_init,
            'pulse_coords_list': pulse_coords_list,
            'green_pulse_power': green_pulse_power,
            'green_image_power': green_image_power,
            'num_steps': num_steps,
            'num_runs': num_runs,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'pulse_time': pulse_time,
            'pulse_time-units': 'ns',
            'opti_coords_list': opti_coords_list,
            'index_list': index_list,
            'readout_counts_array': readout_counts_array.tolist(),
            'readout_counts_array-units': 'counts',
            'target_counts_array': target_counts_array.tolist(),
            'target_counts_array-units': 'counts',
            }
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
            
    return readout_counts_array, target_counts_array, opti_coords_list

# %%
def do_moving_target_1D_line(nv_sig, start_coords, end_coords,opti_coords,  pulse_time, 
                             num_steps, num_runs, init_color, pulse_color, siv_init, readout_color = 589):
    
    startFunctionTime = time.time()
    
    # calculate the x and y values for linearly spaced points between start and end
    x_voltages = numpy.linspace(start_coords[0], 
                                end_coords[0], num_steps)
    y_voltages = numpy.linspace(start_coords[1], 
                                end_coords[1], num_steps)
    # Zip the two list together
    coords_list = list(zip(x_voltages, y_voltages))
    
    # Run the data collection
    ret_vals = main_data_collection(nv_sig, start_coords, opti_coords, coords_list, pulse_time,  
                         num_runs, init_color, pulse_color, readout_color, siv_init)
    readout_counts_array, target_counts_array, opti_coords_list = ret_vals
            
    # calculate the radial distances from the readout NV to the target points
    x_diffs = (x_voltages - start_coords[0])
    y_diffs = (y_voltages- start_coords[1])
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
    ax.plot(rad_dist*35,readout_counts_avg, label = nv_sig['name'])
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
                'siv_init': siv_init,
                'init_color': init_color,
                'pulse_color': pulse_color,
                'readout_color': readout_color,
            'start_coords': start_coords,
            'end_coords': end_coords,
            'opti_coords': opti_coords,
            'num_steps': num_steps,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            
            'bright_reset_range': bright_reset_range,
            'bright_reset_steps':bright_reset_steps,
            'bright_reset_power':bright_reset_power,
            'bright_reset_time':bright_reset_time,
            
            'dark_reset_range': dark_reset_range,
            'dark_reset_steps':dark_reset_steps,
            'dark_reset_power':dark_reset_power,
            'dark_reset_time':dark_reset_time,
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
            'pulse_time': pulse_time,
            'pulse_time-unit': 'ns',
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
    
    return

# %% 
#def do_moving_target_2D_image(nv_sig, start_coords, opti_coords, img_range, 
#                              pulse_time, reset_range, reset_time, num_steps, 
#                              num_runs, init_color, pulse_color, readout_color = 589):
#
#    # color_filter = nv_sig['color_filter']
#    startFunctionTime = time.time()
#    
#    # calculate the list of x and y voltages we'll need to step through
#    ret_vals= build_voltages_image(start_coords, img_range, num_steps)
#    x_voltages, y_voltages, x_voltages_1d, y_voltages_1d  = ret_vals
#    
##    z_voltages =  [start_coords[2] for el in x_voltages]
#    
#    # Combine the x and y voltages together into pairs
#    coords_voltages = list(zip(x_voltages, y_voltages))
#    num_samples = len(coords_voltages)
#    
#    # Create some empty data lists
#    readout_counts_array = numpy.empty([num_samples, num_runs])
#    target_counts_array = numpy.empty([num_samples, num_runs])
#    
#    readout_image_array = numpy.empty([num_steps, num_steps])
#    readout_image_array[:] = numpy.nan
#    target_image_array = numpy.empty([num_steps, num_steps])
#    target_image_array[:] = numpy.nan
#    
#    # shuffle the voltages that we're stepping thru
#    ind_list = list(range(num_samples))
#    shuffle(ind_list)
#    
#    # shuffle the voltages to run
#    coords_voltages_shuffle = []
#    for i in ind_list:
#        coords_voltages_shuffle.append(coords_voltages[i])
##    
#    coords_voltages_shuffle_list = [list(el) for el in coords_voltages_shuffle]
#
#    # Run the data collection
#    ret_vals = main_data_collection(nv_sig, start_coords, opti_coords, coords_voltages_shuffle_list, reset_range, pulse_time, reset_time,  
#                         num_runs, init_color, pulse_color, readout_color, index_list = ind_list)
#    
#    readout_counts_array_shfl, target_counts_array_shfl, opti_coords_list = ret_vals
#    readout_counts_array_shfl = numpy.array(readout_counts_array_shfl)
#    target_counts_array_shfl = numpy.array(target_counts_array_shfl)
#    
#    # unshuffle the raw data
#    list_ind = 0
#    for f in ind_list:
#        readout_counts_array[f] = readout_counts_array_shfl[list_ind]
#        target_counts_array[f] = target_counts_array_shfl[list_ind]
#        list_ind += 1
#        
#    # Take the average and ste
#    readout_counts_avg = numpy.average(readout_counts_array, axis = 1)
#    readout_counts_ste = stats.sem(readout_counts_array, axis = 1)
#    target_counts_avg = numpy.empty([1])
#    target_counts_ste = numpy.empty([1])
#
#    # create the img arrays
#    writePos = []
#    readout_image_array = image_sample.populate_img_array(readout_counts_avg, readout_image_array, writePos)
#    
#    # image extent
#    x_low = x_voltages_1d[0]
#    x_high = x_voltages_1d[num_steps-1]
#    y_low = y_voltages_1d[0]
#    y_high = y_voltages_1d[num_steps-1]
#
#    pixel_size = (x_voltages_1d[1] - x_voltages_1d[0])
#    
#    half_pixel_size = pixel_size / 2
#    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
#                  y_low - half_pixel_size, y_high + half_pixel_size]
#    
#    # Create the figure
#    title = 'Counts on readout NV from moving target {} nm init pulse \n{} nm {} ms pulse'.format(init_color, pulse_color, pulse_time/10**6)
#    fig_readout = tool_belt.create_image_figure(readout_image_array, numpy.array(img_extent)*35,
#                                                title = title, um_scaled = True)
#    # If the pusle time is longer that 1 s, we don't have target count data
#    if pulse_time < 10**9:
#        # Take the average and ste
#        target_counts_avg = numpy.average(target_counts_array, axis = 1)
#        target_counts_ste = stats.sem(target_counts_array, axis = 1)
#    
#        # create the img arrays
#        writePos = []
#        target_image_array = image_sample.populate_img_array(target_counts_avg, target_image_array, writePos)
#        
#        title = 'Counts on target  {} nm init pulse \n{} nm {} ms pulse'.format(init_color, pulse_color, pulse_time/10**6)
#        fig_target = tool_belt.create_image_figure(target_image_array, numpy.array(img_extent)*35,
#                                                    title = title, um_scaled = True)
#    
#    # measure laser powers:
#    green_optical_power_pd, green_optical_power_mW, \
#            red_optical_power_pd, red_optical_power_mW, \
#            yellow_optical_power_pd, yellow_optical_power_mW = \
#            tool_belt.measure_g_r_y_power(
#                              nv_sig['am_589_power'], nv_sig['nd_filter'])
#    
#    
#    endFunctionTime = time.time()
#    # Save
#    timeElapsed = endFunctionTime - startFunctionTime
#    timestamp = tool_belt.get_time_stamp()
#    raw_data = {'timestamp': timestamp,
#                'timeElapsed': timeElapsed,
#                'init_color': init_color,
#                'pulse_color': pulse_color,
#                'readout_color': readout_color,
#                'pulse_time': pulse_time,
#                'pulse_time-units': 'ns',
#            'start_coords': start_coords,
#            'opti_coords': opti_coords,
#            'img_range': img_range,
#            'img_range-units': 'V',
#            'reset_range': reset_range,
#            'reset_time': reset_time,
#            'reset_time-units': 'ns',
#            'num_steps': num_steps,
#            'num_runs':num_runs,
#            'nv_sig': nv_sig,
#            'nv_sig-units': tool_belt.get_nv_sig_units(),
#            'bright_reset_range': bright_reset_range,
#            'bright_reset_steps':bright_reset_steps,
#            'bright_reset_power':bright_reset_power,
#            'bright_reset_time':bright_reset_time,
#            
#            'dark_reset_range': dark_reset_range,
#            'dark_reset_steps':dark_reset_steps,
#            'dark_reset_power':dark_reset_power,
#            'dark_reset_time':dark_reset_time,
#            'green_optical_power_pd': green_optical_power_pd,
#            'green_optical_power_pd-units': 'V',
#            'green_optical_power_mW': green_optical_power_mW,
#            'green_optical_power_mW-units': 'mW',
#            'red_optical_power_pd': red_optical_power_pd,
#            'red_optical_power_pd-units': 'V',
#            'red_optical_power_mW': red_optical_power_mW,
#            'red_optical_power_mW-units': 'mW',
#            'yellow_optical_power_pd': yellow_optical_power_pd,
#            'yellow_optical_power_pd-units': 'V',
#            'yellow_optical_power_mW': yellow_optical_power_mW,
#            'yellow_optical_power_mW-units': 'mW',
#            'coords_voltages': coords_voltages,
#            'coords_voltages-units': '[V, V]',
#             'ind_list': ind_list,
#            'x_voltages_1d': x_voltages_1d.tolist(),
#            'y_voltages_1d': y_voltages_1d.tolist(),
#            
#            'img_extent': img_extent,
#            'img_extent-units': 'V',
#            
#            'readout_image_array': readout_image_array.tolist(),
#            'readout_image_array-units': 'counts',
#            'target_image_array': target_image_array.tolist(),
#            'target_image_array-units': 'counts',
#                    
#            'readout_counts_array': readout_counts_array.tolist(),
#            'readout_counts_array-units': 'counts',
#            'target_counts_array': target_counts_array.tolist(),
#            'target_counts_array-units': 'counts',
#
#            'readout_counts_avg': readout_counts_avg.tolist(),
#            'readout_counts_avg-units': 'counts',
#            'target_counts_avg': target_counts_avg.tolist(),
#            'target_counts_avg-units': 'counts',
#
#            'readout_counts_ste': readout_counts_ste.tolist(),
#            'readout_counts_ste-units': 'counts',
#            'target_counts_ste': target_counts_ste.tolist(),
#            'target_counts_ste-units': 'counts',
#            }
#        
#    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
#    tool_belt.save_raw_data(raw_data, file_path)
#    tool_belt.save_figure(fig_readout, file_path)
#    if pulse_time < 10**9:
#        tool_belt.save_figure(fig_target, file_path + '-target_counts')
#    
#    return
 
 
# %% Run the files

if __name__ == '__main__':
    sample_name= 'goeppert-mayer'



    base_sig = { 'coords':[], 
            'name': '{}-2021_02_26'.format(sample_name),
            'expected_count_rate': None,'nd_filter': 'nd_1.0',
            'color_filter': '635-715 bp', 
#            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 8*10**7,  'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 130, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':10, 
            'ao_515_pwr': 0.7,
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}   

    start_coords = [-0.081, 0.096, 5.48]
    end_coords = [-0.081 + 0.15, 0.096, 5.48]
    
    opti_coords = [0.307, 0.441, 5.50]
    num_steps = 30
#    image_range = 0.3
    num_runs = 35
    
    

    init_color = '515a'
    pulse_color = '515a'
    nv_sig = copy.deepcopy(base_sig)
    # Set up for current NV
    nv_sig['name']= 'goeppert-mayer-nv{}_2021_03_01'.format(3)
    nv_sig['expected_count_rate'] = 41
    # Measurements
    t =10*10**6
    do_moving_target_1D_line(nv_sig, start_coords, end_coords,opti_coords,  t, 
                             num_steps, num_runs, init_color, pulse_color, siv_init = None)
#    do_moving_target_1D_line(nv_sig, start_coords, end_coords,opti_coords,  t,
#                             num_steps, num_runs, init_color, pulse_color, siv_init = 'dark')
#    do_moving_target_1D_line(nv_sig, start_coords, end_coords,opti_coords,  t, 
#                             num_steps, num_runs, init_color, pulse_color, siv_init = 'bright')
