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
#import matplotlib.pyplot as plt
import labrad
#from random import shuffle
import majorroutines.image_sample as image_sample
import copy
import scipy.stats as stats

# %%
def build_voltages_from_list(start_coords_list_drift, sample_coords_list_drift):

    num_start_coords = len(start_coords_list_drift)
    # calculate the x values we want to step thru
    start_x_value = start_coords_list_drift[0][0]
    start_y_value = start_coords_list_drift[0][1]
    
    num_sample_coords = len(sample_coords_list_drift)
    
    # we want this list to have the pattern [[NV1], 
    #                                        ( [NV2], [NV3], ...
    #                                        [sample coord], 
    #                                        [NV1], [NV2], [NV3], ... 
    #                                        [NV1] ) * num_sample_coords]
    # The glavo needs a 0th coord, so we'll pass the 1st readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]
    
    # now create a list of all the coords we want to feed to the galvo
    for i in range(num_sample_coords):
        for s in range(num_start_coords-1):
            x_points.append(start_coords_list_drift[s+1][0])
        x_points.append(sample_coords_list_drift[i][0])
        for s in range(num_start_coords):
            x_points.append(start_coords_list_drift[s][0])
        x_points.append(start_x_value)
        
        for s in range(num_start_coords-1):
            y_points.append(start_coords_list_drift[s+1][1])
        y_points.append(sample_coords_list_drift[i][1])
        for s in range(num_start_coords):
            y_points.append(start_coords_list_drift[s][1])
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

def calc_img_extent(x_voltages_1d,y_voltages_1d, num_steps):
        # image extent
    x_low = x_voltages_1d[0]
    x_high = x_voltages_1d[num_steps-1]
    y_low = y_voltages_1d[0]
    y_high = y_voltages_1d[num_steps-1]

    pixel_size = (x_voltages_1d[1] - x_voltages_1d[0])
    
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
    return img_extent
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
def main_data_collection(nv_sig, start_coords_list, sample_coords_list, pulse_time, 
                         num_runs, init_color, pulse_color, readout_color, index_list = []):
    with labrad.connect() as cxn:
        ret_vals = main_data_collection_with_cxn(cxn, nv_sig, 
                        start_coords_list, sample_coords_list,pulse_time, 
                        num_runs, init_color, pulse_color, readout_color, index_list)
    
    readout_counts_array, opti_coords_list = ret_vals
                        
    return readout_counts_array, opti_coords_list
        
def main_data_collection_with_cxn(cxn, nv_sig, start_coords_list, sample_coords_list,
                     pulse_time, num_runs, init_color, pulse_color, readout_color, index_list = []):
    '''
    Runs a measurement where an initial pulse is pulsed on each of the start coords, 
    then a pulse is set on the first point in the coords list, then the 
    counts are recorded on each of the start coords. The routine steps through 
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
    start_coords_list : 2D list (float)
        The coordinates that will be read out from. Note that the routine takes
        this coord from this input not the nv_sig. [x,y,z]
    sample_coords_list : 2D list (float)
        A list of each coordinate that we will pulse the laser at.
    pulse_time: int
        The duration of the pulse on the target coords
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
        The first index refers to the coordinate, the second index refers to the 
        run.        
        
    opti_coords_list : list(float)
        A list of the optimized coordinates recorded during the measurement.
        In the form of [[x,y,z],...]

    '''
    tool_belt.reset_cfm_wout_uwaves(cxn)
    disable_boo = False # can disable the optimize function here.
    
#    # Some checking of the initial pulse colors
#    if init_color == 532 or init_color==638:
#        pass
#    else:
#        print("Please only use '532' or '638' for init_color!")
#    if pulse_color == 532 or pulse_color==638:
#        pass
#    else:
#        print("Please only use '532' or '638' for pulse_color!")
        
    # Define paramters
    apd_indices = [0]
    
    num_samples = len(sample_coords_list)
    num_readout = len(start_coords_list)
    
    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay'] + 5*10**7
         
    # copy the first start coord onto the nv_sig
    start_nv_sig = copy.deepcopy(nv_sig)
    start_nv_sig['coords'] = start_coords_list[0]

    am_589_power = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    color_filter = nv_sig['color_filter']
    cxn.filter_slider_ell9k_color.set_filter(color_filter)  
    
    # define some times for the routine
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
    if init_color == 532:
        initialization_time = nv_sig['pulsed_reionization_dur']
#        init_laser_delay = laser_515_delay
    elif init_color == 638:
        initialization_time = nv_sig['pulsed_ionization_dur']
    
    opti_coords_list = []
    
    ### different building of readout array and estimating measurement time
    # Readout array will be a list in this case. This will be a matrix with 
    # dimensions [num_readout][num_runs][num_samples]. We'll transpose this in the end
    readout_counts_array_transp = numpy.zeros([num_readout,num_runs, num_samples])
#        target_counts_array_transp = []
    
    # define the sequence paramters
    file_name = 'moving_target_multi_readout.py'
    seq_args = [ initialization_time, pulse_time, readout_pulse_time, 
        laser_515_delay, aom_589_delay, laser_638_delay, galvo_delay, 
        am_589_power, apd_indices[0], init_color, pulse_color, readout_color, num_readout]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    
    # print the expected run time
    period = ret_vals[0]
    period_s = period/10**9
    period_s_total = (period_s*num_samples + 1)*num_runs
    print('{} ms pulse time'.format(pulse_time/10**6))
    print('Expected total run time: {:.0f} m'.format(period_s_total/60))
    ### Backto the same
    # Optimize at the start of the routine
    opti_coords = optimize.main_xy_with_cxn(cxn, start_nv_sig, apd_indices, 532, disable=disable_boo)
    opti_coords_list.append(opti_coords)
        
        
    # record the time starting at the beginning of the runs
    start_timestamp = tool_belt.get_time_stamp()
    run_start_time = time.time()
    
    for r in range(num_runs):  
        print( 'run {}'.format(r))
        #optimize every 5 min or so
        # So first check the time. If the time that has passed since the last
        # optimize is longer that 5 min, optimize again
        current_time = time.time()
        if current_time - run_start_time >= 5*60:
            opti_coords = optimize.main_xy_with_cxn(cxn, start_nv_sig, apd_indices, 532, disable=disable_boo)
            opti_coords_list.append(opti_coords) 
            run_start_time = current_time
            
        drift = numpy.array(tool_belt.get_drift())
            
        # get the readout coords with drift
        start_coords_list_drift = numpy.array(start_coords_list) + drift
        sample_coords_list_drift = numpy.array(sample_coords_list) + [drift[0], drift[1]]
                            
        # start on the readout NV
        tool_belt.set_xyz(cxn, start_coords_list_drift[0])
        
        # load the sequence
        ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
        
        # Build the list to step through the coords on readout NV and targets
        x_voltages, y_voltages = build_voltages_from_list(start_coords_list_drift, sample_coords_list_drift)
        # Load the galvo
        cxn.galvo.load_arb_points_scan(x_voltages, y_voltages, int(period))
        
        #  Set up the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        
        cxn.pulse_streamer.stream_start(num_samples)
        # There will be the number of clock pulses of samples, multiplied 
        # the number of coordinates that we are measuring.
        # That wil look like the following:
        num_clk_pulses = 2*num_readout + 1
        total_num_samples = num_clk_pulses*num_samples
        num_read_so_far = 0
        
        total_samples_list = []

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
            
        # Sort the counts that we care about
        for i in range(num_readout):
            gate_ind = num_readout + 1 + i
            readout_counts = total_samples_list[gate_ind::num_clk_pulses]
        
            readout_counts_array_transp[i][r] = readout_counts
        
        cxn.apd_tagger.stop_tag_stream()
        
                # save incrimentally 
        raw_data = {'start_timestamp': start_timestamp,
            'init_color': init_color,
            'pulse_color': pulse_color,
            'readout_color': readout_color,
            'start_coords_list': start_coords_list,
            'sample_coords_list': sample_coords_list,
            'num_steps': num_steps,
            'num_runs': num_runs,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'initialization_time': initialization_time,
            'initialization_time-units': 'ns',
            'pulse_time': pulse_time,
            'pulse_time-units': 'ns',
            'opti_coords_list': opti_coords_list,
            'index_list': index_list,
            'readout_counts_array_transp': readout_counts_array_transp.tolist(),
            'readout_counts_array_transp-units': 'counts',
            }
                
        file_path = tool_belt.get_file_path(__file__, start_timestamp, nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
        
    # save the arrays as a transpose (or flip the 1st and 2nd index) ,
    # so that the 0th index refers to readout NV,
    # the first index refers to the target coordinate, 
    # and the 2nd index refers to the number of run
    readout_counts_array = numpy.transpose(readout_counts_array_transp, (0, 2, 1))   
            
    return readout_counts_array, opti_coords_list

    
# %% 
def do_moving_target_multi_NV_2D(nv_sig, start_coords_list, central_img_coord, img_range, pulse_time, 
                                  num_steps, num_runs, init_color, pulse_color, title_list, readout_color = 589):
    # color_filter = nv_sig['color_filter']
    startFunctionTime = time.time()
    
    # calculate the list of x and y voltages we'll need to step through
    ret_vals= build_voltages_image(central_img_coord, img_range, num_steps)
    x_voltages, y_voltages, x_voltages_1d, y_voltages_1d  = ret_vals
    
#    # check that the readout NVs are in the image area
#    x_low = x_voltages_1d[0]
#    x_high = x_voltages_1d[num_steps-1]
#    y_low = y_voltages_1d[0]
#    y_high = y_voltages_1d[num_steps-1]
#    for c in start_coords_list:
#        if c[0] < x_low or c[0] > x_high:
#            print('Coord {} is outside the image range for x'.format(c))
#            return
#        if c[1] < y_low or c[1] > y_high:
#            print('Coord {} is outside the image range for y'.format(c))
#            return
    
    # Combine the x and y voltages together into pairs
    coords_voltages = list(zip(x_voltages, y_voltages))

    # Run the data collection
    ret_vals = main_data_collection(nv_sig, start_coords_list, coords_voltages,
                            pulse_time, num_runs, init_color, pulse_color, readout_color)
    
    readout_counts_array, opti_coords_list = ret_vals
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power(
                              nv_sig['am_589_power'], nv_sig['nd_filter'])
    endFunctionTime = time.time()
    timeElapsed = endFunctionTime - startFunctionTime
        
    for i in range(len(start_coords_list)):
        # Create an empty image array
        readout_image_array = numpy.empty([num_steps, num_steps])
        
        #pick the ith readout NV to work with
        readout_counts = readout_counts_array[i]
        # Take the average and ste
        readout_counts_avg = numpy.average(readout_counts, axis = 1)
        readout_counts_ste = stats.sem(readout_counts, axis = 1)
    
        # create the img arrays
        writePos = []
        readout_image_array = image_sample.populate_img_array(readout_counts_avg, readout_image_array, writePos)
        
        # image extent
        img_extent =  calc_img_extent(x_voltages_1d,y_voltages_1d, num_steps)
        
        # Create the figure
        
        title = '{}\nwith {} nm {} ms pulse'.format(title_list[i], pulse_color, pulse_time/10**6)
        fig_readout = tool_belt.create_image_figure(readout_image_array, numpy.array(img_extent)*35,
                                                    title = title, um_scaled = True)
    

        # Save file and figure for each NV
        timestamp = tool_belt.get_time_stamp()
        raw_data = {'timestamp': timestamp,
                    'total timeElapsed': timeElapsed,
                    'init_color': init_color,
                    'pulse_color': pulse_color,
                    'readout_color': readout_color,
                    'pulse_time': pulse_time,
                    'pulse_time-units': 'ns',
                    'central_coords': start_coords_list[i],
                'start_coords_list': start_coords_list,
                'img_range': img_range,
                'img_range-units': 'V',
                'num_steps': num_steps,
                'num_runs':num_runs,
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
                'coords_voltages': coords_voltages,
                'coords_voltages-units': '[V, V]',
                'x_voltages_1d': x_voltages_1d.tolist(),
                'y_voltages_1d': y_voltages_1d.tolist(),
                
                'img_extent': img_extent,
                'img_extent-units': 'V',
                
                'readout_image_array': readout_image_array.tolist(),
                'readout_image_array-units': 'counts',
                        
                'readout_counts': readout_counts.tolist(),
                'readout_counts_array-units': 'counts',
                'readout_counts_avg': readout_counts_avg.tolist(),
                'readout_counts_avg-units': 'counts',
                'readout_counts_ste': readout_counts_ste.tolist(),
                'readout_counts_ste-units': 'counts',
                }
            
        file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
        tool_belt.save_raw_data(raw_data, file_path)
        tool_belt.save_figure(fig_readout, file_path)
        # make sure files have unique timestamps
        time.sleep(1)
    
    return
# %% Run the files

if __name__ == '__main__':
    sample_name= 'goeppert-mayer'



    base_sig = { 'coords':[], 
            'name': '{}-2021_01_26'.format(sample_name),
            'expected_count_rate': None,'nd_filter': 'nd_1.0',
            'color_filter': '635-715 bp', 
#            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 10**7,  'am_589_power': 0.3, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 130, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':10, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}   
    

    start_coords_list =[[0.292, -0.370, 5.01],]
    
    title_list = ['goeppert-mayer-nv19_2021_01_26']
    central_img_coord = start_coords_list[0]
    expected_count = 42
    num_steps = 40
    num_runs = 15
    img_range = 0.4
#    pulse_time = 5*10**7
    init_color = 532
    pulse_color = 638
    
    nv_sig = copy.deepcopy(base_sig)
    nv_sig['expected_count_rate'] = expected_count

#    pulse_time = 10**7
#    do_moving_target_multi_NV_2D(nv_sig, start_coords_list, central_img_coord, img_range, pulse_time, 
#                      num_steps, num_runs, init_color, pulse_color, title_list)
    pulse_time = 5*10**7
    do_moving_target_multi_NV_2D(nv_sig, start_coords_list, central_img_coord, img_range, pulse_time, 
                      num_steps, num_runs, init_color, pulse_color, title_list)
    
    # other measurement, include more NVs: see my list and the size we should use. Try at 25*10**7
    
   