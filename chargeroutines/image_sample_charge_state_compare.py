# -*- coding: utf-8 -*-
"""
Scan the galvos over the designated area, collecting counts at each point.
Generate an image of the sample.

Created on Tue Apr  9 15:18:53 2019

@author: mccambria
"""


import numpy
import utils.tool_belt as tool_belt
import time
import labrad
import majorroutines.optimize as optimize


def build_xy_voltages(x_center, y_center, x_range, y_range, num_steps, num_reps):
        x_num_steps = num_steps
        y_num_steps = num_steps

        # Force the scan to have square pixels by only applying num_steps
        # to the shorter axis
        half_x_range = x_range / 2
        half_y_range = y_range / 2

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

        ######### Works for any x_range, y_range #########

        # Winding cartesian product
        # The x values are repeated and the y values are mirrored and tiled
        # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

        # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
        x_inter = numpy.concatenate((x_voltages_1d, numpy.flipud(x_voltages_1d)))
        # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
        if y_num_steps % 2 == 0:  # Even x size
            x_voltages = numpy.tile(x_inter, int(y_num_steps / 2))
        else:  # Odd x size
            x_voltages = numpy.tile(x_inter, int(numpy.floor(y_num_steps / 2)))
            x_voltages = numpy.concatenate((x_voltages, x_voltages_1d))

        # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
        y_voltages = numpy.repeat(y_voltages_1d, x_num_steps)
        
        reps = [el for el in range(num_reps)]
        x_voltages = [val for val in x_voltages for _ in reps]
        y_voltages = [val for val in y_voltages for _ in reps]
        
        return x_voltages, y_voltages, x_voltages_1d, y_voltages_1d
def populate_img_array(valsToAdd, imgArray, writePos):
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
            to the bottom right corner.
    """
    yDim = imgArray.shape[0]
    xDim = imgArray.shape[1]

    if len(writePos) == 0:
        writePos[:] = [xDim, yDim - 1]

    xPos = writePos[0]
    yPos = writePos[1]

    # Figure out what direction we're heading
    headingLeft = ((yDim - 1 - yPos) % 2 == 0)

    for val in valsToAdd:
        if headingLeft:
            # Determine if we're at the left x edge
            if (xPos == 0):
                yPos = yPos - 1
                imgArray[yPos, xPos] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos - 1
                imgArray[yPos, xPos] = val
        else:
            # Determine if we're at the right x edge
            if (xPos == xDim - 1):
                yPos = yPos - 1
                imgArray[yPos, xPos] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos + 1
                imgArray[yPos, xPos] = val
    writePos[:] = [xPos, yPos]

    return imgArray


def on_click_image(event):
    """
    Click handler for images. Prints the click coordinates to the console.

    Params:
        event: dictionary
            Dictionary containing event details
    """

    try:
        print('{:.3f}, {:.3f}'.format(event.xdata, event.ydata))
#        print('[{:.3f}, {:.3f}, 50.0],'.format(event.xdata, event.ydata))
    except TypeError:
        # Ignore TypeError if you click in the figure but out of the image
        pass
    
   
# %% Main
    

def main(nv_sig, x_range, y_range, num_steps, num_reps, apd_indices,
         save_data=True, plot_data=True, 
         um_scaled=False):

    with labrad.connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(cxn, nv_sig, x_range,
                      y_range, num_steps,  num_reps,apd_indices, save_data, plot_data, 
                      um_scaled)

    return img_array, x_voltages, y_voltages

def main_with_cxn(cxn, nv_sig, x_range, y_range, num_steps,  num_reps,
                  apd_indices, save_data=True, plot_data=True, 
                  um_scaled=False):

    # %% Some initial setup
    
    tool_belt.reset_cfm(cxn)
    charge_readout_laser_key = 'charge_readout_laser'
    nvm_prep_laser_key = 'nvm_prep_laser'
    nv0_prep_laser_key= 'nv0_prep_laser'

    drift = tool_belt.get_drift()
    coords = nv_sig['coords']
    adjusted_coords = (numpy.array(coords) + numpy.array(drift)).tolist()
    x_center, y_center, z_center = adjusted_coords
    optimize.prepare_microscope(cxn, nv_sig, adjusted_coords)

    charge_readout_laser_name = nv_sig[charge_readout_laser_key]
    tool_belt.set_filter(cxn, nv_sig, charge_readout_laser_key)
    charge_readout_laser_power = tool_belt.set_laser_power(cxn, nv_sig, charge_readout_laser_key)
    charge_readout = nv_sig['charge_readout_dur']
    
    nvm_prep_laser_name = nv_sig[nvm_prep_laser_key]
    tool_belt.set_filter(cxn, nv_sig, nvm_prep_laser_key)
    nvm_prep_laser_power = tool_belt.set_laser_power(cxn, nv_sig, nvm_prep_laser_key)
    nvm_prep_laser_dur =  nv_sig['nvm_prep_laser_dur']
    
    nv0_prep_laser_name = nv_sig[nv0_prep_laser_key]
    tool_belt.set_filter(cxn, nv_sig, nv0_prep_laser_key)
    nv0_prep_laser_power = tool_belt.set_laser_power(cxn, nv_sig, nv0_prep_laser_key)
    nv0_prep_laser_dur =  nv_sig['nv0_prep_laser_dur']
    
    
    if x_range != y_range:
        raise RuntimeError('x and y resolutions must match for now.')

    xy_server = tool_belt.get_xy_server(cxn)
    xy_delay = tool_belt.get_registry_entry(cxn, 'xy_small_response_delay', ['', 'Config', 'Positioning'])
    # Get the scale in um per unit
    xy_scale = tool_belt.get_registry_entry(cxn, 'xy_nm_per_unit', ['', 'Config', 'Positioning'])
    if xy_scale == -1:
        um_scaled = False
    else: 
        xy_scale *= 1000

    total_num_samples = num_steps**2

    # %% Load the PulseStreamer
    
    
    readout_sec = charge_readout / 10**9
    readout_us = charge_readout / 10**3

    
    seq_args = [nvm_prep_laser_dur, nv0_prep_laser_dur, charge_readout, 
                nvm_prep_laser_name, nv0_prep_laser_name, charge_readout_laser_name,
                nvm_prep_laser_power, nv0_prep_laser_power, charge_readout_laser_power, 
                xy_delay, apd_indices[0]]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    # print(seq_args_string)
    # return
    ret_vals = cxn.pulse_streamer.stream_load('charge_state_comparison.py',
                                              seq_args_string)
    period = ret_vals[0]
    period_s = period/10**9
    total_measurment_time = period_s*total_num_samples*num_reps
    print('Expected run time: {:.1f} m'.format(total_measurment_time/60))
    # return


    # %% Initialize at the starting point

    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])
    
    # %% Set up the xy_server
    x_voltages, y_voltages, x_voltages_1d, y_voltages_1d=  build_xy_voltages(x_center, 
                             y_center, x_range, y_range, num_steps, num_reps)
    xy_server.load_arb_scan_xy(x_voltages, y_voltages, period)

    x_num_steps = len(x_voltages_1d)
    x_low = x_voltages_1d[0]
    x_high = x_voltages_1d[x_num_steps-1]
    y_num_steps = len(y_voltages_1d)
    y_low = y_voltages_1d[0]
    y_high = y_voltages_1d[y_num_steps-1]

    pixel_size = x_voltages[1] - x_voltages[0]

    # %% Set up the APD

    cxn.apd_tagger.start_tag_stream(apd_indices)

    # %% Set up our raw data objects

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = numpy.empty((x_num_steps, y_num_steps))
    img_array[:] = numpy.nan
    img_write_pos = []

    # %% Set up the image display

    if plot_data:

        img_array_kcps = numpy.copy(img_array)

        # For the image extent, we need to bump out the min/max x/y by half the
        # pixel size in each direction so that the center of each pixel properly
        # lies at its x/y voltages.
        half_pixel_size = pixel_size / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]
        if um_scaled:
            img_extent = [(x_high + half_pixel_size)*xy_scale, (x_low - half_pixel_size)*xy_scale,
                      (y_low - half_pixel_size)*xy_scale, (y_high + half_pixel_size)*xy_scale]
        title = r'Difference between charge states'
        fig = tool_belt.create_image_figure(img_array, img_extent,
                        clickHandler=on_click_image, color_bar_label='kcps',
                        title=title, um_scaled=um_scaled)

    # %% Collect the data 
    
    
    start_time = time.time()
    cxn.apd_tagger.clear_buffer()
    cxn.pulse_streamer.stream_start(total_num_samples*num_reps)


    nvm_counts = []
    nv0_counts = []
    num_read_so_far = 0
    tool_belt.init_safe_stop()

    total_samples = []
    while num_read_so_far < total_num_samples*num_reps:

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_tagger.read_counter_separate_gates(1)
        num_new_samples = len(new_samples)

        if num_new_samples > 0:
            for n in range(num_new_samples):
                total_samples.append(new_samples[n])
            num_read_so_far += num_new_samples
            
    # average the repeated counts at each pixel
    for s in range(total_num_samples):
        temp_list_nvm = []
        temp_list_nv0 = []
        for r in range( num_reps):
            temp_list_nvm.append(int(total_samples[s*num_reps + r][0]))
            temp_list_nv0.append(int(total_samples[s*num_reps + r][1]))
        nvm = numpy.average(temp_list_nvm)
        nv0 = numpy.average(temp_list_nv0)
        nvm_counts.append(nvm)
        nv0_counts.append(nv0)
                
    # %% Plot nv0

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array_nv0 = numpy.empty((x_num_steps, y_num_steps))
    img_array_nv0[:] = numpy.nan
    img_write_pos = []

    #  Set up the image display

    if plot_data:

        img_array_nv0_kcps = numpy.copy(img_array_nv0)

        # For the image extent, we need to bump out the min/max x/y by half the
        # pixel size in each direction so that the center of each pixel properly
        # lies at its x/y voltages.
        half_pixel_size = pixel_size / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]
        if um_scaled:
            img_extent = [(x_high + half_pixel_size)*xy_scale, (x_low - half_pixel_size)*xy_scale,
                      (y_low - half_pixel_size)*xy_scale, (y_high + half_pixel_size)*xy_scale]
        title = r'Red pulse followed by yellow'
        fig_nv0 = tool_belt.create_image_figure(img_array_nv0, img_extent,
                        clickHandler=on_click_image, color_bar_label='counts',
                        title=title, um_scaled=um_scaled)

    #  Collect the data
    
    populate_img_array(nv0_counts, img_array_nv0, img_write_pos)
    # This is a horribly inefficient way of getting kcps, but it
    # is easy and readable and probably fine up to some resolution
    if plot_data:
        tool_belt.update_image_figure(fig_nv0, img_array_nv0)
        
    # %% Plot nvm


    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array_nvm = numpy.empty((x_num_steps, y_num_steps))
    img_array_nvm[:] = numpy.nan
    img_write_pos = []

    #  Set up the image display

    if plot_data:

        img_array_nvm_kcps = numpy.copy(img_array_nvm)

        # For the image extent, we need to bump out the min/max x/y by half the
        # pixel size in each direction so that the center of each pixel properly
        # lies at its x/y voltages.
        half_pixel_size = pixel_size / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]
        if um_scaled:
            img_extent = [(x_high + half_pixel_size)*xy_scale, (x_low - half_pixel_size)*xy_scale,
                      (y_low - half_pixel_size)*xy_scale, (y_high + half_pixel_size)*xy_scale]
        title = r'Green pulse followed by yellow'
        fig_nvm = tool_belt.create_image_figure(img_array_nvm, img_extent,
                        clickHandler=on_click_image, color_bar_label='counts',
                        title=title, um_scaled=um_scaled)

    #  Collect the data
    
    populate_img_array(nvm_counts, img_array_nvm, img_write_pos)
    # This is a horribly inefficient way of getting kcps, but it
    # is easy and readable and probably fine up to some resolution
    if plot_data:
        tool_belt.update_image_figure(fig_nvm, img_array_nvm)

    # %% Plot subtracted

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = numpy.empty((x_num_steps, y_num_steps))
    img_array[:] = numpy.nan
    img_write_pos = []

    #  Set up the image display

    if plot_data:

        img_array_kcps = numpy.copy(img_array)

        # For the image extent, we need to bump out the min/max x/y by half the
        # pixel size in each direction so that the center of each pixel properly
        # lies at its x/y voltages.
        half_pixel_size = pixel_size / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]
        if um_scaled:
            img_extent = [(x_high + half_pixel_size)*xy_scale, (x_low - half_pixel_size)*xy_scale,
                      (y_low - half_pixel_size)*xy_scale, (y_high + half_pixel_size)*xy_scale]
        title = r'Confocal scan, {}, {} us readout'.format(charge_readout_laser_power, readout_us)
        fig = tool_belt.create_image_figure(img_array, img_extent,
                        clickHandler=on_click_image, color_bar_label='Diff (counts)',
                        title=title, um_scaled=um_scaled)

    #  Collect the data
   
    diff = numpy.array(nvm_counts) - numpy.array(nv0_counts)
    
    populate_img_array(diff, img_array, img_write_pos)
    # This is a horribly inefficient way of getting kcps, but it
    # is easy and readable and probably fine up to some resolution
    if plot_data:
        tool_belt.update_image_figure(fig, img_array)
    # %% Clean up

    tool_belt.reset_cfm(cxn)

    # Return to center
    xy_server.write_xy(x_center, y_center)

    # %% Save the data

    timestamp = tool_belt.get_time_stamp()
    end_time = time.time()
    
    timeElapsed = end_time - start_time

    rawData = {'timestamp': timestamp,
               'timeElapsed': timeElapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'num_reps': num_reps,
                'x_range': x_range,
                'x_range-units': 'V',
                'y_range': y_range,
                'y_range-units': 'V',
                'num_steps': num_steps,
                'x_voltages': x_voltages,
                'x_voltages-units': 'V',
                'y_voltages': y_voltages,
                'y_voltages-units': 'V',
                'x_voltages_1d': x_voltages_1d.tolist(),
                'x_voltages_1d-units': 'V',
                'y_voltages_1d': y_voltages_1d.tolist(),
                'y_voltages_1d-units': 'V',
                'img_array': img_array.astype(int).tolist(),
                'img_array-units': 'counts',
                'img_array_nv0': img_array_nv0.astype(int).tolist(),
                'img_array_nv0-units': 'counts',
                'img_array_nvm': img_array_nvm.astype(int).tolist(),
                'img_array_nvm-units': 'counts'}

    if save_data:

        filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
        tool_belt.save_raw_data(rawData, filePath)

        if plot_data:

            tool_belt.save_figure(fig, filePath)
            tool_belt.save_figure(fig_nv0, filePath + '_nv0')
            tool_belt.save_figure(fig_nvm, filePath + '_nvm')

    return img_array, x_voltages, y_voltages
# %% Run the file


if __name__ == '__main__':

    scan_range = 0.2
    num_steps = 90
    num_reps = 3
    apd_indices  =[0]
    
     # load the data here 
    sample_name = 'johnson'    
    
    green_laser = 'cobolt_515'
    yellow_laser = 'laserglow_589'
    red_laser = 'cobolt_638'
    # nd_green = 'nd_0.5'
    
    nv_sig = {
        "coords": [0,0,4.7],
        "name": "{}-search".format(sample_name),
        "disable_opt": False,
        "expected_count_rate": None,
        # "imaging_laser": yellow_laser,
        # "imaging_laser_filter": 'nd_0',
        # "imaging_laser_power": 1,
        # "imaging_readout_dur": 5*1e7,
        "imaging_laser": green_laser,
        "imaging_laser_power": 6,
        "imaging_readout_dur": 1e7,
            'nvm_prep_laser': green_laser, 'nvm_prep_laser_power': 6, 'nvm_prep_laser_dur': 1E3,
            'nv0_prep_laser': red_laser, 'nv0_prep_laser_value': 110, 'nv0_prep_laser_dur': 1E3,
            'charge_readout_laser': yellow_laser, 'charge_readout_laser_filter': 'nd_0.5', 
            'charge_readout_laser_power': 0.1, 'charge_readout_dur':50e6,
            'collection_filter': '630_lp', 'magnet_angle': None,
            'resonance_LOW': 2.8012, 'rabi_LOW': 141.5, 'uwave_power_LOW': 15.5,  # 15.5 max
            'resonance_HIGH': 2.9445, 'rabi_HIGH': 191.9, 'uwave_power_HIGH': 14.5}   # 14.5 max

    # Best to try 100ms and nd_0.5 and 0.2 V. Will be a long measurement though...
    try:
        main(nv_sig, scan_range, scan_range, num_steps, num_reps, apd_indices) 
    finally:
        # Reset our hardware - this should be done in each routine, but
        # let's double check here
        tool_belt.reset_cfm()
        # Kill safe stop
        if tool_belt.check_safe_stop_alive():
            print('\n\nRoutine complete. Press enter to exit.')
            tool_belt.poll_safe_stop()
