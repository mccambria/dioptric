# -*- coding: utf-8 -*-
"""
Image the count differential of a sample when under NIR light in a raster scan.
Only designed for ensemble.

Created on July 25th, 2022

@author: cdfox
"""


import numpy
import utils.tool_belt as tool_belt
import time
import labrad
import majorroutines.optimize as optimize


def populate_img_array_bottom_left(valsToAdd, imgArray, writePos):
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
        writePos[:] = [xDim , yDim - 1]

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
#    print([xPos, yPos])
    writePos[:] = [xPos, yPos]

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

def replot_for_presentation(file_name, scale_um_to_V, centered_at_0 = False):
    '''
    Replot measurements based on the scaling of um to V. Useful for preparing
    presentation figures. 
    The coordinates can be centered at (0,0), or use the voltage values

    '''
    scale = scale_um_to_V

    data = tool_belt.get_raw_data(file_name)
    nv_sig = data['nv_sig']
    # timestamp = data['timestamp']
    img_array = numpy.array(data['img_array'])
    x_range= data['x_range']
    y_range= data['y_range']
    x_voltages = data['x_voltages']
    y_voltages = data['y_voltages']
    readout = nv_sig['imaging_readout_dur']

    readout_sec = readout / 10**9

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    
    if centered_at_0:
        x_low = -x_range/2
        x_high = x_range/2
        y_low = -y_range/2
        y_high = y_range/2
        
        img_extent = [x_low - half_pixel_size, x_high + half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]

        
    else:
        x_low = x_voltages[0]
        x_high = x_voltages[-1]
        y_low = y_voltages[0]
        y_high = y_voltages[-1]
        
        img_extent = [x_high - half_pixel_size,x_low + half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    #convert to kcps
    img_array = (img_array[:] / 1000) / readout_sec
    
    tool_belt.create_image_figure(img_array, numpy.array(img_extent)*scale, clickHandler=on_click_image,
                        title=None, color_bar_label='kcps',
                        min_value=None, um_scaled=True)
    
    
def replot_for_analysis(file_name):
    '''
    Replot data just as it appears in measurements
    '''
    data = tool_belt.get_raw_data(file_name)
    nv_sig = data['nv_sig']
    img_array = numpy.array(data['img_array'])
    x_voltages = data['x_voltages']
    y_voltages = data['y_voltages']
    readout = nv_sig['imaging_readout_dur']

    readout_sec = readout / 10**9


    x_low = x_voltages[0]
    x_high = x_voltages[-1]
    y_low = y_voltages[0]
    y_high = y_voltages[-1]


    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [x_high - half_pixel_size,x_low + half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]

    #convert to kcps
    img_array = (img_array[:] / 1000) / readout_sec
    
    tool_belt.create_image_figure(img_array, numpy.array(img_extent), clickHandler=on_click_image,
                        title=None, color_bar_label='kcps',
                        min_value=None, um_scaled=False)
    
# %% Main


def main(nv_sig, x_range, y_range, num_steps, apd_indices, nir_laser_voltage, 
         save_data=True, plot_data=True,
         um_scaled=False, nv_minus_initialization=False):

    with labrad.connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(cxn, nv_sig, x_range,
                      y_range, num_steps, apd_indices, nir_laser_voltage, save_data, plot_data,
                      um_scaled, nv_minus_initialization)

    return img_array, x_voltages, y_voltages

def main_with_cxn(cxn, nv_sig, x_range, y_range, num_steps,
                  apd_indices, nir_laser_voltage, save_data=True, plot_data=True,
                  um_scaled=False, nv_minus_initialization=False):

    # %% Some initial setup

    tool_belt.reset_cfm(cxn)

    drift = tool_belt.get_drift()
    coords = nv_sig['coords']
    adjusted_coords = (numpy.array(coords) + numpy.array(drift)).tolist()
    x_center, y_center, z_center = adjusted_coords
    optimize.prepare_microscope(cxn, nv_sig, adjusted_coords)

    laser_key = 'imaging_laser'
    readout_laser = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    time.sleep(2)
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    # print(readout_power)

    if x_range != y_range:
        raise RuntimeError('x and y resolutions must match for now.')

    xy_server = tool_belt.get_xy_server(cxn)

    # Get a couple registry entries
    # See if this setup has finely specified delay times, else just get the
    # one-size-fits-all value.
    dir_path = ['', 'Config', 'Positioning']
    cxn.registry.cd(*dir_path)
    _, keys = cxn.registry.dir()
    if 'xy_small_response_delay' in keys:
        xy_delay = tool_belt.get_registry_entry(cxn,
                                        'xy_small_response_delay', dir_path)
    else:
        xy_delay = tool_belt.get_registry_entry(cxn, 'xy_delay', dir_path)
    # Get the scale in um per unit
    xy_scale = tool_belt.get_registry_entry(cxn, 'xy_nm_per_unit', dir_path)
    if xy_scale == -1:
        um_scaled = False
    else:
        xy_scale *= 1000

    total_num_samples = num_steps**2

    # %% Load the PulseStreamer

    readout = nv_sig['imaging_readout_dur']

    readout_sec = readout / 10**9
    readout_us = readout / 10**3

    if nv_minus_initialization:
        laser_key = "nv-_prep_laser"
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        init = nv_sig['{}_dur'.format(laser_key)]
        init_laser = nv_sig[laser_key]
        init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        seq_args = [init, readout, apd_indices[0], init_laser, init_power,
                    readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load('charge_initialization-simple_readout.py',
                                                  seq_args_string)
    else:
        seq_args = [xy_delay, readout, apd_indices[0], readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',
                                                  seq_args_string)
    # print(seq_args)
    period = ret_vals[0]


    # %% Initialize at the starting point

    # tool_belt.set_xyz(cxn, [x_center, y_center, z_center])

    # %% Set up the xy_server

    x_voltages, y_voltages = xy_server.load_sweep_scan_xy(x_center, y_center,
                                       x_range, y_range, num_steps, period)

    # return
    x_num_steps = len(x_voltages)
    x_low = x_voltages[0]
    x_high = x_voltages[x_num_steps-1]
    y_num_steps = len(y_voltages)
    y_low = y_voltages[0]
    y_high = y_voltages[y_num_steps-1]

    pixel_size = x_voltages[1] - x_voltages[0]
    
    cxn_power_supply = cxn.power_supply_mp710087 # CF: setup NIR laser

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
        title = r'Confocal scan, {}, {} us readout'.format(readout_laser, readout_us)
        fig = tool_belt.create_image_figure(img_array, img_extent,
                        clickHandler=on_click_image, color_bar_label='Change in Counts (kcps)',
                        title=title, um_scaled=um_scaled)

    # %% Collect the data
    cxn.apd_tagger.clear_buffer()
    # cxn.pulse_streamer.stream_start(total_num_samples) # CF: removed
    cxn.pulse_streamer.stream_start(total_num_samples*2) # CF: now we do two readouts at each pixel. need to alter how I am doing this.


    # timeout_duration = ((period*(10**-9)) * total_num_samples) + 10 # CF: removed
    timeout_duration = ((period*(10**-9)) * total_num_samples*2) + 10 # CF: now we do two readouts at each pixel
    timeout_inst = time.time() + timeout_duration

    num_read_so_far = 0

    charge_initialization = nv_minus_initialization

    tool_belt.init_safe_stop()

    while num_read_so_far < total_num_samples:

        if time.time() > timeout_inst:
            break

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        if charge_initialization:
            cxn_power_supply.output_off() # CF: added for NIR
            time.sleep(1) # CF: added for NIR
            new_samples_noNIR = cxn.apd_tagger.read_counter_modulo_gates(2)
            
            cxn_power_supply.output_on()# CF: added for NIR
            cxn_power_supply.set_voltage(nir_laser_voltage) # CF: added for NIR
            time.sleep(1) # CF: added for NIR
            new_samples_NIR = cxn.apd_tagger.read_counter_modulo_gates(2) # CF: added for NIR
            
            # new_samples = new_samples_NIR - new_samples_noNIR # CF: added for NIR
        else:
            cxn_power_supply.output_off() # CF: added for NIR
            time.sleep(1) # CF: added for NIR
            new_samples_noNIR = cxn.apd_tagger.read_counter_simple()
            
            cxn_power_supply.output_on() # CF: added for NIR
            cxn_power_supply.set_voltage(nir_laser_voltage) # CF: added for NIR
            time.sleep(1) # CF: added for NIR
            new_samples_NIR = cxn.apd_tagger.read_counter_simple() # CF: added for NIR
            
            # new_samples = new_samples_NIR - new_samples_noNIR # CF: added for NIR. this is the issue. how to interpret samples???

#        print(new_samples)
        num_new_samples = len(new_samples)
        if num_new_samples > 0: 

            # If we did charge initialization, subtract out the background
            if charge_initialization:
                new_samples = [max(int(el[0]) - int(el[1]), 0) for el in new_samples]

            populate_img_array(new_samples_noNIR, img_array, img_write_pos) # CF: for noNIR
            populate_img_array(new_samples_NIR, img_array, img_write_pos) # CF: for NIR
            # This is a horribly inefficient way of getting kcps, but it
            # is easy and readable and probably fine up to some resolution
            if plot_data:
                img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
                tool_belt.update_image_figure(fig, img_array_kcps)
            num_read_so_far += num_new_samples
            
    img_array_difference = ...
    cxn_power_supply.output_off()

    # %% Clean up

    tool_belt.reset_cfm(cxn)

    # Return to center
    xy_server.write_xy(x_center, y_center)

    # %% Save the data

    timestamp = tool_belt.get_time_stamp()
    # print(nv_sig['coords'])
    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'drift': drift,
               'x_range': x_range,
               'x_range-units': 'V',
               'y_range': y_range,
               'y_range-units': 'V',
               'num_steps': num_steps,
               'readout': readout,
               'readout-units': 'ns',
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

    return img_array, x_voltages, y_voltages


# %% Run the file


if __name__ == '__main__':

    file_name = '2022_06_20-16_36_36-sandia-R21-a8'
    scale = 83
    
    replot_for_presentation(file_name, scale)
    
    # replot_for_analysis(file_name)
    
