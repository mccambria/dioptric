# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:46:28 2021

@author: agardill
"""


import numpy
import utils.tool_belt as tool_belt
import utils.positioning as positioning
import time
import labrad
import majorroutines.optimize_digital as optimize
import majorroutines.image_sample as image_sample
import matplotlib.pyplot as plt
  
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


def xz_scan_voltages(x_center, z_center, x_range, z_range, num_steps):
    
        if x_range != z_range:
            raise ValueError("x_range must equal y_range for now")

        x_num_steps = num_steps
        z_num_steps = num_steps

        # Force the scan to have square pixels by only applying num_steps
        # to the shorter axis
        half_x_range = x_range / 2
        half_z_range = z_range / 2

        x_low = x_center - half_x_range
        x_high = x_center + half_x_range
        z_low = z_center - half_z_range
        z_high = z_center + half_z_range

        # Apply scale and offset to get the voltages we'll apply to the stage
        # Note that the polar/azimuthal angles, not the actual x/y positions
        # are linear in these voltages. For a small range, however, we don't
        # really care.
        x_positions_1d = numpy.linspace(x_low, x_high, num_steps)
        z_positions_1d = numpy.linspace(z_low, z_high, num_steps)

        ######### Works for any x_range, y_range #########

        # Winding cartesian product
        # The x values are repeated and the y values are mirrored and tiled
        # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

        # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
        x_inter = numpy.concatenate((x_positions_1d, numpy.flipud(x_positions_1d)))
        # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
        if z_num_steps % 2 == 0:  # Even x size
            x_postions = numpy.tile(x_inter, int(z_num_steps / 2))
        else:  # Odd x size
            x_postions = numpy.tile(x_inter, int(numpy.floor(z_num_steps / 2)))
            x_postions = numpy.concatenate((x_postions, x_positions_1d))

        # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
        z_postions = numpy.repeat(z_positions_1d, x_num_steps)

        return x_postions, z_postions, x_positions_1d, z_positions_1d
# %% Main
    

def main(nv_sig, x_range, z_range, num_steps, apd_indices,
         save_data=True, plot_data=True, 
         um_scaled=False,cbarmin=None,cbarmax=None):

    with labrad.connect() as cxn:
        img_array, x_voltages, z_voltages = main_with_cxn(cxn, nv_sig, x_range,
                      z_range, num_steps, apd_indices, save_data, plot_data, 
                      um_scaled,cbarmin,cbarmax)

    return img_array, x_voltages, z_voltages

def main_with_cxn(cxn, nv_sig, x_range, z_range, num_steps,
                  apd_indices, save_data=True, plot_data=True, 
                  um_scaled=False,cbarmin=None,cbarmax=None):

    # %% Some initial setup
    counter_server = tool_belt.get_server_counter(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)
    startFunctionTime = time.time()
    
    tool_belt.reset_cfm(cxn)
    
    laser_key = 'imaging_laser'

    drift = positioning.get_drift(cxn) 
    coords = nv_sig['coords']
    adjusted_coords = (numpy.array(coords) + numpy.array(drift)).tolist() 
    x_center, y_center, z_center = adjusted_coords
    optimize.prepare_microscope(cxn, nv_sig, adjusted_coords)
    
    readout = nv_sig['imaging_readout_dur']
    readout_us = readout / 10**3
    readout_sec = readout / 10**9

    laser_name = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    laser_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    # print(laser_power)
    
    if x_range != z_range:
        raise RuntimeError('x and z resolutions must match for now.')

    xyz_server = positioning.get_server_pos_xyz(cxn)
    # z_server = tool_belt.get_z_server(cxn)
    
    # Get a couple registry entries
    # See if this setup has finely specified delay times, else just get the 
    # one-size-fits-all value.
    dir_path = ['', 'Config', 'Positioning']
    
    
    cxn.registry.cd(*dir_path)
    _, keys = cxn.registry.dir()
    

    total_num_samples = num_steps**2  
    
    
    
    
    # %% calculate x z positions

    ret_vals = xz_scan_voltages(x_center, z_center,
                                       x_range, z_range, num_steps)
    x_positions, z_positions, x_positions_1d, z_positions_1d = ret_vals
    
    # return
    x_num_steps = len(x_positions_1d)
    x_low = x_positions_1d[0]
    x_high = x_positions_1d[x_num_steps-1]
    z_num_steps = len(z_positions_1d)
    z_low = z_positions_1d[0]
    z_high = z_positions_1d[z_num_steps-1]

    pixel_size = x_positions_1d[1] - x_positions_1d[0]
    
    
    # %% Set up our raw data objects

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = numpy.empty((x_num_steps, z_num_steps))
    img_array[:] = numpy.nan
    img_write_pos = []
    #make an array to save information if the piezo did not reach it's target
    flag_img_array = numpy.empty((x_num_steps, z_num_steps))
    flag_img_write_pos = []
    #array for dx values
    dx_img_array = numpy.empty((x_num_steps, z_num_steps))
    dx_img_write_pos = []
    #array for dz values
    dz_img_array = numpy.empty((x_num_steps, z_num_steps))
    dz_img_write_pos = []


    # %% Set up the image display

    if plot_data:

        img_array_kcps = numpy.copy(img_array)

        # For the image extent, we need to bump out the min/max x/y by half the
        # pixel size in each direction so that the center of each pixel properly
        # lies at its x/y voltages.
        half_pixel_size = pixel_size / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      z_low - half_pixel_size, z_high + half_pixel_size]
        title = r'Confocal scan, {}, {} us readout'.format(laser_name, readout_us)
        fig = tool_belt.create_image_figure(img_array, img_extent,
                        clickHandler=on_click_image, color_bar_label='kcps',
                        title=title, um_scaled=um_scaled)
        
    # %% Collect the data
    update_image_figure = tool_belt.update_image_figure
    tool_belt.init_safe_stop()
    
    counter_server.start_tag_stream(apd_indices) #move outside of sequence
    
    dx_list = []
    dz_list = []
    x_center1, y_center1, z_center1 = coords
    #ret_vals = xy_scan_voltages(x_center1, y_center1,
     #                                  x_range, y_range, num_steps)
    #x_positions1, y_positions1, _, _ = ret_vals
    time_start= time.time()
    opti_interval=2
    seq_args = [0, readout, laser_name, laser_power]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    pulsegen_server.stream_load('simple_readout.py',seq_args_string)
    
    for i in range(total_num_samples): 
        
        # print(i)
        
        # start_t = time.time()
        
        cur_x_pos = x_positions[i]
        cur_z_pos = z_positions[i]
        
        if tool_belt.safe_stop():
            break
        # t2 = time.time()
        # print(t2-start_t)
        
        flag = xyz_server.write_xyz(cur_x_pos, y_center1, cur_z_pos)
        # t3 = time.time()
        # print(t3-t2)
        
        # Some diagnostic stuff - checking how far we are from the target pos
        actual_x_pos, actual_y_pos, actual_z_pos = xyz_server.read_xyz()
        dx_list.append((actual_x_pos-cur_x_pos)*1e3)
        dz_list.append((actual_z_pos-cur_z_pos)*1e3)
        # read the counts at this location
        
        pulsegen_server.stream_start(1)

        new_samples = counter_server.read_counter_simple(1) 
        # update the image arrays
        populate_img_array(new_samples, img_array, img_write_pos)
        populate_img_array([flag], flag_img_array, flag_img_write_pos)
        
        populate_img_array([(actual_x_pos-cur_x_pos)*1e3], dx_img_array, dx_img_write_pos)
        populate_img_array([(actual_z_pos-cur_z_pos)*1e3], dz_img_array, dz_img_write_pos)
        # Either include this in loop so it plots data as it takes it (takes about 2x as long)
        # or put it ourside loop so it plots after data is complete
        if plot_data: ###########################################################
            img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
            update_image_figure(fig, img_array_kcps,cmin=cbarmin,cmax=cbarmax)
        
        # print(time.time() - tt)    
        
    do_analysis=False
    if do_analysis:
       tool_belt.create_image_figure(dx_img_array, img_extent,
                        clickHandler=image_sample.on_click_image, color_bar_label='nm',
                        title = "positional accuracy (dx)", um_scaled=um_scaled,
                        color_map = 'bwr')
       tool_belt.create_image_figure(dz_img_array, img_extent,
                        clickHandler=image_sample.on_click_image, color_bar_label='nm',
                        title = "positional accuracy (dz)", um_scaled=um_scaled,
                        color_map = 'bwr')
        
        
    
    
    
    
       print(numpy.std(abs(numpy.array(dx_list))))
       print(numpy.std(abs(numpy.array(dz_list))))
       fig_pos, axes = plt.subplots(1,2)
       ax = axes[0]
       ax.plot(dx_list)
       ax.set_xlabel('data point')
       ax.set_ylabel('Difference between set values and actual value (nm)')
       ax.set_title('X')
       ax = axes[1]
       ax.plot(dz_list)
       ax.set_xlabel('data point')
       ax.set_ylabel('Difference between set values and actual value (nm)')
       ax.set_title('z')

    # %% Clean up

    tool_belt.reset_cfm(cxn)
    
    # %% Save the data

    endFunctionTime = time.time()
    time_elapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()

    rawData = {
        'timestamp': timestamp,
                'time_elapsed': time_elapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'x_range': x_range,
                'x_range-units': 'um',
                'z_range': z_range,
                'z_range-units': 'um',
                'num_steps': num_steps,
                'readout': readout,
                'readout-units': 'ns',
                'dx_list': dx_list,
                'dx_list-units': 'nm',
                'dz_list': dz_list,
                'dz_list-units': 'nm',
                'x_positions_1d': x_positions_1d.tolist(),
                'x_positions_1d-units': 'um',
                'z_positions_1d': z_positions_1d.tolist(),
                'z_positions_1d-units': 'um',
                'img_array': img_array.astype(int).tolist(),
                'img_array-units': 'counts',
                'flag_img_array': flag_img_array.tolist(),
               }

    if save_data:

        filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
        # print(filePath)
        tool_belt.save_raw_data(rawData, filePath)
        
        # print('here')
        if plot_data:

            tool_belt.save_figure(fig, filePath)


    
    return img_array, x_positions_1d, z_positions_1d
 

# %% Run the file


if __name__ == '__main__':


    path = 'pc_carr/branch_opx-setup/image_sample_xz_digital/2022_11'
    file_name = '2022_11_09-10_20_21-johnson-search'

    data = tool_belt.get_raw_data( file_name, path)
    nv_sig = data['nv_sig']
    timestamp = data['timestamp']
    img_array = data['img_array']
    x_range= data['x_range']
    z_range= data['z_range']
    x_voltages = data['x_positions_1d']
    z_voltages = data['z_positions_1d']
    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    
    x_low = x_voltages[0]
    x_high = x_voltages[-1]
    z_low = z_voltages[0]
    z_high = z_voltages[-1]
    img_extent = [x_high + half_pixel_size,x_low - half_pixel_size,
                  z_low - half_pixel_size, z_high + half_pixel_size]
    
    
    # x_low = -x_range/2
    # x_high = x_range/2
    # z_low = -z_range/2
    # z_high = z_range/2
    # img_extent = [x_low - half_pixel_size,x_high + half_pixel_size,
    #               z_low - half_pixel_size, z_high + half_pixel_size]
    
    # csv_name = '{}_{}'.format(timestamp, nv_sig['name'])
    
    
    tool_belt.create_image_figure(img_array, numpy.array(img_extent), 
                                  clickHandler=image_sample.on_click_image,
                        title=None, color_bar_label='Counts', 
                        min_value=None, um_scaled=True)
    
    
    # tool_belt.save_image_data_csv(img_array, x_voltages, z_voltages,  path, 
    #                               csv_name)   