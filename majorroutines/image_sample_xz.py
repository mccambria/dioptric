# -*- coding: utf-8 -*-
"""
Scan the galvos over the designated area, collecting counts at each point.
Generate an image of the sample.

Created on Tue Apr  9 15:18:53 2019

@author: mccambria
"""


import numpy as np
import utils.tool_belt as tool_belt
import utils.common as common
import utils.kplotlib as kpl
import matplotlib.pylab as plt
import utils.positioning as positioning
import time
import labrad
import majorroutines.optimize as optimize
from utils.tool_belt import ControlStyle


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
        valsToAdd: np.ndarray
            The increment of raw data to add to the image array
        imgArray: np.ndarray
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


# %% Main
    

def main(
    nv_sig, 
    x_range, 
    z_range, 
    num_steps,
    um_scaled=False,
    nv_minus_init=False,
    vmin=None,
    vmax=None,
):

    with labrad.connect() as cxn:
        img_array, x_voltages, z_voltages = main_with_cxn(
            cxn,
            nv_sig,
            x_range,
            z_range,
            num_steps,
            um_scaled,
            nv_minus_init,
            vmin,
            vmax,
        )

    return img_array, x_voltages, z_voltages

def main_with_cxn(
    cxn, 
    nv_sig, 
    x_range, 
    z_range, 
    num_steps,
    um_scaled=False,
    nv_minus_init=False,
    vmin=None,
    vmax=None,
):

    ### Some initial setup
    
    xy_control_style = tool_belt.get_xy_control_style()
    z_control_style = tool_belt.get_z_control_style()
    
    if (xy_control_style == ControlStyle.STREAM) and (z_control_style == ControlStyle.STREAM):
        xyz_control_style = ControlStyle.STREAM
    
    elif (xy_control_style == ControlStyle.STEP) or (z_control_style == ControlStyle.STEP):
        xyz_control_style = ControlStyle.STEP
    
    tool_belt.reset_cfm(cxn)
    x_center, y_center, z_center = positioning.set_xyz_on_nv(cxn, nv_sig)
    optimize.prepare_microscope(cxn, nv_sig)
    xyz_server = tool_belt.get_xyz_server(cxn)
    counter = tool_belt.get_server_counter(cxn)
    pulse_gen = tool_belt.get_server_pulse_gen(cxn)
    total_num_samples = num_steps**2
    
    laser_key = 'imaging_laser'
    readout_laser = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    time.sleep(1)
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    
    # if x_range != y_range:
    #     raise RuntimeError('x and y resolutions must match for now.')
    dir_path = ['', 'Config', 'Positioning']
    cxn.registry.cd(*dir_path)
    _, keys = cxn.registry.dir()
    
    if "xy_small_response_delay" in keys:
        xy_delay = tool_belt.get_registry_entry(
            cxn, "xy_small_response_delay", dir_path
        )
    else:
        xy_delay = tool_belt.get_registry_entry(cxn, "xy_delay", dir_path)


    z_delay = tool_belt.get_registry_entry(cxn, 'z_delay', ['', 'Config', 'Positioning'])
    z_scale = tool_belt.get_registry_entry(cxn, 'z_nm_per_unit', ['', 'Config', 'Positioning'])
    # use whichever delay is longer: 
    if z_delay > xy_delay:
        delay = z_delay
    else:
        delay = xy_delay
    # Get the scale in um per unit
    xy_scale = tool_belt.get_registry_entry(cxn, "xy_nm_per_unit", dir_path)
    if xy_scale == -1:
        um_scaled = False
    else:
        xy_scale *= 1000
        
    try:
        xy_units = common.get_registry_entry(cxn,
            "xy_units", ["", "Config", "Positioning"]
        )
    except Exception as exc:
        print("xy_units not in config")
        xy_units = None
    
    try:
        z_units = common.get_registry_entry(cxn,
            "z_units", ["", "Config", "Positioning"]
        )
    except Exception as exc:
        print("xy_units not in config")
        z_units = None
        
    ### Load the pulse generator
    
    readout = nv_sig['imaging_readout_dur']
    readout_sec = readout / 10**9
    readout_us = readout / 10**3
    
    
    if nv_minus_init:
        laser_key = 'nv-_prep_laser'
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        init = nv_sig["{}_dur".format(laser_key)]
        init_laser = nv_sig[laser_key]
        init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        seq_args = [init, readout, init_laser,init_power, readout_laser,readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        seq_file = 'charge_init-simple_readout.py'
    else:
        seq_args = [delay, readout, readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        seq_file = 'simple_readout.py'
    
    print(seq_args_string)
    ret_vals = pulse_gen.stream_load('simple_readout.py', seq_args_string)
    period = ret_vals[0]


    ### Set up the xy_server
    
    x_num_steps = num_steps
    z_num_steps = num_steps

    ret_vals = positioning.get_scan_grid_2d(
        x_center,z_center,x_range, z_range, x_num_steps, z_num_steps)
    
    if xyz_control_style == ControlStyle.STEP:
        x_positions, z_positions, x_positions_1d, z_positions_1d, extent = ret_vals
        pos_units = 'um'
    
        x_low = x_positions_1d[0]
        x_high = x_positions_1d[x_num_steps-1]
        z_low = z_positions_1d[0]
        z_high = z_positions_1d[z_num_steps-1]
    
        pixel_size = x_positions_1d[1] - x_positions_1d[0]
        
        # %% Set up our raw data objects
        #make an array to save information if the piezo did not reach it's target
        flag_img_array = np.empty((x_num_steps, z_num_steps))
        flag_img_write_pos = []
        #array for dx values
        dx_img_array = np.empty((x_num_steps, z_num_steps))
        dx_img_write_pos = []
        #array for dy values
        dz_img_array = np.empty((x_num_steps, z_num_steps))
        dz_img_write_pos = []
        
    elif xyz_control_style == ControlStyle.STREAM:
        x_voltages, z_voltages, x_voltages_1d, z_voltages_1d, extent = ret_vals
        x_positions_1d, z_positions_1d = x_voltages_1d, z_voltages_1d
        pos_units = "V"
        xyz_server.load_stream_xyz(x_voltages, z_voltages)
    # %% Set up the APD

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = np.empty((x_num_steps, z_num_steps))
    img_array[:] = np.nan
    img_array_kcps = np.copy(img_array)
    img_write_pos = []

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL, no_latex=True)
    
    if um_scaled:
        extent = [el * xy_scale for el in extent]
        axes_labels = ["um", "um"]
    elif xy_units is not None:
        axes_labels = [xy_units, xy_units]
    
    title = f"XZ image under {readout_laser}, {readout_us} us readout"
    
    fig, ax = plt.subplots()
    kpl.imshow(
        ax,
        img_array_kcps,
        title=title,
        axes_labels=axes_labels,
        cbar_label="kcps",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    ### Collect the data
    

    counter.clear_buffer()
    
    counter.start_tag_stream() 
    tool_belt.init_safe_stop()
    
    if xyz_control_style == ControlStyle.STEP:
        
        dx_list = []
        dz_list = []
        
        for i in range(total_num_samples): 
            
            cur_x_pos = x_positions[i]
            cur_z_pos = z_positions[i]
            
            if tool_belt.safe_stop():
                break
            
            flag = xyz_server.write_xyz(cur_x_pos, y_center, cur_z_pos)
            
            # Some diagnostic stuff - checking how far we are from the target pos
            actual_x_pos, actual_z_pos = xyz_server.read_xyz()
            dx_list.append((actual_x_pos-cur_x_pos)*1e3)
            dz_list.append((actual_z_pos-cur_z_pos)*1e3)
            # read the counts at this location
            
            pulse_gen.stream_start(1)
    
            new_samples = counter.read_counter_simple(1) 
            # update the image arrays
            populate_img_array(new_samples, img_array, img_write_pos)
            populate_img_array([flag], flag_img_array, flag_img_write_pos)
            
            populate_img_array([(actual_x_pos-cur_x_pos)*1e3], dx_img_array, dx_img_write_pos)
            populate_img_array([(actual_z_pos-cur_z_pos)*1e3], dz_img_array, dz_img_write_pos)
            # Either include this in loop so it plots data as it takes it (takes about 2x as long)
            # or put it ourside loop so it plots after data is complete
            img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
            kpl.imshow_update(ax, img_array_kcps, vmin, vmax)
        
    elif xyz_control_style == ControlStyle.STREAM:
        
        pulse_gen.stream_start(total_num_samples)

        charge_init = nv_minus_init

        timeout_duration = ((period * (10**-9)) * total_num_samples) + 10
        timeout_inst = time.time() + timeout_duration
        num_read_so_far = 0

        while num_read_so_far < total_num_samples:

            if (time.time() > timeout_inst) or tool_belt.safe_stop():
                break

            # Read the samples
            if charge_init:
                new_samples = counter.read_counter_modulo_gates(2)
            else:
                new_samples = counter.read_counter_simple()

            # Update the image
            num_new_samples = len(new_samples)
            if num_new_samples > 0:
                # If we did charge initialization, subtract out the non-initialized counts
                if charge_init:
                    new_samples = [max(int(el[0]) - int(el[1]), 0) for el in new_samples]
                populate_img_array(new_samples, img_array, img_write_pos)
                img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
                kpl.imshow_update(ax, img_array_kcps, vmin, vmax)
                num_read_so_far += num_new_samples

    ### Clean up and save the data

    tool_belt.reset_cfm(cxn)
    xyz_server.write_xyz(x_center, y_center,z_center)
    
    timestamp = tool_belt.get_time_stamp()
    rawData = {
        'timestamp': timestamp,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                "x_center": x_center,
                "y_center": y_center,
                "z_center": z_center,
                'x_range': x_range,
                'x_range-units': 'um',
                'z_range': z_range,
                'z_range-units': 'um',
                'num_steps': num_steps,
                'readout': readout,
                'readout-units': 'ns',
                "title": title,
                'x_positions_1d': x_positions_1d.tolist(),
                'x_positions_1d-units': pos_units,
                'z_positions_1d': z_positions_1d.tolist(),
                'z_positions_1d-units': pos_units,
                'img_array': img_array.astype(int).tolist(),
                'img_array-units': 'counts',
               }

    filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(rawData, filePath)
    tool_belt.save_figure(fig, filePath)
    
    return img_array, x_positions_1d, z_positions_1d


# %% Run the file


if __name__ == '__main__':

    
    
    file_name = '2022_10_12-15_05_10-ayrton_12-nv_search'

    
    # replot_for_presentation(file_name, 80, 16, centered_at_0 = True)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #
    # csv_name = '{}_{}'.format(timestamp, nv_sig['name'])
    
    # tool_belt.save_image_data_csv(img_array, x_voltages, y_voltages,  path, 
    #                               csv_name)
