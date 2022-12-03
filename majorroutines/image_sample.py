# -*- coding: utf-8 -*-
"""
Scan over the designated area, collecting counts at each point.
Generate an image of the sample.

Created on April 9th, 2019

@author: mccambria
"""


import numpy as np
import utils.tool_belt as tool_belt
import time
import labrad
import majorroutines.optimize as optimize
import utils.kplotlib as kpl
import matplotlib.pyplot as plt
import cv2
import utils.common as common
import utils.positioning as positioning


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
    headingLeft = (yDim - 1 - yPos) % 2 == 0

    for val in valsToAdd:
        if headingLeft:
            # Determine if we're at the left x edge
            if xPos == 0:
                yPos = yPos - 1
                imgArray[yPos, xPos] = val
                headingLeft = not headingLeft  # Flip directions
            else:
                xPos = xPos - 1
                imgArray[yPos, xPos] = val
        else:
            # Determine if we're at the right x edge
            if xPos == xDim - 1:
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
        print("{:.3f}, {:.3f}".format(event.xdata, event.ydata))
    #        print('[{:.3f}, {:.3f}, 50.0],'.format(event.xdata, event.ydata))
    except TypeError:
        # Ignore TypeError if you click in the figure but out of the image
        pass


def replot_for_presentation(file_name, scale_um_to_V, centered_at_0=False):
    """
    Replot measurements based on the scaling of um to V. Useful for preparing
    presentation figures.
    The coordinates can be centered at (0,0), or use the voltage values

    """
    scale = scale_um_to_V

    data = tool_belt.get_raw_data(file_name)
    nv_sig = data["nv_sig"]
    # timestamp = data['timestamp']
    img_array = np.array(data["img_array"])
    x_range = data["x_range"]
    y_range = data["y_range"]
    x_voltages = data["x_voltages"]
    y_voltages = data["y_voltages"]
    readout = nv_sig["imaging_readout_dur"]

    readout_sec = readout / 10**9

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2

    if centered_at_0:
        x_low = -x_range / 2
        x_high = x_range / 2
        y_low = -y_range / 2
        y_high = y_range / 2

        img_extent = [
            x_low - half_pixel_size,
            x_high + half_pixel_size,
            y_low - half_pixel_size,
            y_high + half_pixel_size,
        ]

    else:
        x_low = x_voltages[0]
        x_high = x_voltages[-1]
        y_low = y_voltages[0]
        y_high = y_voltages[-1]

        img_extent = [
            x_high - half_pixel_size,
            x_low + half_pixel_size,
            y_low - half_pixel_size,
            y_high + half_pixel_size,
        ]

    # convert to kcps
    img_array = (img_array[:] / 1000) / readout_sec

    tool_belt.create_image_figure(
        img_array,
        np.array(img_extent) * scale,
        clickHandler=on_click_image,
        title=None,
        color_bar_label="kcps",
        axes_labels=["x (um)", "y (um)"],
    )


def replot_for_analysis(file_name, cmin=None, cmax=None):
    """
    Replot data just as it appears in measurements
    """
    data = tool_belt.get_raw_data(file_name)
    nv_sig = data["nv_sig"]
    img_array = np.array(data["img_array"])
    x_voltages = data["x_voltages"]
    y_voltages = data["y_voltages"]
    readout = nv_sig["imaging_readout_dur"]

    readout_sec = readout / 10**9

    x_low = x_voltages[0]
    x_high = x_voltages[-1]
    y_low = y_voltages[0]
    y_high = y_voltages[-1]

    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [
        x_high - half_pixel_size,
        x_low + half_pixel_size,
        y_low - half_pixel_size,
        y_high + half_pixel_size,
    ]

    # convert to kcps
    img_array = (img_array[:] / 1000) / readout_sec

    tool_belt.create_image_figure(
        img_array,
        np.array(img_extent),
        clickHandler=on_click_image,
        title=None,
        color_bar_label="kcps",
        cmin=cmin,
        cmax=cmax,
    )


### Main


def main(
    nv_sig,
    x_range,
    y_range,
    num_steps,
    save_data=True,
    plot_data=True,
    um_scaled=False,
    nv_minus_initialization=False,
    cmin=None,
    cmax=None,
):

    with labrad.connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(
            cxn,
            nv_sig,
            x_range,
            y_range,
            num_steps,
            save_data,
            plot_data,
            um_scaled,
            nv_minus_initialization,
            cmin,
            cmax,
        )

    return img_array, x_voltages, y_voltages


def main_with_cxn(
    cxn,
    nv_sig,
    x_range,
    y_range,
    num_steps,
    save_data=True,
    plot_data=True,
    um_scaled=False,
    nv_minus_initialization=False,
    cmin=None,
    cmax=None,
):

    ### Some initial setup

    tool_belt.reset_cfm(cxn)

    drift = tool_belt.get_drift()
    coords = nv_sig["coords"]
    adjusted_coords = (np.array(coords) + np.array(drift)).tolist()
    x_center, y_center, z_center = adjusted_coords
    optimize.prepare_microscope(cxn, nv_sig, adjusted_coords)

    laser_key = "imaging_laser"
    readout_laser = nv_sig[laser_key]
    tool_belt.set_filter(cxn, nv_sig, laser_key)
    time.sleep(2)
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    # print(readout_power)

    xy_server = positioning.get_pos_xy_server(cxn)

    # Get a couple registry entries
    # See if this setup has finely specified delay times, else just get the
    # one-size-fits-all value.
    dir_path = ["", "Config", "Positioning"]
    cxn.registry.cd(*dir_path)
    _, keys = cxn.registry.dir()
    if "xy_small_response_delay" in keys:
        xy_delay = tool_belt.get_registry_entry(
            cxn, "xy_small_response_delay", dir_path
        )
    else:
        xy_delay = tool_belt.get_registry_entry(cxn, "xy_delay", dir_path)
    # Get the scale in um per unit
    xy_scale = tool_belt.get_registry_entry(cxn, "xy_nm_per_unit", dir_path)
    if xy_scale == -1:
        um_scaled = False
    else:
        xy_scale *= 1000

    total_num_samples = num_steps**2

    ### Load the pulse generator

    pulse_gen = tool_belt.get_server_pulse_gen(cxn)

    readout = nv_sig["imaging_readout_dur"]
    readout_sec = readout / 10**9
    readout_us = readout / 10**3

    if nv_minus_initialization:
        laser_key = "nv-_prep_laser"
        tool_belt.set_filter(cxn, nv_sig, laser_key)
        init = nv_sig["{}_dur".format(laser_key)]
        init_laser = nv_sig[laser_key]
        init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
        seq_args = [
            init,
            readout,
            init_laser,
            init_power,
            readout_laser,
            readout_power,
        ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = pulse_gen.stream_load(
            "charge_initialization-simple_readout.py", seq_args_string
        )
    else:
        seq_args = [
            xy_delay,
            readout,
            readout_laser,
            readout_power,
        ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = pulse_gen.stream_load("simple_readout.py", seq_args_string)
    # print(seq_args)
    period = ret_vals[0]

    ### Set up the xy_server

    x_num_steps = num_steps
    y_num_steps = num_steps
    x_voltages, y_voltages = positioning.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, x_num_steps, y_num_steps
    )
    xy_server.load_stream_xy(x_voltages, y_voltages)

    ### Set up the APD

    counter = tool_belt.get_server_counter(cxn)
    counter.start_tag_stream()

    ### Set up our raw data objects

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = np.empty((x_num_steps, y_num_steps))
    img_array[:] = np.nan
    img_write_pos = []

    ### Set up the image display

    if plot_data:

        kpl.init_kplotlib(font_size=kpl.Size.SMALL, no_latex=True)

        img_array_kcps = np.copy(img_array)

        x_low = min(x_voltages)
        x_high = max(x_voltages)
        y_low = min(y_voltages)
        y_high = max(y_voltages)

        # For the image extent, we need to bump out the min/max x/y by half the
        # pixel size in each direction so that the center of each pixel properly
        # lies at its x/y voltages.
        x_half_pixel = (x_voltages[1] - x_voltages[0]) / 2
        y_half_pixel = (y_voltages[1] - y_voltages[0]) / 2
        img_extent = [
            x_high + x_half_pixel,
            x_low - x_half_pixel,
            y_low - y_half_pixel,
            y_high + y_half_pixel,
        ]

        if um_scaled:
            img_extent = [
                (x_high + x_half_pixel) * xy_scale,
                (x_low - x_half_pixel) * xy_scale,
                (y_low - y_half_pixel) * xy_scale,
                (y_high + y_half_pixel) * xy_scale,
            ]
        # readout_laser_text = kpl.latex_escape(readout_laser)
        title = f"XY image under {readout_laser}, {readout_us} us readout"
        fig = tool_belt.create_image_figure(
            img_array,
            img_extent,
            clickHandler=on_click_image,
            color_bar_label="kcps",
            title=title,
            um_scaled=um_scaled,
        )

    ### Collect the data

    counter.clear_buffer()
    pulse_gen.stream_start(total_num_samples)

    charge_initialization = nv_minus_initialization

    timeout_duration = ((period * (10**-9)) * total_num_samples) + 10
    timeout_inst = time.time() + timeout_duration
    num_read_so_far = 0
    tool_belt.init_safe_stop()

    while num_read_so_far < total_num_samples:

        if (time.time() > timeout_inst) or tool_belt.safe_stop():
            break

        # Read the samples and update the image
        if charge_initialization:
            new_samples = counter.read_counter_modulo_gates(2)
        else:
            new_samples = counter.read_counter_simple()

        num_new_samples = len(new_samples)
        if num_new_samples > 0:
            # If we did charge initialization, subtract out the background
            if charge_initialization:
                new_samples = [max(int(el[0]) - int(el[1]), 0) for el in new_samples]
            # print(img_write_pos)
            populate_img_array(new_samples, img_array, img_write_pos)
            # Inefficient, but easy and readable way of getting kcps
            if plot_data:
                img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
                tool_belt.update_image_figure(fig, img_array_kcps, cmin, cmax)
            num_read_so_far += num_new_samples

    ### Clean up and save the data

    tool_belt.reset_cfm(cxn)
    xy_server.write_xy(x_center, y_center)

    timestamp = tool_belt.get_time_stamp()
    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "drift": drift,
        "x_center": x_center,
        "y_center": y_center,
        "z_center": z_center,
        "x_range": x_range,
        "x_range-units": "V",
        "y_range": y_range,
        "y_range-units": "V",
        "num_steps": num_steps,
        "readout": readout,
        "readout-units": "ns",
        "x_voltages": x_voltages.tolist(),
        "x_voltages-units": "V",
        "y_voltages": y_voltages.tolist(),
        "y_voltages-units": "V",
        "img_array": img_array.astype(int).tolist(),
        "img_array-units": "counts",
    }

    if save_data:
        filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
        tool_belt.save_raw_data(rawData, filePath)
        if plot_data:
            tool_belt.save_figure(fig, filePath)

    return img_array, x_voltages, y_voltages


if __name__ == "__main__":

    pass
