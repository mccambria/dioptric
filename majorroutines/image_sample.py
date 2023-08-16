# -*- coding: utf-8 -*-
"""
Scan over the designated area, collecting counts at each point.
Generate an image of the sample.

Created on April 9th, 2019

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy as np
import time
import labrad
import majorroutines.optimize as optimize
from utils.constants import ControlStyle
from utils import tool_belt as tb
from utils import common
from utils import kplotlib as kpl
from utils import positioning
from scipy import ndimage
from numpy import fft


def circle_mask(radius, center=None):
    h = 512
    w = 512

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center > radius
    return mask


def widefield_process(img_array):
    subt_img = "2023_08_14-14_36_52-johnson-nvref"
    subt_img_data = tb.get_raw_data(subt_img)
    subt_img_array = subt_img_data["img_array"]
    ret_img_array = subt_img_array - img_array

    # lowpass = ndimage.gaussian_filter(img_array, 7)
    # highpass = img_array - lowpass
    # highpass = ndimage.gaussian_filter(highpass, 3)

    lowpass = ndimage.gaussian_filter(img_array, 5)
    proc = img_array - lowpass
    # proc = ndimage.gaussian_filter(proc, 3)

    # highpass = circle_mask(100)
    # return np.abs(highpass)
    # return highpass
    return ret_img_array


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


def main(
    nv_sig,
    x_range,
    y_range,
    num_steps,
    um_scaled=False,
    nv_minus_init=False,
    vmin=None,
    vmax=None,
    scan_type="XY",
    camera_mode=False,
):
    with labrad.connect(username="", password="") as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(
            cxn,
            nv_sig,
            x_range,
            y_range,
            num_steps,
            um_scaled,
            nv_minus_init,
            vmin,
            vmax,
            scan_type,
            camera_mode,
        )

    return img_array, x_voltages, y_voltages


def main_with_cxn(
    cxn,
    nv_sig,
    x_range,
    y_range,
    num_steps,
    um_scaled=False,
    nv_minus_init=False,
    vmin=None,
    vmax=None,
    scan_type="XY",
    camera_mode=False,
):
    ### Some initial setup

    xy_control_style = positioning.get_xy_control_style()

    tb.reset_cfm(cxn)
    x_center, y_center, z_center = positioning.set_xyz_on_nv(cxn, nv_sig)
    optimize.prepare_microscope(cxn, nv_sig)
    xy_server = positioning.get_server_pos_xy(cxn)
    # print(xy_server)
    xyz_server = positioning.get_server_pos_xyz(cxn)
    counter = tb.get_server_counter(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    total_num_samples = num_steps**2

    laser_key = "imaging_laser"
    readout_laser = nv_sig[laser_key]
    tb.set_filter(cxn, nv_sig, laser_key)
    time.sleep(1)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)

    # See if this setup has finely specified delay times, else just get the
    # one-size-fits-all value.
    config = common.get_config_dict()
    config_positioning = config["Positioning"]

    if "xy_small_response_delay" in config_positioning:
        xy_delay = config_positioning["xy_small_response_delay"]
    else:
        xy_delay = config_positioning["xy_delay"]
    z_delay = config_positioning["z_delay"]

    # Get the scale in um per unit
    if um_scaled:
        if "xy_nm_per_unit" in config_positioning:
            xy_scale = config_positioning["xy_nm_per_unit"] * 1000
        if "z_nm_per_unit" in config_positioning:
            z_scale = config_positioning["z_nm_per_unit"] * 1000

    # Use whichever delay is longer
    if (z_delay > xy_delay) and scan_type == "XZ":
        delay = z_delay
    elif (z_delay > xy_delay) and scan_type == "YZ":
        delay = z_delay
    else:
        delay = xy_delay

    xy_units = config_positioning["xy_units"]
    z_units = config_positioning["z_units"]

    if camera_mode:
        cam = cxn.camera_NUVU_hnu512gamma

    ### Load the pulse generator

    readout = nv_sig["imaging_readout_dur"]
    readout_us = readout / 10**3
    readout_sec = readout / 10**9

    if nv_minus_init:
        laser_key = "nv-_prep_laser"
        tb.set_filter(cxn, nv_sig, laser_key)
        init = nv_sig["{}_dur".format(laser_key)]
        init_laser = nv_sig[laser_key]
        init_power = tb.set_laser_power(cxn, nv_sig, laser_key)
        seq_args = [init, readout, init_laser, init_power, readout_laser, readout_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        seq_file = "charge_init-simple_readout.py"
    elif camera_mode:
        seq_args = [xy_delay, readout, readout_laser, readout_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        seq_file = "simple_readout-camera.py"
    else:
        seq_args = [xy_delay, readout, readout_laser, readout_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        seq_file = "simple_readout.py"

    # print(seq_args)
    # return
    ret_vals = pulse_gen.stream_load(seq_file, seq_args_string)
    period = ret_vals[0]

    ### Set up the xy_server (xyz_server if 'xz' scan_type)

    x_num_steps = num_steps
    y_num_steps = num_steps

    if scan_type == "XY":
        ret_vals = positioning.get_scan_grid_2d(
            x_center, y_center, x_range, y_range, x_num_steps, y_num_steps
        )
    elif scan_type == "XZ":
        ret_vals = positioning.get_scan_grid_2d(
            x_center, z_center, x_range, y_range, x_num_steps, y_num_steps
        )
    elif scan_type == "YZ":
        ret_vals = positioning.get_scan_grid_2d(
            y_center, z_center, x_range, y_range, x_num_steps, y_num_steps
        )

    if xy_control_style == ControlStyle.STEP:
        x_positions, y_positions, x_positions_1d, y_positions_1d, extent = ret_vals
        pos_units = "um"

        x_low = x_positions_1d[0]
        x_high = x_positions_1d[x_num_steps - 1]
        y_low = y_positions_1d[0]
        y_high = y_positions_1d[y_num_steps - 1]

        pixel_size = x_positions_1d[1] - x_positions_1d[0]

        ### Set up our raw data objects
        # make an array to save information if the piezo did not reach it's target
        flag_img_array = np.empty((x_num_steps, y_num_steps))
        flag_img_write_pos = []
        # array for dx values
        dx_img_array = np.empty((x_num_steps, y_num_steps))
        dx_img_write_pos = []
        # array for dy values
        dy_img_array = np.empty((x_num_steps, y_num_steps))
        dy_img_write_pos = []

    elif xy_control_style == ControlStyle.STREAM:
        x_voltages, y_voltages, x_voltages_1d, y_voltages_1d, extent = ret_vals
        x_positions_1d, y_positions_1d = x_voltages_1d, y_voltages_1d
        pos_units = "V"
        # xy_server.load_stream_xy(x_voltages, y_voltages)
        if scan_type == "XY":
            xy_server.load_stream_xy(x_voltages, y_voltages)
        elif scan_type == "XZ":
            z_voltages = y_voltages
            y_vals_static = [y_center] * len(x_voltages)
            xyz_server.load_stream_xyz(x_voltages, y_vals_static, z_voltages)
        elif scan_type == "YZ":
            z_voltages = y_voltages
            x_vals_static = [x_center] * len(x_voltages)
            xyz_server.load_stream_xyz(x_vals_static, x_voltages, z_voltages)

    # Initialize img_array and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = np.empty((x_num_steps, y_num_steps))
    img_array[:] = np.nan
    img_array_kcps = np.copy(img_array)
    img_write_pos = []

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)

    if um_scaled:
        extent = [el * xy_scale for el in extent]
        axes_labels = ["um", "um"]
    elif xy_units is not None:
        axes_labels = [xy_units, xy_units]

    title = f"{scan_type} image under {readout_laser}, {readout_us} us readout"

    fig, ax = plt.subplots()
    if not camera_mode:
        kpl.imshow(
            ax,
            img_array_kcps,
            title=title,
            x_label=axes_labels[0],
            y_label=axes_labels[1],
            cbar_label="kcps",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

    ### Collect the data

    if not camera_mode:
        counter.start_tag_stream()
    tb.init_safe_stop()

    if xy_control_style == ControlStyle.STEP:
        dx_list = []
        dy_list = []
        dz_list = []

        for i in range(total_num_samples):
            cur_x_pos = x_positions[i]
            cur_y_pos = y_positions[i]

            if tb.safe_stop():
                break

            if scan_type == "XY":
                flag = xy_server.write_xy(cur_x_pos, cur_y_pos)
            elif scan_type == "XZ" or scan_type == "YZ":
                flag = xyz_server.write_xyz(cur_x_pos, y_center, cur_y_pos)

            # Some diagnostic stuff - checking how far we are from the target pos
            actual_x_pos, actual_y_pos, actual_z_pos = xyz_server.read_xyz()
            dx_list.append((actual_x_pos - cur_x_pos) * 1e3)
            if scan_type == "XY":
                dy_list.append((actual_y_pos - cur_y_pos) * 1e3)
            elif scan_type == "XZ" or scan_type == "YZ":
                cur_z_pos = cur_y_pos
                dy_list.append((actual_z_pos - cur_z_pos) * 1e3)
            # read the counts at this location

            pulse_gen.stream_start(1)

            new_samples = counter.read_counter_simple(1)
            # update the image arrays
            populate_img_array(new_samples, img_array, img_write_pos)
            populate_img_array([flag], flag_img_array, flag_img_write_pos)

            populate_img_array(
                [(actual_x_pos - cur_x_pos) * 1e3], dx_img_array, dx_img_write_pos
            )
            populate_img_array(
                [(actual_y_pos - cur_y_pos) * 1e3], dy_img_array, dy_img_write_pos
            )
            # Either include this in loop so it plots data as it takes it (takes about 2x as long)
            # or put it ourside loop so it plots after data is complete
            img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
            kpl.imshow_update(ax, img_array_kcps, vmin, vmax)

    elif xy_control_style == ControlStyle.STREAM:
        if camera_mode:
            cam.arm()

        pulse_gen.stream_start(total_num_samples)

        if camera_mode:
            img_array = cam.read()
            cam.disarm()

            kpl.imshow(
                ax,
                img_array,
                title=title,
                x_label=axes_labels[0],
                y_label=axes_labels[1],
                cbar_label="Pixel values",
                # extent=extent,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )

        else:
            charge_init = nv_minus_init

            timeout_duration = ((period * (10**-9)) * total_num_samples) + 10
            timeout_inst = time.time() + timeout_duration
            num_read_so_far = 0

            while num_read_so_far < total_num_samples:
                if (time.time() > timeout_inst) or tb.safe_stop():
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
                        new_samples = [
                            max(int(el[0]) - int(el[1]), 0) for el in new_samples
                        ]
                    populate_img_array(new_samples, img_array, img_write_pos)
                    img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
                    kpl.imshow_update(ax, img_array_kcps, vmin, vmax)
                    num_read_so_far += num_new_samples

    if not camera_mode:
        counter.clear_buffer()

    ### Clean up and save the data

    tb.reset_cfm(cxn)
    if scan_type == "XY":
        xy_server.write_xy(x_center, y_center)
    else:
        xyz_server.write_xyz(x_center, y_center, z_center)

    timestamp = tb.get_time_stamp()
    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        # 'nv_sig-units': tb.get_nv_sig_units(),
        "x_center": x_center,
        "y_center": y_center,
        "z_center": z_center,
        "x_range": x_range,
        "x_range-units": "um",
        "y_range": y_range,
        "y_range-units": "um",
        "num_steps": num_steps,
        "scan_type": scan_type,
        "readout": readout,
        "readout-units": "ns",
        "title": title,
        "x_positions_1d": x_positions_1d.tolist(),
        "x_positions_1d-units": pos_units,
        "y_positions_1d": y_positions_1d.tolist(),
        "y_positions_1d-units": pos_units,
        "img_array": img_array.astype(int).tolist(),
        "img_array-units": "counts",
    }

    filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    tb.save_figure(fig, filePath)
    tb.save_raw_data(rawData, filePath)

    return img_array, x_positions_1d, y_positions_1d


if __name__ == "__main__":
    file_name = "2023_08_15-14_34_47-johnson-nvref"
    data = tb.get_raw_data(file_name)
    img_array = np.array(data["img_array"])
    readout = data["readout"]
    img_array_kcps = (img_array / 1000) / (readout * 1e-9)

    # 114.5, 134.5, 206.5, 186.5

    do_presentation = False
    try:
        scan_type = data["scan_type"]
        if scan_type == "XY":
            x_center = data["x_center"]
            y_center = data["y_center"]
            # x_range = data["x_range"]
            # y_range = data["y_range"]
            # x_num_steps = data["num_steps"]
            # y_num_steps = data["num_steps"]

        elif scan_type == "XZ":
            x_center = data["x_center"]
            y_center = data["z_center"]
        elif scan_type == "YZ":
            x_center = data["y_center"]
            y_center = data["z_center"]
    except Exception:
        nv_sig = data["nv_sig"]
        x_center = nv_sig["coords"][0]
        y_center = nv_sig["coords"][1]

    x_range = data["x_range"]
    y_range = data["y_range"]
    num_steps = data["num_steps"]
    num_steps = data["num_steps"]
    xlabel = "X"
    ylabel = "Y"
    if do_presentation:
        pass
        # x_center=0
        # y_center=0
        # with labrad.connect() as cxn:
        #     scale = common.get_registry_entry(
        #         cxn, "xy_nm_per_unit", ["", "Config", "Positioning"])
        # scale=35000
        # x_range = x_range * scale/1000
        # y_range = y_range * scale/1000
        # xlabel = "um"
        # ylabel = "um"

    ret_vals = positioning.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, num_steps, num_steps
    )
    extent = ret_vals[-1]

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    im = kpl.imshow(
        ax,
        img_array,
        # img_array_kcps,
        # title=title,
        x_label=xlabel,
        y_label=ylabel,
        # cbar_label="kcps",
        cbar_label="Pixel values",
        # vmax=55,
        # extent=extent,
        # aspect="auto",
    )
    ax.set_xlim([124.5-15, 124.5+15])
    ax.set_ylim([196.5+15, 196.5-15])

    # Processing tests
    if False:
        fig, ax = plt.subplots()
        process_img_array = widefield_process(img_array)
        im = kpl.imshow(
            ax,
            process_img_array,
            # img_array_kcps,
            # title=title,
            x_label=xlabel,
            y_label=ylabel,
            # cbar_label="kcps",
            cbar_label="Pixel values",
            # vmax=55,
            # extent=extent,
            # aspect="auto",
        )

    plt.show(block=True)
