# -*- coding: utf-8 -*-
"""
Scan over the designated area, collecting counts at each point.
Generate an image of the sample.

Created on April 9th, 2019

@author: mccambria
"""


import time
from enum import Enum, auto

import labrad
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import majorroutines.targeting as targeting
from utils import common
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CollectionMode, CountFormat, PosControlMode, VirtualLaserKey


class ScanAxes(Enum):
    XY = auto()
    XZ = auto()
    YZ = auto()


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
    nv_sig, range_1, range_2, num_steps, nv_minus_init=False, scan_axes=ScanAxes.XY
):
    with common.labrad_connect() as cxn:
        return main_with_cxn(
            cxn, nv_sig, range_1, range_2, num_steps, nv_minus_init, scan_axes
        )


def main_with_cxn(
    cxn, nv_sig, range_1, range_2, num_steps, nv_minus_init=False, scan_axes=ScanAxes.XY
):
    ### Some initial setup

    config = common.get_config_dict()
    config_positioning = config["Positioning"]
    collection_mode = config["collection_mode"]
    xy_control_mode = pos.get_xy_control_mode()
    z_control_mode = pos.get_z_control_mode()
    if scan_axes == ScanAxes.XY:
        control_mode = xy_control_mode
    elif (
        xy_control_mode == PosControlMode.STREAM
        and z_control_mode == PosControlMode.STREAM
    ):
        control_mode = PosControlMode.STREAM
    else:
        control_mode = PosControlMode.STEP

    tb.reset_cfm(cxn)
    center_coords = pos.adjust_coords_for_drift(nv_sig["coords"])
    x_center, y_center, z_center = center_coords
    targeting.pos.set_xyz_on_nv(cxn, nv_sig)
    pos_server = (
        pos.get_server_pos_xy(cxn)
        if scan_axes == ScanAxes.XY
        else pos.get_server_pos_xyz(cxn)
    )
    if collection_mode == CollectionMode.COUNTER:
        counter = tb.get_server_counter(cxn)
    elif collection_mode == CollectionMode.CAMERA:
        camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)

    laser_key = VirtualLaserKey.IMAGING
    laser_dict = nv_sig[laser_key]
    readout_laser = laser_dict["name"]
    tb.set_filter(cxn, nv_sig, laser_key)
    readout_power = tb.set_laser_power(cxn, nv_sig, laser_key)

    xy_delay = config_positioning["xy_delay"]
    z_delay = config_positioning["z_delay"]
    scanning_z = scan_axes in [ScanAxes.XZ, ScanAxes.YZ]
    delay = max(xy_delay, z_delay) if scanning_z else xy_delay

    xy_units = config_positioning["xy_units"]
    z_units = config_positioning["z_units"]
    axis_1_units = xy_units
    axis_2_units = xy_units if scan_axes == ScanAxes.XY else z_units

    # Only support square grids at the moment
    num_steps_1 = num_steps
    num_steps_2 = num_steps
    total_num_samples = num_steps_1 * num_steps_2

    ### Load the pulse generator

    readout = laser_dict["readout_dur"]
    readout_us = readout / 10**3
    readout_sec = readout / 10**9

    if collection_mode == CollectionMode.CAMERA:
        seq_args = [delay, readout, readout_laser, readout_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        seq_file = "widefield-simple_readout.py"
    elif collection_mode == CollectionMode.COUNTER:
        if nv_minus_init:
            laser_key = "nv-_prep_laser"
            tb.set_filter(cxn, nv_sig, laser_key)
            init = nv_sig["{}_dur".format(laser_key)]
            init_laser = nv_sig[laser_key]
            init_power = tb.set_laser_power(cxn, nv_sig, laser_key)
            seq_args = [
                init,
                readout,
                init_laser,
                init_power,
                readout_laser,
                readout_power,
            ]
            seq_args_string = tb.encode_seq_args(seq_args)
            seq_file = "charge_init-simple_readout.py"
        else:
            seq_args = [delay, readout, readout_laser, readout_power]
            seq_args_string = tb.encode_seq_args(seq_args)
            seq_file = "simple_readout.py"

    # print(seq_file)
    # print(seq_args)
    # return
    ret_vals = pulse_gen.stream_load(seq_file, seq_args_string)
    period = ret_vals[0]

    ### Set up the xy_server (xyz_server if 'xz' scan_axes)

    if scan_axes == ScanAxes.XY:
        center_1 = x_center
        center_2 = y_center
    elif scan_axes == ScanAxes.XZ:
        center_1 = x_center
        center_2 = z_center
    elif scan_axes == ScanAxes.YZ:
        center_1 = y_center
        center_2 = z_center
    ret_vals = pos.get_scan_grid_2d(
        center_1, center_2, range_1, range_2, num_steps_1, num_steps_2
    )
    coords_1, coords_2, coords_1_1d, coords_2_1d, extent = ret_vals
    num_pixels = num_steps_1 * num_steps_2

    if control_mode == PosControlMode.STREAM:
        if scan_axes == ScanAxes.XY:
            pos_server.load_stream_xy(coords_1, coords_2)
        elif scan_axes == ScanAxes.XZ:
            y_vals_static = [y_center] * num_pixels
            pos_server.load_stream_xyz(coords_1, y_vals_static, coords_2)
        elif scan_axes == ScanAxes.YZ:
            x_vals_static = [x_center] * num_pixels
            pos_server.load_stream_xyz(x_vals_static, coords_1, coords_2)

    # Initialize tracking variables that will be populated as the image is collected in
    # a scanning configuration, e.g. with an APD as opposed to a camera
    count_format = config["count_format"]
    if collection_mode != CollectionMode.CAMERA:
        img_array = np.empty((num_steps_1, num_steps_2))
        # matplotlib will show nothing for NaN, instead of 0 or a random value
        img_array[:] = np.nan
        img_write_pos = []
        if count_format == CountFormat.KCPS:
            img_array_kcps = np.copy(img_array)

    ### Set up the image display

    kpl.init_kplotlib(font_size=kpl.Size.SMALL)
    if collection_mode == CollectionMode.CAMERA:
        hor_label = "X"
        ver_label = "Y"
    else:
        hor_label = xy_units
        ver_label = xy_units if scan_axes == ScanAxes.XY else z_units
    if count_format == CountFormat.RAW:
        cbar_label = "Counts"
    if count_format == CountFormat.KCPS:
        cbar_label = "Kcps"
    title = f"{scan_axes.name} image under {readout_laser}, {readout_us} us readout"
    imshow_extent = None if collection_mode == CollectionMode.CAMERA else extent
    imshow_kwargs = {
        "title": title,
        "x_label": hor_label,
        "y_label": ver_label,
        "cbar_label": cbar_label,
        "extent": imshow_extent,
    }

    fig, ax = plt.subplots()

    ### Collect the data

    if collection_mode == CollectionMode.COUNTER:
        # Show blank image to be filled
        kpl.imshow(ax, img_array, **imshow_kwargs)
        counter.start_tag_stream()
    tb.init_safe_stop()

    if control_mode == PosControlMode.STEP:
        for ind in range(total_num_samples):
            if tb.safe_stop():
                break

            # Write
            cur_coord_1 = coords_1_1d[ind]
            cur_coord_2 = coords_2_1d[ind]
            if scan_axes == ScanAxes.XY:
                pos_server.write_xy(cur_coord_1, cur_coord_2)
            elif scan_axes == ScanAxes.XZ:
                pos_server.write_xyz(cur_coord_1, y_center, cur_coord_2)
            elif scan_axes == ScanAxes.XZ:
                pos_server.write_xyz(x_center, cur_coord_1, cur_coord_2)

            # Read
            pulse_gen.stream_start(1)
            new_samples = counter.read_counter_simple(1)
            populate_img_array(new_samples, img_array, img_write_pos)
            if count_format == CountFormat.RAW:
                kpl.imshow_update(ax, img_array)
            elif count_format == CountFormat.KCPS:
                img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
                kpl.imshow_update(ax, img_array_kcps)

    elif control_mode == PosControlMode.STREAM:
        if collection_mode == CollectionMode.CAMERA:
            camera.arm()

        pulse_gen.stream_start(total_num_samples)

        if collection_mode == CollectionMode.CAMERA:
            img_array = camera.read()
            camera.disarm()
            if count_format == CountFormat.RAW:
                kpl.imshow(ax, img_array, **imshow_kwargs)
            elif count_format == CountFormat.KCPS:
                img_array_kcps = (np.copy(img_array) / 1000) / readout_sec
                kpl.imshow(ax, img_array_kcps, **imshow_kwargs)

        elif collection_mode == CollectionMode.COUNTER:
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
                    if count_format == CountFormat.RAW:
                        kpl.imshow_update(ax, img_array)
                    elif count_format == CountFormat.KCPS:
                        img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
                        kpl.imshow_update(ax, img_array_kcps)
                    num_read_so_far += num_new_samples

    if collection_mode == CollectionMode.COUNTER:
        counter.clear_buffer()

    ### Clean up and save the data

    tb.reset_cfm(cxn)
    pos.set_xyz(cxn, center_coords)

    timestamp = tb.get_time_stamp()
    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "x_center": x_center,
        "y_center": y_center,
        "range_1": z_center,
        "x_range": range_1,
        "range_2": range_2,
        "num_steps": num_steps,
        "extent": extent,
        "scan_axes": scan_axes,
        "readout": readout,
        "readout-units": "ns",
        "title": title,
        "coords_1_1d": coords_1_1d,
        "coords_1_1d-units": axis_1_units,
        "coords_2_1d": coords_1_1d,
        "coords_2_1d-units": axis_2_units,
        "img_array": img_array.astype(int),
        "img_array-units": "counts",
    }

    filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
    tb.save_figure(fig, filePath)
    tb.save_raw_data(rawData, filePath)

    return img_array, coords_1_1d, coords_2_1d


if __name__ == "__main__":
    file_name = "2023_09_11-13_52_01-johnson-nvref"

    data = tb.get_raw_data(file_name)
    img_array = np.array(data["img_array"])
    readout = data["readout"]
    img_array_kcps = (img_array / 1000) / (readout * 1e-9)
    extent = data["extent"] if "extent" in data else None

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    im = kpl.imshow(ax, img_array, cbar_label="ADUs")
    ax.set_xticks(range(0, 501, 100))
    # im = kpl.imshow(ax, img_array_kcps, extent=extent)
    # ax.set_xlim([124.5 - 15, 124.5 + 15])
    # ax.set_ylim([196.5 + 15, 196.5 - 15])

    # plot_coords = [
    #     [183.66, 201.62],
    #     [177.28, 233.34],
    #     [237.42, 314.84],
    #     [239.56, 262.84],
    #     [315.58, 203.56],
    # ]
    # cal_coords = [
    #     [139.5840657600651, 257.70994378810946],
    #     [324.4796398557366, 218.27466265286117],
    # ]
    # for coords in plot_coords:
    #     ax.plot(*coords, color="blue", marker="o", markersize=3)
    # for coords in cal_coords:
    #     ax.plot(*coords, color="green", marker="o", markersize=3)

    plt.show(block=True)
