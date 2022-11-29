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


# %% Main


def main(
    nv_sig,
    x_range,
    y_range,
    num_steps,
    apd_indices,
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
            apd_indices,
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
    apd_indices,
    save_data=True,
    plot_data=True,
    um_scaled=False,
    nv_minus_initialization=False,
    cmin=None,
    cmax=None,
):

    # %% Some initial setup

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

    if x_range != y_range:
        raise RuntimeError("x and y resolutions must match for now.")

    xy_server = tool_belt.get_xy_server(cxn)

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

    # %% Load the PulseStreamer

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
            apd_indices[0],
            init_laser,
            init_power,
            readout_laser,
            readout_power,
        ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(
            "charge_initialization-simple_readout.py", seq_args_string
        )
    else:
        seq_args = [
            xy_delay,
            readout,
            apd_indices[0],
            readout_laser,
            readout_power,
        ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(
            "simple_readout.py", seq_args_string
        )
    # print(seq_args)
    period = ret_vals[0]

    # %% Set up the xy_server

    x_voltages, y_voltages = xy_server.load_sweep_scan_xy(
        x_center, y_center, x_range, y_range, num_steps, period
    )

    # return
    x_num_steps = len(x_voltages)
    x_low = x_voltages[0]
    x_high = x_voltages[x_num_steps - 1]
    y_num_steps = len(y_voltages)
    y_low = y_voltages[0]
    y_high = y_voltages[y_num_steps - 1]

    pixel_size = x_voltages[1] - x_voltages[0]

    # %% Set up the APD

    cxn.apd_tagger.start_tag_stream(apd_indices)

    # %% Set up our raw data objects

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = np.empty((x_num_steps, y_num_steps))
    img_array[:] = np.nan
    img_write_pos = []

    # %% Set up the image display

    if plot_data:

        kpl.init_kplotlib(font_size="small", no_latex=True)

        img_array_kcps = np.copy(img_array)

        # For the image extent, we need to bump out the min/max x/y by half the
        # pixel size in each direction so that the center of each pixel properly
        # lies at its x/y voltages.
        half_pixel_size = pixel_size / 2
        img_extent = [
            x_high + half_pixel_size,
            x_low - half_pixel_size,
            y_low - half_pixel_size,
            y_high + half_pixel_size,
        ]

        # img_extent = tool_belt.calc_image_extent(x_center, y_center, x_range, num_steps) # assumes square image

        if um_scaled:
            img_extent = [
                (x_high + half_pixel_size) * xy_scale,
                (x_low - half_pixel_size) * xy_scale,
                (y_low - half_pixel_size) * xy_scale,
                (y_high + half_pixel_size) * xy_scale,
            ]
        # readout_laser_text = kpl.latex_escape(readout_laser)
        title = f"Confocal scan, {readout_laser}, {readout_us} us readout"
        fig = tool_belt.create_image_figure(
            img_array,
            img_extent,
            clickHandler=on_click_image,
            color_bar_label="kcps",
            title=title,
            um_scaled=um_scaled,
        )

    # %% Collect the data
    cxn.apd_tagger.clear_buffer()
    cxn.pulse_streamer.stream_start(total_num_samples)

    timeout_duration = ((period * (10**-9)) * total_num_samples) + 10
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
            new_samples = cxn.apd_tagger.read_counter_modulo_gates(2)
        else:
            new_samples = cxn.apd_tagger.read_counter_simple()

        #        print(new_samples)
        num_new_samples = len(new_samples)

        if num_new_samples > 0:

            # If we did charge initialization, subtract out the background
            if charge_initialization:
                new_samples = [
                    max(int(el[0]) - int(el[1]), 0) for el in new_samples
                ]
            # print(img_write_pos)
            populate_img_array(new_samples, img_array, img_write_pos)
            # This is a horribly inefficient way of getting kcps, but it
            # is easy and readable and probably fine up to some resolution
            if plot_data:
                img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
                tool_belt.update_image_figure(fig, img_array_kcps, cmin, cmax)
            num_read_so_far += num_new_samples

    # %% Clean up

    tool_belt.reset_cfm(cxn)

    # Return to center
    xy_server.write_xy(x_center, y_center)

    # %% Save the data

    timestamp = tool_belt.get_time_stamp()
    # print(nv_sig['coords'])
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


# %% Run the file


if __name__ == "__main__":

    # file_name = '2022_09_08-13_02_34-rubin-nv1_2022_08_10'
    # # file_name = '2022_08_18-15_37_38-hopper-search'
    # # file_name = '2022_10_18-17_16_48-siena-nv_search'
    # # file_name = '2022_10_06-18_33_05-siena-nv1_10_06_2022'
    # data = tool_belt.get_raw_data(file_name)
    # # img = data['img_array']
    # # y = data['y_voltages']
    # # x = data['x_voltages']
    # # import matplotlib.pylab as plt
    # # plt.figure()
    # # print(np.array(x)[30])
    # # d = np.array(img)[15:45,30]
    # # ys = np.array(y)[15:45]
    # # plt.plot(ys,d/max(d))
    # # plt.xlabel('y [V]')
    # # plt.ylabel('counts/max(counts)')
    # # plt.show()
    # scale = 83

    # replot_for_presentation(file_name, scale, centered_at_0 = True)

    # replot_for_analysis(file_name,)
    # 0, 70)

    ### Simple

    wu_files = [
        "2022_11_23-16_31_52-wu-nvref_zfs_vs_t",
        "2022_11_23-16_37_38-wu-nvref_zfs_vs_t",
        "2022_11_23-16_43_23-wu-nvref_zfs_vs_t",
        "2022_11_23-16_49_09-wu-nvref_zfs_vs_t",
        "2022_11_23-16_54_55-wu-nvref_zfs_vs_t",
        "2022_11_23-17_00_41-wu-nvref_zfs_vs_t",
        "2022_11_23-17_06_27-wu-nvref_zfs_vs_t",
        "2022_11_23-17_12_12-wu-nvref_zfs_vs_t",
        "2022_11_23-17_17_58-wu-nvref_zfs_vs_t",
        "2022_11_23-17_23_44-wu-nvref_zfs_vs_t",
        "2022_11_23-17_29_30-wu-nvref_zfs_vs_t",
        "2022_11_23-17_35_15-wu-nvref_zfs_vs_t",
        "2022_11_23-17_41_02-wu-nvref_zfs_vs_t",
        "2022_11_23-17_46_48-wu-nvref_zfs_vs_t",
        "2022_11_23-17_52_34-wu-nvref_zfs_vs_t",
        "2022_11_23-17_58_20-wu-nvref_zfs_vs_t",
        "2022_11_23-18_04_06-wu-nvref_zfs_vs_t",
        "2022_11_23-18_09_52-wu-nvref_zfs_vs_t",
        "2022_11_23-18_15_38-wu-nvref_zfs_vs_t",
        "2022_11_23-18_21_23-wu-nvref_zfs_vs_t",
        "2022_11_23-18_27_09-wu-nvref_zfs_vs_t",
        "2022_11_23-18_32_55-wu-nvref_zfs_vs_t",
        "2022_11_23-18_38_40-wu-nvref_zfs_vs_t",
        "2022_11_23-18_44_25-wu-nvref_zfs_vs_t",
        "2022_11_23-18_50_11-wu-nvref_zfs_vs_t",
    ]
    
    for ind in range(len(wu_files)):

        file_name = wu_files[ind]
        data = tool_belt.get_raw_data(file_name)
        img_array = np.array(data["img_array"], dtype=float)
        readout_sec = data["readout"] * 1e-9
        img_array_kcps = np.copy(img_array)
        img_array_kcps[:] *= 1 / (1000 * readout_sec)
        x_range = data["x_range"]
        num_steps = data["num_steps"]
        drift = data["drift"]
        coords = data["nv_sig"]["coords"]
        adjusted_coords = (np.array(coords) + np.array(drift)).tolist()
        x_center, y_center, z_center = adjusted_coords
        img_extent = tool_belt.calc_image_extent(
            x_center, y_center, x_range, num_steps
        )

        kpl.init_kplotlib()

        img_array_kcps = cv2.GaussianBlur(img_array_kcps, (5, 5), 2)
        kernel = np.array(
            [
                [0, -1, -1, -1, 0],
                [-1, 1, 1, 1, -1],
                [-1, 1, 4, 1, -1],
                [-1, 1, 1, 1, -1],
                [0, -1, -1, -1, 0],
            ]
        )
        kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
        img_array_kcps = cv2.filter2D(img_array_kcps, -1, kernel)

        fig = tool_belt.create_image_figure(
            img_array_kcps,
            img_extent,
            color_bar_label="kcps",
            title=f"{kpl.tex_escape(file_name)} replot",
            cmin=0,
        )

        path = (
            common.get_nvdata_dir()
            / f"paper_materials/zfs_temp_dep/figures/composite_wu/processed/{file_name}.svg"
        )
        tool_belt.save_figure(fig, path)

        # plt.show(block=True)
