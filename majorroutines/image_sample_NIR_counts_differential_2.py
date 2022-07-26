# -*- coding: utf-8 -*-
"""
Image the counts differential of a sample with vs without NIR light in a raster scan.
Only designed for ensemble.

Created on July 26th, 2022

@author: cdfox
"""

from json import tool
from black import diff
import numpy as np
import utils.tool_belt as tool_belt
import time
import labrad
import copy
import matplotlib.pyplot as plt
import glob
import utils.common as common

# region Functions

    # def process_resonances(ref_resonances, ref_res_errs,
    #                         sig_resonances, sig_res_errs):

    ref_zfss = [[(el[1] + el[0]) / 2 for el in row] for row in ref_resonances]
    # ref_zfs_errs
    sig_zfss = [[(el[1] + el[0]) / 2 for el in row] for row in sig_resonances]

    ref_temps = [
        [temp_from_resonances.main(zfs) for zfs in row] for row in ref_zfss
    ]
    sig_temps = [
        [temp_from_resonances.main(zfs) for zfs in row] for row in sig_zfss
    ]

    ref_zfss = np.array(ref_zfss)
    sig_zfss = np.array(sig_zfss)
    ref_temps = np.array(ref_temps)
    sig_temps = np.array(sig_temps)

    diff_temps = sig_temps - ref_temps

    return diff_temps


def plot_diff_counts(diff_counts, image_extent):

    fig = tool_belt.create_image_figure(
        diff_counts,
        image_extent,
        color_bar_label="NIR - noNIR Counts (kcps)",
    )
    return fig


# endregion

# region Main


def main(
    nv_sig,
    image_range,
    num_steps,
    apd_indices,
    nir_laser_voltage,
):

    with labrad.connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(
            cxn,
            nv_sig,
            image_range,
            num_steps,
            apd_indices,
            nir_laser_voltage,
        )

    return img_array, x_voltages, y_voltages


def main_with_cxn(
    cxn,
    nv_sig,
    image_range,
    num_steps,
    apd_indices,
    nir_laser_voltage,
):
    
    # Some initial setup

    tool_belt.init_matplotlib()

    tool_belt.reset_cfm(cxn)

    drift = tool_belt.get_drift()
    coords = nv_sig["coords"]
    image_center_coords = (np.array(coords) + np.array(drift)).tolist()
    x_center, y_center, z_center = image_center_coords

    gen_blank_square_list = lambda size: [
        [
            None,
        ]
        * size
        for ind in range(size)
    ]

    diff_counts_img = gen_blank_square_list(num_steps)
    
    laser_key = 'imaging_laser'
    readout_laser = nv_sig[laser_key]
    readout_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
    
    dir_path = ['', 'Config', 'Positioning']
    cxn.registry.cd(*dir_path)
    _, keys = cxn.registry.dir()
    if 'xy_small_response_delay' in keys:
        xy_delay = tool_belt.get_registry_entry(cxn,
                                        'xy_small_response_delay', dir_path)
    else:
        xy_delay = tool_belt.get_registry_entry(cxn, 'xy_delay', dir_path)

    
    cxn_power_supply = cxn.power_supply_mp710087

    # Get the voltages for the raster
    x_voltages_1d, y_voltages_1d = tool_belt.calc_image_scan_vals(
        x_center, y_center, image_range, num_steps
    )
    image_extent = tool_belt.calc_image_extent(
        x_center, y_center, image_range, num_steps
    )
    
    readout = nv_sig['imaging_readout_dur']
    readout_sec = readout / 10**9
    readout_us = readout / 10**3

    # Start rasterin'

    parity = +1  # Determines x scan direction
    adjusted_nv_sig = copy.deepcopy(nv_sig)

    for y_ind in range(num_steps):

        y_voltage = y_voltages_1d[y_ind]

        # We want the image array to match how it'll be displayed (lowest voltages in lower left corner),
        # so we'll need to fill the array from bottom to top since we're starting with the lowest voltages.
        image_y_ind = -1 - y_ind

        for x_ind in range(num_steps):

            adj_x_ind = x_ind if parity == +1 else -1 - x_ind
            x_voltage = x_voltages_1d[adj_x_ind]
            adjusted_nv_sig["coords"] = [x_voltage, y_voltage, z_center]
            
            ### First turn off the NIR laser and readout at this location
            cxn_power_supply.output_off()
            time.sleep(1)
             
            ### Set up pulse streamer and apd
            seq_args = [xy_delay, readout, apd_indices[0], readout_laser, readout_power]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',seq_args_string)
            period = ret_vals[0]
            total_num_samples = 1
            timeout_duration = ((period*(10**-9)) * total_num_samples) + 10 
            timeout_inst = time.time() + timeout_duration
            cxn.apd_tagger.start_tag_stream(apd_indices)
            
            ### Collect then read the data
            cxn.apd_tagger.clear_buffer()
            cxn.pulse_streamer.stream_start(total_num_samples)
            new_samples_noNIR = cxn.apd_tagger.read_counter_simple()            

            ### Now turn on NIR laser and readout at the same location
            cxn_power_supply.output_on()
            cxn_power_supply.set_voltage(nir_laser_voltage)
            time.sleep(1)

            seq_args = [xy_delay, readout, apd_indices[0], readout_laser, readout_power]
            seq_args_string = tool_belt.encode_seq_args(seq_args)
            ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',seq_args_string)
            period = ret_vals[0]
            total_num_samples = 1
            timeout_duration = ((period*(10**-9)) * total_num_samples) + 10 
            timeout_inst = time.time() + timeout_duration
            cxn.apd_tagger.start_tag_stream(apd_indices)
            
            cxn.apd_tagger.clear_buffer()
            cxn.pulse_streamer.stream_start(total_num_samples)
         
            new_samples_NIR = cxn.apd_tagger.read_counter_simple()
            
            ### Now populate the image with the subtracted value
            diff_counts_img[image_y_ind][adj_x_ind] = new_samples_NIR - new_samples_noNIR

        parity *= -1

    cxn_power_supply.output_off()

    # Plot
    fig = plot_diff_counts(diff_counts_img, image_extent)

    # Clean up

    tool_belt.reset_cfm(cxn)
    xy_server = tool_belt.get_xy_server(cxn)
    xy_server.write_xy(x_center, y_center)

    # Save the data

    timestamp = tool_belt.get_time_stamp()
    xy_units = tool_belt.get_registry_entry(
        cxn, "xy_units", ["", "Config", "Positioning"]
    )
    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "nv_sig-units": tool_belt.get_nv_sig_units(),
        "drift": drift,
        "image_range": image_range,
        "image_center_coords": image_center_coords,
        "image_extent": image_extent,
        "num_steps": num_steps,
        "readout-units": "ns",
        "x_voltages": x_voltages_1d.tolist(),
        "y_voltages": y_voltages_1d.tolist(),
        "xy_units": xy_units,
        "diff_counts_img": diff_counts_img,
        "diff_counts-units": "kcps",
    }

    filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_raw_data(rawData, filePath)
    tool_belt.save_figure(fig, filePath)

    return diff_counts_img, x_voltages_1d, y_voltages_1d


# endregion

# region Run the file

if __name__ == "__main__":

    pass

    # plt.show(block=True)

# endregion
