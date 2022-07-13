# -*- coding: utf-8 -*-
"""
Image the temperature of a sample by zfs thermometry in a raster scan.
Only designed for ensemble.

Created on June 19th, 2022

@author: mccambria
"""

from json import tool
from black import diff
import numpy as np
import utils.tool_belt as tool_belt
import time
import labrad
from majorroutines import pulsed_resonance
from utils.tool_belt import States
import copy
import matplotlib.pyplot as plt
import analysis.temp_from_resonances as temp_from_resonances
import glob
import utils.common as common

# region Functions


def process_resonances(ref_resonances, sig_resonances):
    """Deprecated, but still maybe still useful. Use process_res_files instead"""

    ref_zfss = [[(el[1] + el[0]) / 2 for el in row] for row in ref_resonances]
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


def process_res_files(ref_files, sig_files):

    ref_temps = [
        [temp_from_resonances.main_files(pair) for pair in row]
        for row in ref_files
    ]
    sig_temps = [
        [temp_from_resonances.main_files(pair) for pair in row]
        for row in sig_files
    ]

    ref_temps = np.array(ref_temps)
    sig_temps = np.array(sig_temps)
    diff_temps = sig_temps - ref_temps

    return diff_temps


def plot_diff_temps(diff_temps, image_extent):

    fig = tool_belt.create_image_figure(
        diff_temps,
        image_extent,
        color_bar_label=r"$\mathrm{\Delta}\mathit{T}$ (K)",
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
    esr_freq_range,
    esr_num_steps,
    esr_num_reps,
    esr_num_runs,
):

    with labrad.connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(
            cxn,
            nv_sig,
            image_range,
            num_steps,
            apd_indices,
            nir_laser_voltage,
            esr_freq_range,
            esr_num_steps,
            esr_num_reps,
            esr_num_runs,
        )

    return img_array, x_voltages, y_voltages


def main_with_cxn(
    cxn,
    nv_sig,
    image_range,
    num_steps,
    apd_indices,
    nir_laser_voltage,
    esr_freq_range,
    esr_num_steps,
    esr_num_reps,
    esr_num_runs,
):

    # Some initial setup

    tool_belt.init_matplotlib()

    tool_belt.reset_cfm(cxn)

    drift = tool_belt.get_drift()
    coords = nv_sig["coords"]
    image_center_coords = (np.array(coords) + np.array(drift)).tolist()
    x_center, y_center, z_center = image_center_coords

    gen_blank_square_list = lambda num_steps: [
        [
            None,
        ]
        * num_steps
        for ind in range(num_steps)
    ]

    ref_files = gen_blank_square_list()
    sig_files = gen_blank_square_list()

    pesr_low_lambda = lambda adj_nv_sig: pulsed_resonance.state(
        adj_nv_sig,
        apd_indices,
        States.LOW,
        esr_freq_range,
        esr_num_steps,
        esr_num_reps,
        esr_num_runs,
        ret_file_name=True,
    )
    pesr_high_lambda = lambda adj_nv_sig: pulsed_resonance.state(
        adj_nv_sig,
        apd_indices,
        States.HIGH,
        esr_freq_range,
        esr_num_steps,
        esr_num_reps,
        esr_num_runs,
        ret_file_name=True,
    )

    cxn_power_supply = cxn.power_supply_mp710087

    # Get the voltages for the raster

    x_voltages_1d, y_voltages_1d = tool_belt.calc_image_scan_vals(
        x_center, y_center, image_range, num_steps
    )
    image_extent = tool_belt.calc_image_extent(
        x_center, y_center, image_range, num_steps
    )

    # Start rasterin'

    parity = +1  # Determines x scan direction

    for y_ind in range(num_steps):

        y_voltage = y_voltages_1d[y_ind]

        # We want the image array to match how it'll be displayed (lowest voltages in lower left corner),
        # so we'll need to fill the array from bottom to top since we're starting with the lowest voltages.
        image_y_ind = -1 - y_ind

        for x_ind in range(num_steps):

            adj_x_ind = x_ind if parity == +1 else -1 - x_ind
            x_voltage = x_voltages_1d[adj_x_ind]

            adjusted_nv_sig = copy.deepcopy(nv_sig)
            adjusted_nv_sig["coords"] = [x_voltage, y_voltage, z_center]

            cxn_power_supply.output_off()

            time.sleep(10)

            _, _, file_name_low = pesr_low_lambda(adjusted_nv_sig)
            _, _, file_name_high = pesr_high_lambda(adjusted_nv_sig)
            ref_files[image_y_ind][adj_x_ind] = (
                file_name_low,
                file_name_high,
            )

            cxn_power_supply.output_on()
            cxn_power_supply.set_voltage(nir_laser_voltage)

            time.sleep(10)

            _, _, file_name_low = pesr_low_lambda(adjusted_nv_sig)
            _, _, file_name_high = pesr_high_lambda(adjusted_nv_sig)
            sig_files[image_y_ind][adj_x_ind] = (
                file_name_low,
                file_name_high,
            )

        parity *= -1

    cxn_power_supply.output_off()

    # Processing

    diff_temps = process_res_files(ref_files, sig_files)
    fig = plot_diff_temps(diff_temps, image_extent)

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
        "ref_files": ref_files,
        "sig_files": sig_files,
        "diff_temps": diff_temps.astype(float).tolist(),
        "diff_temps-units": "Kelvin",
    }

    filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    tool_belt.save_raw_data(rawData, filePath)
    tool_belt.save_figure(fig, filePath)

    return diff_temps, x_voltages_1d, y_voltages_1d


# endregion

# region Run the file

if __name__ == "__main__":

    file_name = "2022_07_03-13_09_30-hopper-search"
    data = tool_belt.get_raw_data(file_name)

    ref_files = data["ref_files"]
    sig_files = data["sig_files"]
    nv_sig = data["nv_sig"]
    coords = nv_sig["coords"]
    drift = data["drift"]
    x_center, y_center, z_center = tool_belt.adjust_coords_for_drift(
        coords, drift
    )
    image_range = data["image_range"]
    num_steps = data["num_steps"]

    diff_temps = process_res_files(ref_files, sig_files)
    image_extent = tool_belt.calc_image_extent(
        x_center, y_center, image_range, num_steps
    )
    plot_diff_temps(diff_temps, image_extent)

    # plt.show(block=True)

# endregion
