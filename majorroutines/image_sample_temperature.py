# -*- coding: utf-8 -*-
"""
Image the temperature of a sample by zfs thermometry in a raster scan.
Only designed for ensemble.

Created on June 19th, 2022

@author: mccambria
"""


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


def process_resonances(ref_resonances, signal_resonances):

    ref_zfss = [[el[1] - el[0] for el in row] for row in ref_resonances]
    signal_zfss = [[el[1] - el[0] for el in row] for row in signal_resonances]

    ref_temps = [
        [temp_from_resonances.main(zfs) for zfs in row] for row in ref_zfss
    ]
    signal_temps = [
        [temp_from_resonances.main(zfs) for zfs in row] for row in signal_zfss
    ]

    ref_zfss = np.array(ref_zfss)
    signal_zfss = np.array(signal_zfss)
    ref_temps = np.array(ref_temps)
    signal_temps = np.array(signal_temps)

    diff_temps = signal_temps - ref_temps

    return diff_temps
    # return ref_temps


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


# def main_with_cxn(
#         files,
#         image_range,
#         y_range,
#         num_steps,
#     ):
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

    ref_resonances = gen_blank_square_list()
    ref_files = gen_blank_square_list()
    signal_resonances = gen_blank_square_list()
    signal_files = gen_blank_square_list()

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

    # x_center = 0.0
    # y_center = 0.0
    # z_center = 0

    # Get the voltages for the raster
    x_voltages_1d, y_voltages_1d = tool_belt.calc_image_scan_vals(
        x_center, y_center, image_range, num_steps
    )
    image_extent = tool_belt.calc_image_extent(
        x_center, y_center, image_range, num_steps
    )

    # Start rasterin'

    parity = +1  # Determines x scan direction

    # path_from_nv_data = "pc_hahn/branch_master/pulsed_resonance/2022_06/image_sample_temperature-2022_06_30-crashed"

    for y_ind in range(num_steps):
        y_voltage = y_voltages_1d[y_ind]

        for x_ind in range(num_steps):
            adj_x_ind = x_ind if parity == +1 else -1 - x_ind
            x_voltage = x_voltages_1d[adj_x_ind]

            adjusted_nv_sig = copy.deepcopy(nv_sig)
            adjusted_nv_sig["coords"] = [x_voltage, y_voltage, z_center]

            cxn_power_supply.output_off()

            time.sleep(10)

            res_low, _, file_name_low = pesr_low_lambda(adjusted_nv_sig)
            res_high, _, file_name_high = pesr_high_lambda(adjusted_nv_sig)
            ref_resonances[y_ind][adj_x_ind] = (res_low, res_high)
            ref_files[y_ind][adj_x_ind] = (file_name_low, file_name_high)

            cxn_power_supply.output_on()
            cxn_power_supply.set_voltage(nir_laser_voltage)

            time.sleep(10)

            res_low, _, file_name_low = pesr_low_lambda(adjusted_nv_sig)
            res_high, _, file_name_high = pesr_high_lambda(adjusted_nv_sig)
            signal_resonances[y_ind][adj_x_ind] = (res_low, res_high)
            signal_files[y_ind][adj_x_ind] = (file_name_low, file_name_high)

            # f = files.pop(0)
            # f_name_with_ext = f.split("/")[-1]
            # f_name_wo_ext = f_name_with_ext.split(".")[0]
            # data = tool_belt.get_raw_data(f_name_wo_ext, path_from_nv_data)
            # res_low, error_low = pulsed_resonance.return_res_with_error(data)
            # f = files.pop(0)
            # f_name_with_ext = f.split("/")[-1]
            # f_name_wo_ext = f_name_with_ext.split(".")[0]
            # data = tool_belt.get_raw_data(f_name_wo_ext, path_from_nv_data)
            # res_high, error_high = pulsed_resonance.return_res_with_error(data)
            # ref_resonances[y_ind][adj_x_ind] = (res_low, res_high)

            # f = files.pop(0)
            # f_name_with_ext = f.split("/")[-1]
            # f_name_wo_ext = f_name_with_ext.split(".")[0]
            # data = tool_belt.get_raw_data(f_name_wo_ext, path_from_nv_data)
            # res_low, error_low = pulsed_resonance.return_res_with_error(data)
            # f = files.pop(0)
            # f_name_with_ext = f.split("/")[-1]
            # f_name_wo_ext = f_name_with_ext.split(".")[0]
            # data = tool_belt.get_raw_data(f_name_wo_ext, path_from_nv_data)
            # res_high, error_high = pulsed_resonance.return_res_with_error(data)
            # signal_resonances[y_ind][adj_x_ind] = (res_low, res_high)

        parity *= -1

    cxn_power_supply.output_off()

    # Processing

    diff_temps = process_resonances(ref_resonances, signal_resonances)

    fig = tool_belt.create_image_figure(
        diff_temps,
        image_extent,
        color_bar_label=r"$\mathrm{\Delta}\mathit{T}$ (K)",
    )

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
        "ref_resonances": ref_resonances,
        "ref_files": ref_files,
        "signal_resonances": signal_resonances,
        "signal_files": signal_files,
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

    # home = common.get_nvdata_dir()
    # path = (
    #     home
    #     / "pc_hahn/branch_master/pulsed_resonance/2022_06/image_sample_temperature-2022_06_30-crashed/"
    # )

    # # absolute path to search all text files inside a specific folder
    # file_glob = str(path) + "/*.txt"
    # files = glob.glob(file_glob)
    # for f in files:
    #     print(f)

    # scan_range = 0.05
    # num_steps = 3

    # main_with_cxn(files, scan_range, scan_range, num_steps)

    file_name = "2022_07_03-13_09_30-hopper-search"
    data = tool_belt.get_raw_data(file_name)

    ref_resonances = data["ref_resonances"]
    signal_resonances = data["signal_resonances"]
    nv_sig = data["nv_sig"]
    coords = nv_sig["coords"]
    drift = data["drift"]
    adjusted_coords = (np.array(coords) + np.array(drift)).tolist()
    x_center, y_center, z_center = adjusted_coords
    image_range = data["x_range"]
    num_steps = data["num_steps"]
    image_extent = tool_belt.calc_image_extent(
        x_center, y_center, image_range, num_steps
    )

    diff_temps = process_resonances(ref_resonances, signal_resonances)

    fig = tool_belt.create_image_figure(
        diff_temps,
        image_extent,
        color_bar_label=r"$\mathrm{\Delta}\mathit{T}$ (K)",
    )

    plt.show(block=True)

# endregion
