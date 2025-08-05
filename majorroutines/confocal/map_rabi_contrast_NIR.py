# -*- coding: utf-8 -*-
"""
Image the temperature of a sample by zfs thermometry in a raster scan.
Only designed for ensemble.

Created on June 19th, 2022

@author: mccambria
"""

from json import tool
import numpy as np
import utils.tool_belt as tool_belt
import time
import labrad
from majorroutines import rabi
import copy
import matplotlib.pyplot as plt
import analysis.temp_from_resonances as temp_from_resonances



def make_plots(diff_counts, image_extent,imgtitle,cbarlabel):

    fig = tool_belt.create_image_figure(
        diff_counts,
        image_extent,
        title=imgtitle,
        color_bar_label=cbarlabel,
    )
    return fig




def main(
    nv_sig,
    image_range,
    num_steps,
    apd_indices,
    nir_laser_voltage,
    rabi_num_reps,
    rabi_num_runs,
    state,
    uwave_time_range
):

    with labrad.connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(
            cxn,
            nv_sig,
            image_range,
            num_steps,
            apd_indices,
            nir_laser_voltage,
            rabi_num_reps,
            rabi_num_runs,
            state,
            uwave_time_range
        )

    return img_array, x_voltages, y_voltages


def main_with_cxn(
    cxn,
    nv_sig,
    image_range,
    num_steps,
    apd_indices,
    nir_laser_voltage,
    rabi_num_reps,
    rabi_num_runs,
    state,
    uwave_time_range
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

    
    contrast_noNIR_img = gen_blank_square_list(num_steps)
    contrast_NIR_img = gen_blank_square_list(num_steps)
    
    rabi_meas = lambda adj_nv_sig: rabi.main_with_cxn(
        cxn, 
        nv_sig, 
        apd_indices, 
        uwave_time_range, 
        state,
        num_steps, 
        rabi_num_reps, 
        rabi_num_runs, 
        opti_nv_sig = None,
        return_popt=True)

    
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

            cxn_power_supply.output_off()

            time.sleep(1)

            rabi_per_noNIR, rabi_popt_noNIR = rabi_meas(adjusted_nv_sig)
            A0_noNIR = 1 - rabi_popt_noNIR[0]
            contrast_noNIR_img[image_y_ind][adj_x_ind] = A0_noNIR * 2
            
            cxn_power_supply.output_on()
            cxn_power_supply.set_voltage(nir_laser_voltage)

            time.sleep(1)
            
            rabi_per_NIR, rabi_popt_NIR = rabi_meas(adjusted_nv_sig)
            A0_NIR = 1 - rabi_popt_NIR[0]
            contrast_NIR_img[image_y_ind][adj_x_ind] = A0_NIR * 2


        parity *= -1

    cxn_power_supply.output_off()
    
    # Processing
       
    contrast_diff = np.asarray(contrast_NIR_img) - np.asarray(contrast_noNIR_img)
    contrast_percent_diff = ( np.asarray(contrast_NIR_img) - np.asarray(contrast_noNIR_img) ) / np.asarray(contrast_noNIR_img)
       
    
    title =  'change in rabi contrast'  #########this is where I left off. I need to finish writing the routine. 
    fig1 = make_plots(contrast_percent_diff, image_extent,imgtitle=title,cbarlabel='(NIR-noNIR)/noNIR contrast')
    fig2 = make_plots(contrast_diff, image_extent,imgtitle=title,cbarlabel='NIR-noNIR Contrast')
    fig3 = make_plots(contrast_noNIR_img, image_extent,imgtitle=title,cbarlabel='noNIR Contrast')
    fig4 = make_plots(contrast_NIR_img, image_extent,imgtitle=title,cbarlabel='NIR Contrast')
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
        # "charge_initialization": str(nv_minus_initialization),
        # "readout_laser": readout_laser,
        # "readout_time": readout,
        # "readout_laser_power": readout_power,
        "image_range": image_range,
        "image_center_coords": image_center_coords,
        "image_extent": image_extent,
        "num_steps": num_steps,
        "readout-units": "ns",
        "x_voltages": x_voltages_1d.tolist(),
        "y_voltages": y_voltages_1d.tolist(),
        "xy_units": xy_units,
        "diff_contrast_img": contrast_diff.tolist(),
        "contrast_NIR_img": contrast_NIR_img,
        "contrast_noNIR_img": contrast_noNIR_img,
        "contrast_percent_diff_img": contrast_percent_diff.tolist(),
        "diff_counts-units": "kcps",
    }

    filePath1 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_percentdiff")
    filePath2 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_diff")
    filePath3 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_noNIR")
    filePath4 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_NIR")
    tool_belt.save_raw_data(rawData, filePath1)
    tool_belt.save_figure(fig1, filePath1)
    tool_belt.save_figure(fig2, filePath2)
    tool_belt.save_figure(fig3, filePath3)
    tool_belt.save_figure(fig4, filePath4)


# region Run the file

if __name__ == "__main__":

    pass

    # plt.show(block=True)

# endregion
