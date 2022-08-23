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
import os


def NIR_offon(cxn,ind,nir_laser_voltage):
    cxn_power_supply = cxn.power_supply_mp710087
    if ind == 1:
        cxn_power_supply.output_on()
        cxn_power_supply.set_voltage(nir_laser_voltage)
    elif ind == 0:
        cxn_power_supply.output_off()

def plot_diff_counts(diff_counts, image_extent,imgtitle,cbarlabel):

    fig = tool_belt.create_image_figure(
        diff_counts,
        image_extent,
        title=imgtitle,
        color_bar_label=cbarlabel,
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
    nv_minus_initialization=False
):

    with labrad.connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(
            cxn,
            nv_sig,
            image_range,
            num_steps,
            apd_indices,
            nir_laser_voltage,
            nv_minus_initialization
        )

    return img_array, x_voltages, y_voltages


def main_with_cxn(
    cxn,
    nv_sig,
    image_range,
    num_steps,
    apd_indices,
    nir_laser_voltage,
    nv_minus_initialization
):
    
    # Some initial setup
    tool_belt.init_safe_stop()
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
    counts_noNIR_img = gen_blank_square_list(num_steps)
    counts_NIR_img = gen_blank_square_list(num_steps)
    percent_diff_counts_img = gen_blank_square_list(num_steps)
    
    if nv_minus_initialization:
        readout_laser_key = 'charge_readout_laser' ## CF: correct one?
        readout_laser = nv_sig[readout_laser_key]
        readout = nv_sig['charge_readout_dur']
        readout_power = tool_belt.set_laser_power(cxn, nv_sig, readout_laser_key)
    else:
        readout_laser_key = 'imaging_laser'
        readout_laser = nv_sig[readout_laser_key]
        readout = nv_sig['imaging_readout_dur']
        readout_power = tool_belt.set_laser_power(cxn, nv_sig, readout_laser_key)
    
    
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
    
    readout_sec = readout / 10**9

    ### Start rasterin'

    parity = +1  # Determines x scan direction
    adjusted_nv_sig = copy.deepcopy(nv_sig)

    sleep_time = 1
    # print("expected run time: ", ((sleep_time*2+.7)*num_steps**2)/3600,'hours')
    
    
    for ind in range(2):
        
        NIR_offon(cxn,ind,nir_laser_voltage)
        time.sleep(sleep_time)
        
        for y_ind in range(num_steps):
    
            y_voltage = y_voltages_1d[y_ind]
    
            # We want the image array to match how it'll be displayed (lowest voltages in lower left corner),
            # so we'll need to fill the array from bottom to top since we're starting with the lowest voltages.
            image_y_ind = -1 - y_ind
            
            for x_ind in range(num_steps):
                if tool_belt.safe_stop():
                    break
    
                adj_x_ind = x_ind if parity == +1 else -1 - x_ind
                x_voltage = x_voltages_1d[adj_x_ind]
                adjusted_nv_sig["coords"] = [x_voltage, y_voltage, z_center]
                
                tool_belt.set_xyz(cxn, adjusted_nv_sig["coords"])
                
                new_samples = []
                
                    
                if nv_minus_initialization:
                    laser_key = "nv-_prep_laser"
                    tool_belt.set_filter(cxn, nv_sig, laser_key)
                    init = nv_sig['{}_dur'.format(laser_key)]
                    init_laser = nv_sig[laser_key]
                    init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
                    seq_args = [init, readout, apd_indices[0], init_laser, init_power, readout_laser, readout_power]
                    seq_args_string = tool_belt.encode_seq_args(seq_args)
                    ret_vals = cxn.pulse_streamer.stream_load('charge_initialization-simple_readout.py',seq_args_string)
                    
                else:
                    seq_args = [xy_delay, readout, apd_indices[0], readout_laser, readout_power]
                    seq_args_string = tool_belt.encode_seq_args(seq_args)
                    ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',seq_args_string)

                period = ret_vals[0]
                if (x_ind+y_ind+ind)==0:
                    print("expected run time: ", round( (num_steps**2 * 2 * (period * 10**(-9)) ) / 60 ,2) , 'minutes')
                total_num_samples = 1
                timeout_duration = ((period*(10**-9)) * total_num_samples) + 10
                timeout_inst = time.time() + timeout_duration
                cxn.apd_tagger.start_tag_stream(apd_indices)
                
                ### Collect then read the data
                cxn.apd_tagger.clear_buffer()
                cxn.pulse_streamer.stream_start(total_num_samples)
                new_samples =  cxn.apd_tagger.read_counter_simple(total_num_samples) 
                # print(x_voltage,y_voltage,new_samples)
                # new_samples_noNIR = new_samples[0]
                # new_samples_NIR = new_samples[1]    
                    
                    
                #%% in this collapsed region is the commented out old version of the code before the loop above
                '''
                ### Now turn on NIR laser and readout at the same location
                cxn_power_supply.output_on()
                cxn_power_supply.set_voltage(nir_laser_voltage)
                time.sleep(sleep_time)
    
                if nv_minus_initialization:
                    laser_key = "nv-_prep_laser"
                    tool_belt.set_filter(cxn, nv_sig, laser_key)
                    init = nv_sig['{}_dur'.format(laser_key)]
                    init_laser = nv_sig[laser_key]
                    init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
                    seq_args = [init, readout, apd_indices[0], init_laser, init_power, readout_laser, readout_power]
                    seq_args_string = tool_belt.encode_seq_args(seq_args)
                    ret_vals = cxn.pulse_streamer.stream_load('charge_initialization-simple_readout.py',seq_args_string)
                    
                else:
                    seq_args = [xy_delay, readout, apd_indices[0], readout_laser, readout_power]
                    seq_args_string = tool_belt.encode_seq_args(seq_args)
                    ret_vals = cxn.pulse_streamer.stream_load('simple_readout.py',seq_args_string)
    
                period = ret_vals[0]
                if i == 1:
                    print("expected run time: ", (period*10**(-9))/60, 'minutes')
                total_num_samples = 1
                timeout_duration = ((period*(10**-9)) * total_num_samples) + 10
                timeout_inst = time.time() + timeout_duration
                cxn.apd_tagger.start_tag_stream(apd_indices)
                
                ### Collect then read the data
                cxn.apd_tagger.clear_buffer()
                cxn.pulse_streamer.stream_start(total_num_samples)
                new_samples_NIR = cxn.apd_tagger.read_counter_simple(total_num_samples)
                
    
                
                ### First turn off the NIR laser and readout at this location
                cxn_power_supply.output_off()
                time.sleep(sleep_time)
                 
                ### Set up pulse streamer and apd
                if nv_minus_initialization:
                    laser_key = "nv-_prep_laser"
                    tool_belt.set_filter(cxn, nv_sig, laser_key)
                    init = nv_sig['{}_dur'.format(laser_key)]
                    init_laser = nv_sig[laser_key]
                    init_power = tool_belt.set_laser_power(cxn, nv_sig, laser_key)
                    seq_args = [init, readout, apd_indices[0], init_laser, init_power, readout_laser, readout_power]
                    seq_args_string = tool_belt.encode_seq_args(seq_args)
                    ret_vals = cxn.pulse_streamer.stream_load('charge_initialization-simple_readout.py',seq_args_string)
                else:
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
                new_samples_noNIR = cxn.apd_tagger.read_counter_simple(total_num_samples)
                '''
                #%%
                
                ### Now populate the image with the subtracted value
                if ind == 0:
                    counts_noNIR_img[image_y_ind][adj_x_ind] = np.int32(new_samples[0]) / (readout_sec) / 1000
                    
                elif ind == 1:
                    counts_NIR_img[image_y_ind][adj_x_ind] = np.int32(new_samples[0]) / (readout_sec) / 1000
                    
                
                # diff_counts_img[image_y_ind][adj_x_ind] = (np.int32(new_samples_NIR[0]) - np.int32(new_samples_noNIR[0]))/(readout_sec)/1000
                # percent_diff_counts_img[image_y_ind][adj_x_ind] = (np.int32(new_samples_NIR[0]) - np.int32(new_samples_noNIR[0]))/np.int32(new_samples_noNIR[0])
                # print(i,new_samples_NIR,'',new_samples_noNIR)
                # # print(type(new_samples_noNIR[0]))
                # print(diff_counts_img)
                # print(time.time()-a)
    
            parity *= -1
    
    diff_counts_img = np.asarray(counts_NIR_img) - np.asarray(counts_noNIR_img)
    percent_diff_counts_img = ( np.asarray(counts_NIR_img) - np.asarray(counts_noNIR_img) ) / np.asarray(counts_noNIR_img)
    
    print(np.asarray(counts_noNIR_img))


    cxn_power_supply.output_off()

    # Plot        
    title = r'{}, {} ms readout'.format(readout_laser, readout_sec*1000)

    fig1 = plot_diff_counts(percent_diff_counts_img, image_extent,imgtitle=title,cbarlabel='(NIR-noNIR)/noNIR Counts')
    fig2 = plot_diff_counts(diff_counts_img, image_extent,imgtitle=title,cbarlabel='NIR-noNIR Counts (kcps)')
    fig3 = plot_diff_counts(counts_noNIR_img, image_extent,imgtitle=title,cbarlabel='noNIR Counts (kcps)')
    fig4 = plot_diff_counts(counts_NIR_img, image_extent,imgtitle=title,cbarlabel='NIR Counts (kcps)')
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
        "charge_initialization": str(nv_minus_initialization),
        "readout_laser": readout_laser,
        "readout_time": readout,
        "readout_laser_power": readout_power,
        "image_range": image_range,
        "image_center_coords": image_center_coords,
        "image_extent": image_extent,
        "num_steps": num_steps,
        "readout-units": "ns",
        "x_voltages": x_voltages_1d.tolist(),
        "y_voltages": y_voltages_1d.tolist(),
        "xy_units": xy_units,
        "diff_counts_img": diff_counts_img.tolist(),
        "counts_NIR_img": counts_NIR_img,
        "counts_noNIR_img": counts_noNIR_img,
        "percentdiff_counts_img": percent_diff_counts_img.tolist(),
        "diff_counts-units": "kcps",
    }
    if nv_minus_initialization:
        filePath1 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_pulsed_percentdiff")
        filePath2 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_pulsed_diff")
        filePath3 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_pulsed_noNIR")
        filePath4 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_pulsed_NIR")
    else:
        filePath1 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_percentdiff")
        filePath2 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_diff")
        filePath3 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_noNIR")
        filePath4 = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"]+"_NIR")
    tool_belt.save_raw_data(rawData, filePath1)
    tool_belt.save_figure(fig1, filePath1)
    tool_belt.save_figure(fig2, filePath2)
    tool_belt.save_figure(fig3, filePath3)
    tool_belt.save_figure(fig4, filePath4)

    return diff_counts_img, x_voltages_1d, y_voltages_1d


# endregion

# region Run the file

if __name__ == "__main__":

    pass

    # plt.show(block=True)

# endregion
# -*- coding: utf-8 -*-
