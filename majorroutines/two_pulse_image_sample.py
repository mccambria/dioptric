# -*- coding: utf-8 -*-
"""
image_sample but with two pulses at each pixel. Not included in control panel

Created June 2020

@author: agardill
"""


import numpy
import utils.tool_belt as tool_belt
import time
import labrad
from majorroutines import image_sample


# %% Main


def main(nv_sig, x_range, y_range, num_steps,  apd_indices, init_pulse_time, readout,
                           init_color_ind, read_color_ind,
         save_data=True, plot_data=True, continuous=False):

    with labrad.connect() as cxn:
        img_array, x_voltages, y_voltages = main_with_cxn(cxn, nv_sig, x_range,
                      y_range, num_steps,
                      apd_indices,  init_pulse_time,readout,
                           init_color_ind, read_color_ind, save_data, plot_data, continuous)

    return img_array, x_voltages, y_voltages

def main_with_cxn(cxn, nv_sig, x_range, y_range, num_steps,
                  apd_indices, init_pulse_time, readout,
                           init_color_ind, read_color_ind,  save_data=True,
                  plot_data=True, continuous=False):

    # %% Some initial setup
    tool_belt.reset_cfm(cxn)
    color_filter = nv_sig['color_filter']
    cxn.filter_slider_ell9k_color.set_filter(color_filter)


    shared_params = tool_belt.get_shared_parameters_dict(cxn)
#    readout = shared_params['continuous_readout_dur']
#    init_pulse_time = 10**5

    if init_color_ind == 532:
        init_delay = shared_params['515_DM_laser_delay']
    elif init_color_ind == 589:
        init_delay = shared_params['589_aom_delay']
    elif init_color_ind == 638:
        init_delay = shared_params['638_DM_laser_delay']
    else:
        init_delay = 0

    if read_color_ind == 532:
        read_delay = shared_params['515_DM_laser_delay']
    elif read_color_ind == 589:
        read_delay = shared_params['589_aom_delay']
    elif read_color_ind == 638:
        read_delay = shared_params['638_DM_laser_delay']

    aom_ao_589_pwr = nv_sig['am_589_power']
    readout = shared_params['continuous_readout_dur']

    coords = nv_sig['coords']
    drift = tool_belt.get_drift()
    adj_coords = []
    for i in range(3):
        adj_coords.append(coords[i] + drift[i])
    x_center, y_center, z_center = adj_coords


    if x_range != y_range:
        raise RuntimeError('x and y resolutions must match for now.')

    # Get the xy response time
    delay = int(shared_params['xy_delay'])

    total_num_samples = num_steps**2

    # %% Load the PulseStreamer
    seq_args = [galvo_delay, init_delay, read_delay, init_pulse_time,  readout, aom_ao_589_pwr, apd_indices[0],
            init_color_ind, read_color_ind]
#    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load('simple_readout_two_pulse.py',
                                              seq_args_string)
    period = ret_vals[0]

    # %% Initialize at the passed coordinates

    if hasattr(cxn, 'filter_slider_ell9k'):
        cxn.filter_slider_ell9k.set_filter(nv_sig['nd_filter'])
    tool_belt.set_xyz(cxn, [x_center, y_center, z_center])

    # %% Set up the galvo

    xy_server = tool_belt.get_xy_server(cxn)
    x_voltages, y_voltages = xy_server.load_sweep_xy_scan(x_center, y_center,
                                                          x_range, y_range,
                                                          num_steps, period)

    x_num_steps = len(x_voltages)
    x_low = x_voltages[0]
    x_high = x_voltages[x_num_steps-1]
    y_num_steps = len(y_voltages)
    y_low = y_voltages[0]
    y_high = y_voltages[y_num_steps-1]

    pixel_size = x_voltages[1] - x_voltages[0]

    # If we want to spend the same amount of time on an NV, regardless of the
    # scan range or pixel size, we will scale the readout time so that we spend
    # the same amount of time scanning over an NV, regardless of the relative
    #size of the pixel sizes and NV size.
#    readout = int((pixel_size/nv_size)**2 * base_readout)
#    print(pixel_size)
#    print(str(readout /10**3) + 'us')

    readout_us = float(readout) / 10**3

    # %% Set up the APD

    cxn.apd_tagger.start_tag_stream(apd_indices)

    # %% Set up our raw data objects

    # Initialize imgArray and set all values to NaN so that unset values
    # are not interpreted as 0 by matplotlib's colobar
    img_array = numpy.empty((x_num_steps, y_num_steps))
    img_array[:] = numpy.nan
    img_write_pos = []

    # %% Set up the image display

    if plot_data:
#        img_array_kcps = numpy.copy(img_array)

        # For the image extent, we need to bump out the min/max x/y by half the
        # pixel size in each direction so that the center of each pixel properly
        # lies at its x/y voltages.
        half_pixel_size = pixel_size / 2
        img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                      y_low - half_pixel_size, y_high + half_pixel_size]
        title = 'Confocal scan with {} nm init, {} nm readout.\nReadout {} us'.format(init_color_ind, read_color_ind, readout_us)
        fig = tool_belt.create_image_figure(img_array, img_extent,
                                            clickHandler=image_sample.on_click_image,
                                            title = title)

    # %% Collect the data
    cxn.apd_tagger.clear_buffer()
    cxn.pulse_streamer.stream_start(total_num_samples)

    timeout_duration = ((period*(10**-9)) * total_num_samples) + 10
    timeout_inst = time.time() + timeout_duration

    num_read_so_far = 0

    tool_belt.init_safe_stop()

    while num_read_so_far < total_num_samples:

        if time.time() > timeout_inst:
            break

        if tool_belt.safe_stop():
            break

        # Read the samples and update the image
        new_samples = cxn.apd_tagger.read_counter_simple()
#        print(new_samples)
        num_new_samples = len(new_samples)
        if num_new_samples > 0:
            image_sample.populate_img_array(new_samples, img_array, img_write_pos)
            # This is a horribly inefficient way of getting kcps, but it
            # is easy and readable and probably fine up to some resolution
            if plot_data:
#                img_array_kcps[:] = (img_array[:] / 1000) / readout_sec
                tool_belt.update_image_figure(fig, img_array)
            num_read_so_far += num_new_samples

    # %% Clean up

    tool_belt.reset_cfm(cxn)

    # Return to center
    xy_server.write_xy(x_center, y_center)
#    cxn.galvo.write(0.5, 0.5)

    # %% Read the optical power for either yellow or green light

#    if color_ind == 532:
#        optical_power_pd = tool_belt.opt_power_via_photodiode(color_ind)
#    elif color_ind == 589:
#        optical_power_pd = tool_belt.opt_power_via_photodiode(color_ind,
#           AO_power_settings = aom_ao_589_pwr, nd_filter = nv_sig['nd_filter'])
#    elif color_ind == 638:
#        optical_power_pd = tool_belt.opt_power_via_photodiode(color_ind)

    # Convert V to mW optical power
#    optical_power_mW = tool_belt.calc_optical_power_mW(color_ind, optical_power_pd)
    optical_power_pd = None
    optical_power_mW = None


    # %% Save the data

    # measure laser powers:
    #green_optical_power_pd, green_optical_power_mW, \
    #        red_optical_power_pd, red_optical_power_mW, \
    #        yellow_optical_power_pd, yellow_optical_power_mW = \
    #        tool_belt.measure_g_r_y_power(
    #                          nv_sig['am_589_power'], nv_sig['nd_filter'])

    timestamp = tool_belt.get_time_stamp()

    rawData = {'timestamp': timestamp,
               'nv_sig': nv_sig,
               'nv_sig-units': tool_belt.get_nv_sig_units(),
               'init_color_ind': init_color_ind,
               'read_color_ind': read_color_ind,
               'optical_power_pd': optical_power_pd,
               'optical_power_pd-units': 'V',
               'optical_power_mW': optical_power_mW,
               'optical_power_mW-units': 'mW',
               'aom_ao_589_pwr': aom_ao_589_pwr,
               'aom_ao_589_pwr-units': 'V',
               'x_range': x_range,
               'x_range-units': 'V',
               'y_range': y_range,
               'y_range-units': 'V',
               'num_steps': num_steps,
               'readout': readout,
               'readout-units': 'ns',
               'init_pulse_time': init_pulse_time,
               'init_pulse_time-units': 'ns',

            #'green_optical_power_pd': green_optical_power_pd,
            #'green_optical_power_pd-units': 'V',
            #'green_optical_power_mW': green_optical_power_mW,
            #'green_optical_power_mW-units': 'mW',
            #'red_optical_power_pd': red_optical_power_pd,
            #'red_optical_power_pd-units': 'V',
            #'red_optical_power_mW': red_optical_power_mW,
            #'red_optical_power_mW-units': 'mW',
            #'yellow_optical_power_pd': yellow_optical_power_pd,
            #'yellow_optical_power_pd-units': 'V',
            #'yellow_optical_power_mW': yellow_optical_power_mW,
            #'yellow_optical_power_mW-units': 'mW',
               'x_voltages': x_voltages.tolist(),
               'x_voltages-units': 'V',
               'y_voltages': y_voltages.tolist(),
               'y_voltages-units': 'V',
               'img_array': img_array.astype(int).tolist(),
               'img_array-units': 'counts'}

    if save_data:

        filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
        tool_belt.save_raw_data(rawData, filePath)

        if plot_data:

            tool_belt.save_figure(fig, filePath)

    return img_array, x_voltages, y_voltages


# %% Run the file


if __name__ == '__main__':


    path = 'pc_hahn/branch_cryo-setup/image_sample/2021_03'
    file_name = '2021_03_02-14_57_01-johnson-nv14_2021_02_26'

    data = tool_belt.get_raw_data(path, file_name)
    img_array = data['img_array']
    x_voltages = data['x_voltages']
    y_voltages = data['y_voltages']
    x_low = x_voltages[0]
    x_high = x_voltages[-1]
    y_low = y_voltages[0]
    y_high = y_voltages[-1]
    pixel_size = x_voltages[1] - x_voltages[0]
    half_pixel_size = pixel_size / 2
    img_extent = [x_high + half_pixel_size, x_low - half_pixel_size,
                  y_low - half_pixel_size, y_high + half_pixel_size]
    
    tool_belt.create_image_figure(img_array, img_extent, clickHandler=None,
                        title=None, color_bar_label='Counts', 
                        min_value=None, um_scaled=False)
