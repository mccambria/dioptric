# -*- coding: utf-8 -*-
"""
g(2) routine. For each event on one channel, calculates the deltas relative to
the events on the opposite channel and plots a histogram of the deltas. Here
the events are photon detections from the same source, but split over two
APDs. The splitting is necessary to compensate for the APD dead time, which
is typically significantly longer than the lifetime of the excited state we
are interested in.

+++++++++(from scc branch)+++++++++

Created on Wed Apr 24 17:33:26 2019

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import matplotlib.pyplot as plt
import time
import json
import labrad
from utils.tool_belt import Mod_types

# %% Functions


def calculate_relative_g2_zero(hist):

    # We take the values on the wings to be representatives for g2(inf)
    # We take the wings to be the first and last 1/6 of collected data
    num_bins = len(hist)
    wing_length = num_bins // 12
    neg_wing = hist[0: wing_length]
    pos_wing = hist[num_bins - wing_length: ]
    inf_delay_differences = numpy.average([neg_wing, pos_wing])

    # Use the parity of num_bins to determine differences at 0 ns
    if num_bins % 2 == 0:
        # As an example, say there are 6 bins. Then we want the differences
        # from bins 2 and 3 (indexing starts from 0).
        midpoint_high = num_bins // 2
        zero_delay_differences = numpy.average(hist[midpoint_high - 1,
                                                    midpoint_high])
    else:
        # Now say there are 7 bins. We'd like bin 3.
        midpoint = int(numpy.floor(num_bins / 2))
        zero_delay_differences = hist[midpoint]

    return zero_delay_differences / inf_delay_differences


def process_raw_buffer(timestamps, channels,
                       diff_window_ps, afterpulse_window,
                       differences_append, apd_a_chan_name, apd_b_chan_name):

    indices_to_delete = []
    indices_to_delete_append = indices_to_delete.append

    # Throw out probable afterpulses
    # Something is wrong with this... see 2019-06-03_17-05-01_ayrton12
    if False:
        num_vals = timestamps.size
        for click_index in range(num_vals):

            click_time = timestamps[click_index]

            # Determine the afterpulse channel
            click_channel = channels[click_index]

            # Calculate relevant differences
            next_index = click_index + 1
            while next_index < num_vals:
                diff = timestamps[next_index] - click_time
                if diff > afterpulse_window:
                    break
                if channels[next_index] == click_channel:
                    indices_to_delete_append(next_index)
                next_index += 1

        timestamps = numpy.delete(timestamps, indices_to_delete)

    # Calculate differences
    num_vals = timestamps.size
    for click_index in range(num_vals):

        click_time = timestamps[click_index]

        # Determine the channel to take the difference with
        click_channel = channels[click_index]
        if click_channel == apd_a_chan_name:
            diff_channel = apd_b_chan_name
        else:
            diff_channel = apd_a_chan_name

        # Calculate relevant differences
        next_index = click_index + 1
        while next_index < num_vals:  # Don't go past the buffer end
            # Stop taking differences past the diff window
            diff = timestamps[next_index] - click_time
            if diff > diff_window_ps:
                break
            # Only record the diff between opposite chanels
            if channels[next_index] == diff_channel:
                # Flip the sign for diffs relative to channel 2
                if click_channel == apd_b_chan_name:
                    diff = -diff
                differences_append(int(diff))
            next_index += 1


# %% Main


def main(nv_sig, run_time, diff_window,
         apd_a_index, apd_b_index):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, run_time, diff_window,
                      apd_a_index, apd_b_index)

def main_with_cxn(cxn, nv_sig, run_time, diff_window,
                  apd_a_index, apd_b_index):

    # %% Initial calculations and setup

    tool_belt.reset_cfm(cxn)

    afterpulse_window = 50 * 10**3
    sleep_time = 2
    apd_indices = [apd_a_index, apd_b_index]

    # Set xyz and open the AOM
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)

    wiring = tool_belt.get_pulse_streamer_wiring(cxn)
    imaging_laser = nv_sig['imaging_laser']
    
    mod_type = tool_belt.get_registry_entry_no_cxn('mod_type',
                      ['Config', 'Optics', imaging_laser])
    mod_type = eval(mod_type)
    if mod_type is Mod_types.DIGITAL:
        imaging_laser_key = 'do_{}_dm'.format(imaging_laser)
        cxn.pulse_streamer.constant([wiring[imaging_laser_key]])
    elif mod_type is Mod_types.ANALOG:
        imaging_laser_key = 'ao_{}_am'.format(imaging_laser)
        imaging_laser_power = nv_sig['imaging_laser_power']
        analog_out = tool_belt.get_registry_entry_no_cxn(imaging_laser_key,
                      ['Config', 'Wiring', 'PulseStreamer'])
        if analog_out == 0:
            cxn.pulse_streamer.constant([],imaging_laser_power)
        elif analog_out == 1:
            cxn.pulse_streamer.constant([],0,imaging_laser_power)
            
    num_tags = 0
    collection_index = 0

    diff_window_ps = diff_window * 1000
    differences = []  # Create a list to hold the differences
    differences_append = differences.append  # Skip unnecessary lookup
    num_bins = int(2*1 * diff_window) + 1  # 1 ns bins in ps
    # num_bins = int(2*2 * diff_window) + 1  # 0.5 ns bins in ps
    # num_bins = int(2*3 * diff_window) + 1  # 0.125 ns bins in ps

    # Expose the stream
    cxn.apd_tagger.start_tag_stream(apd_indices, [], False)

    # Get the APD channel names that the tagger will return
    ret_vals = cxn.apd_tagger.get_channel_mapping()
    apd_a_chan_name, apd_b_chan_name = ret_vals

    # %% Collect the data

    start_time = time.time()
    print('Time remaining:')
    tool_belt.init_safe_stop()

    # Python does not have do-while loops so we will use something like
    # a while True
    stop = False
    start_calc_time = start_time
    while not stop:

        # Wait until some data has filled
        now = time.time()
        calc_time_elapsed = now - start_calc_time
        time.sleep(max(sleep_time - calc_time_elapsed, 0))
        # Read the stream and convert from strings to int64s
        ret_vals= cxn.apd_tagger.read_tag_stream()
        # ret_vals = tool_belt.decode_time_tags(ret_vals_string)
        buffer_timetags, buffer_channels = ret_vals
        buffer_timetags = numpy.array(buffer_timetags, dtype=numpy.int64)

        # Check if we should stop
        time_remaining = (start_time + run_time)- time.time()
        if (time_remaining < 0) or tool_belt.safe_stop():
            stop = True
        else:
            print(int(time_remaining))

        # Process data
        start_calc_time = time.time()
        process_raw_buffer(buffer_timetags, buffer_channels,
                           diff_window_ps, afterpulse_window,
                           differences_append,
                           apd_a_chan_name, apd_b_chan_name)

        # Create/update the histogram
        if collection_index == 0:
            fig, ax = plt.subplots()
            hist, bin_edges = numpy.histogram(differences, num_bins)
            bin_edges = bin_edges / 1000  # ps to ns
            bin_center_offset = (bin_edges[1] - bin_edges[0]) / 2
            bin_centers = bin_edges[0: num_bins] + bin_center_offset
            ax.plot(bin_centers, hist)
            xlim = int(1.1 * diff_window)
            ax.set_xlim(-xlim, xlim)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Differences')
            ax.set_title(r'$g^{(2)}(\tau)$')
            fig.set_tight_layout(True)
            fig.canvas.draw()
            fig.canvas.flush_events()
        elif collection_index > 1:
            hist, bin_edges = numpy.histogram(differences, num_bins)
            tool_belt.update_line_plot_figure(fig, hist)

        collection_index += 1
        num_tags += len(buffer_timetags)

    cxn.apd_tagger.stop_tag_stream()

    # %% Calculate a relative value for g2(0)

    g2_zero = calculate_relative_g2_zero(hist)

    # %% Clean up and save the data

    tool_belt.reset_cfm(cxn)

    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'g2_zero': g2_zero,
                'g2_zero-units': 'ratio',
                'opti_coords': opti_coords,
                'opti_coords-units': 'V',
                'run_time': run_time,
                'run_time-units': 's',
                'diff_window': diff_window,
                'diff_window-units': 'ns',
                'num_bins': num_bins,
                'num_tags': num_tags,
                'differences': differences,
                'differences-units': 'ps'}

    filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(raw_data, filePath)

    print('g2(0) = {}'.format(g2_zero))

    return g2_zero


# %% Run the file

def calculate_relative_g2_zero_mod(hist):
    
    # We take the values on the wings to be representatives for g2(inf)
    # We take the wings to be the first and last 1/6 of collected data
    num_bins = len(hist)
    wing_length = num_bins // 12
    neg_wing = hist[0: wing_length]
    pos_wing = hist[num_bins - wing_length: ]
    inf_delay_differences = numpy.average([neg_wing, pos_wing])
    
    # Use the parity of num_bins to determine differences at 0 ns
    if num_bins % 2 == 0:
        # As an example, say there are 6 bins. Then we want the differences
        # from bins 2 and 3 (indexing starts from 0).
        midpoint_high = num_bins // 2
        zero_delay_differences = numpy.average(hist[midpoint_high - 1,
                                                    midpoint_high])
    else:
        # Now say there are 7 bins. We'd like bin 3. 
        midpoint = int(numpy.floor(num_bins / 2))
        zero_delay_differences = hist[midpoint]
        
    return zero_delay_differences / inf_delay_differences, inf_delay_differences

if __name__ == '__main__':
    folder_name = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_Spin_to_charge/g2_measurement/2021_04'
    # folder_name = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_master/g2_measurement/2019_05'
    files = [
            # '2019-05-06_18-39-22_ayrton12.txt',
            
            '2021_04_26-15_14_16-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_15_24-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_16_31-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_17_38-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_18_44-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_19_52-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_20_58-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_22_04-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_23_11-goeppert-mayer-nv5_2021_04_15',
            '2021_04_26-15_24_19-goeppert-mayer-nv5_2021_04_15',

            # '2021_04_12-12_27_58-goeppert-mayer-nv13_2021_04_02', #120
            # # '2021_04_12-12_21_09-goeppert-mayer-nv13_2021_04_02', # 240
            # '2021_04_12-12_16_33-goeppert-mayer-nv13_2021_04_02', #30
            # '2021_04_12-12_14_45-goeppert-mayer-nv13_2021_04_02', #30,
            # '2021_04_12-12_07_47-goeppert-mayer-nv13_2021_04_02', # 30
            # # '2021_04_12-11_02_41-goeppert-mayer-nv13_2021_04_02', # 150
            
            # # '2021_04_12-10_43_16-goeppert-mayer-nv13_2021_04_02', # 150
            # '2021_04_12-10_30_18-goeppert-mayer-nv13_2021_04_02', #150
            # '2021_04_12-12_12_01-goeppert-mayer-nv13_2021_04_02', #30
            # '2021_04_12-10_24_36-goeppert-mayer-nv13_2021_04_02' #150
            
            # '2021_04_13-16_13_28-johnson-nv0_2021_04_13', #120
            # '2021_04_13-16_19_36-johnson-nv0_2021_04_13', #300
            # '2021_04_13-16_56_20-johnson-nv0_2021_04_13', #60
            # '2021_04_13-16_57_27-johnson-nv0_2021_04_13',
            # '2021_04_13-16_58_34-johnson-nv0_2021_04_13',
            # '2021_04_13-16_59_42-johnson-nv0_2021_04_13',
            # '2021_04_13-17_00_49-johnson-nv0_2021_04_13',
            # # '2021_04_13-17_01_56-johnson-nv0_2021_04_13',
             ]
    bkgd = 0
    histogram_list = []
    run_time_total = 0
    for file_name in files:
        with open('{}/{}.txt'.format(folder_name, file_name)) as file:
            data = json.load(file)
            differences = data['differences']
            num_bins = data['num_bins']
            timestamp = data['timestamp']
            nv_sig = data['nv_sig']
            diff_window= data['diff_window']
            run_time = data['run_time']
            run_time_total = run_time_total + run_time

        hist, bin_edges = numpy.histogram(differences, num_bins)
        bin_center_offset = (bin_edges[1] - bin_edges[0]) / 2
        bin_centers = bin_edges[0: num_bins] + bin_center_offset
        
        histogram_list.append(hist.tolist())
#    print(histogram_list)
    histogram = numpy.sum(histogram_list, axis = 0)
    g2_zero, inf_delay_diff = calculate_relative_g2_zero_mod(histogram - bkgd)
    print(run_time_total/60)
    txt_size = 10
    
    fig, ax = plt.subplots()
    ax.plot(bin_centers / 1000, (histogram-bkgd))#/inf_delay_diff)
    ax.set_xlabel('Delay time (ns)', fontsize=txt_size)
    ax.set_ylabel(r'$g^{(2)}(\tau)$', fontsize=txt_size)
#    ax.set_title(title[i], fontsize=30)
#    ax.set_xticks([-150, -75, 0, 75, 150])
#    ax.tick_params(which = 'both', length=10, width=2, colors='k',
#                    direction='in',grid_alpha=0.7, labelsize = txt_size)
    # ax.set_ylim(bottom=0, top = 2)
    
    print(g2_zero)
    raw_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'g2_zero': g2_zero,
                'g2_zero-units': 'ratio',
                'run_time': run_time_total,
                'run_time-units': 's',
                'diff_window': diff_window,
                'background_kcps': 7,
                'diff_window-units': 'ns',
                'num_bins': num_bins,
                'histogram': histogram.tolist(),
                'bin_centers': bin_centers.tolist(),
                'bin_centers-units': 'ps'}

    # filePath = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    filePath = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_rabi/branch_Spin_to_charge/g2_measurement/2021_04/{}'.format(file_name)
    tool_belt.save_figure(fig, filePath + '_accumul_10s')
    tool_belt.save_raw_data(raw_data, filePath + '_accumul_10s')
