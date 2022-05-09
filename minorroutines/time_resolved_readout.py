# -*- coding: utf-8 -*-
"""
This program allows the initial laser pulse and illumination pulse colors to be
specified between 532, 589, and 638. 

The apd then collects the photons duringreadout and plots the binned counts 
over time. This program is a reworked lifetime_v2.

Useful combination of init and illum lasers:
    init green and illum yellow
    init red and illum yellow
    init red and illum green

Created on Tue Mar 24 12:49:55 2020

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import time
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import json
import copy
import labrad


# %% Functions


def process_raw_buffer(new_tags, new_channels,
                       current_tags, current_channels,
                       gate_open_channel, gate_close_channel):
    # The processing here will be bin_size agnostic
    
    # Tack the new data onto the leftover data (leftovers are necessary if
    # the last read contained a gate open without a matching gate close)
    current_tags.extend(new_tags)
    current_channels.extend(new_channels)
    current_channels_array = numpy.array(current_channels)
    
    # Find gate open clicks
    result = numpy.nonzero(current_channels_array == gate_open_channel)
    gate_open_click_inds = result[0].tolist()

    # Find gate close clicks
    result = numpy.nonzero(current_channels_array == gate_close_channel)
    gate_close_click_inds = result[0].tolist()
    
#    try:
#        print(gate_open_click_inds[-1])
#        print(gate_close_click_inds[-1])
#    except:
#        pass
    
    new_processed_tags = []
    
    # Loop over the number of closes we have since there are guaranteed to
    # be opens
    
    num_closed_samples = min(len(gate_close_click_inds),len(gate_open_click_inds))
#    print()
    # print('Num open gate clicks: ' + str(len(gate_open_click_inds)))
    # print('Num close gate clicks: ' + str(len(gate_close_click_inds)))
    for list_ind in range(num_closed_samples):
        
        gate_open_click_ind = gate_open_click_inds[list_ind]
        gate_close_click_ind = gate_close_click_inds[list_ind]
        
        # Extract all the counts between these two indices as a single sample
        rep = current_tags[gate_open_click_ind+1:
                                    gate_close_click_ind]
        rep = numpy.array(rep, dtype=numpy.int64)
        # Make relative to gate open
        rep -= current_tags[gate_open_click_ind]
        new_processed_tags.extend(rep.astype(numpy.int64).tolist())
        
    # Clear processed tags
    if len(gate_close_click_inds) > 0:
        leftover_start = gate_close_click_inds[-1]
        del current_tags[0: leftover_start+1]
        del current_channels[0: leftover_start+1]
        
    return new_processed_tags, num_closed_samples


# %% Main


def main(nv_sig, apd_indices,   num_reps, num_runs, num_bins, plot = True):

    with labrad.connect() as cxn:
        bin_centers, binned_samples_sig = main_with_cxn(cxn, 
                  nv_sig, apd_indices, num_reps, num_runs, num_bins, plot)

    return  bin_centers, binned_samples_sig

def main_with_cxn(cxn, nv_sig, apd_indices, num_reps, num_runs, num_bins, plot):
    
    if len(apd_indices) > 1:
        msg = 'Currently lifetime only supports single APDs!!'
        raise NotImplementedError(msg)
    
    tool_belt.reset_cfm(cxn)
    

    
    # %% Read the optical power forinit and illum pulse light. 
    # Aslo set the init pusle and illum pulse delays
    
    init_laser = 'initialize_laser'
    readout_laser = 'charge_readout_laser'
    init_laser_key = nv_sig[init_laser]
    readout_laser_key = nv_sig[readout_laser]
    
    # Initial Calculation and setup
    tool_belt.set_filter(cxn, nv_sig, init_laser)
    
    tool_belt.set_filter(cxn, nv_sig, readout_laser)
    
    
    init_laser_power = tool_belt.set_laser_power(
        cxn, nv_sig, init_laser
    )
    readout_laser_power = tool_belt.set_laser_power(
        cxn, nv_sig, readout_laser
    )

    # Estimate the lenth of the sequance            
    file_name = 'time_resolved_readout.py'
    
    #### Load the measuremnt 
    readout_on_pulse_ind = 2
    init_laser_duration = nv_sig["{}_dur".format(init_laser)]
    readout_laser_duration = nv_sig["{}_dur".format(readout_laser)]
    readout_apd_duration = readout_laser_duration*2
    
    seq_args = [init_laser_duration, readout_apd_duration, readout_laser_duration,
                init_laser_key, readout_laser_key,
                init_laser_power, readout_laser_power, 
                readout_on_pulse_ind, apd_indices[0]]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    # print(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    seq_time = ret_vals[0]


    # %% Report the expected run time

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * num_runs * seq_time_s  # s
    expected_run_time_m = expected_run_time / 60 # m
    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))
    # return

    # %% Bit more setup

    # Record the start time
    startFunctionTime = time.time()       
    run_start_time = startFunctionTime
    start_timestamp = tool_belt.get_time_stamp()
    
    # %% Collect data on point of interest
    
    # opti_coords_list = []
    optimize.main_with_cxn(cxn, nv_sig, apd_indices)

    
    processed_tags_signal = []

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        
        print(' \nRun index: {}'.format(run_ind))
        current_time = time.time()
        if current_time - run_start_time > 4*60:
            optimize.main_with_cxn(cxn, nv_sig, apd_indices)
            run_start_time = current_time

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break        
        
        # Expose the stream
        cxn.apd_tagger.start_tag_stream(apd_indices, apd_indices, False)
    
        # Find the gate channel
        # The order of channel_mapping is APD, APD gate open, APD gate close
        channel_mapping = cxn.apd_tagger.get_channel_mapping()
        gate_open_channel = channel_mapping[1]
        gate_close_channel = channel_mapping[2]
            
        cxn.pulse_streamer.stream_immediate(file_name, int(num_reps),
                                            seq_args_string)
        # Initialize state
        current_tags = []
        current_channels = []
        num_processed_reps = 0

        # print('sig')
        while num_processed_reps < num_reps:
            
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            
            # Read the stream and convert from strings to int64s
            ret_vals = cxn.apd_tagger.read_tag_stream()
            buffer_timetags, buffer_channels = ret_vals
            buffer_timetags = numpy.array(buffer_timetags, dtype=numpy.int64)
            current_tags.extend(buffer_timetags.tolist())
            current_channels.extend(buffer_channels.tolist())
                
            ret_vals = process_raw_buffer(buffer_timetags, buffer_channels,
                                   current_tags, current_channels,
                                   gate_open_channel, gate_close_channel)
        
            new_processed_tags, num_new_processed_reps = ret_vals
            # print(new_processed_tags)
                
            
            num_processed_reps += num_new_processed_reps
            
            processed_tags_signal.extend(new_processed_tags)
            
        #  Save the data we have incrementally for long measurements
        processed_tags_signal = [int(el) for el in processed_tags_signal]
        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'init_laser_duration': init_laser_duration,
                    'readout_laser_duration': readout_laser_duration,
                    'readout_apd_duration': readout_apd_duration,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'run_ind': run_ind,
                    'processed_tags': processed_tags_signal,
                    'processed_tags-units': 'ps',
                    }

        cxn.apd_tagger.stop_tag_stream()

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
        

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Bin the data
    
    readout_time_ps = 1000*readout_apd_duration
    
    binned_samples_sig, bin_edges_sig = numpy.histogram(processed_tags_signal, num_bins,
                                (0, readout_time_ps))
    
    # Compute the centers of the bins
    bin_size = readout_apd_duration / num_bins
    bin_center_offset = bin_size / 2
    bin_centers = numpy.linspace(0, readout_apd_duration, num_bins) + bin_center_offset
#    print(bin_centers)

    # %% Plot
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                          ['Config', 'Optics', nv_sig[init_laser]])
        readout_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                          ['Config', 'Optics', nv_sig[readout_laser]])
    
        ax.plot(bin_centers, binned_samples_sig, 'k-')
        ax.set_title('Lifetime')
        ax.set_xlabel('Readout time (ns)')
        ax.set_ylabel('Counts')
        ax.set_title('{} initial pulse, {} readout'.format(init_color, readout_color))
           
        params_text = '\n'.join(('Init pulse time: {} us'.format(init_laser_duration/10**3),
                          'bin size: ' + '%.1f'%(bin_size) + 'ns'))
    
        ax.text(0.55, 0.85, params_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
          
            
        fig.canvas.draw()
        fig.set_tight_layout(True)
        fig.canvas.flush_events()

    # %% Save the data

    endFunctionTime = time.time()
    time_elapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    
    raw_data = {'timestamp': timestamp,
                'time_elapsed': time_elapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'init_laser_duration': init_laser_duration,
                'readout_laser_duration': readout_laser_duration,
                'readout_apd_duration': readout_apd_duration,
                'num_bins': num_bins,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'binned_samples': binned_samples_sig.tolist(),
                'bin_centers': bin_centers.tolist(),
                'processed_tags_signal': processed_tags_signal,
                'processed_tags-units': 'ps',
                }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    if plot:
        tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)
    
    return bin_centers, binned_samples_sig


# %% Main for three pulses


def main_three_pulses(nv_sig, apd_indices,   num_reps, num_runs, num_bins, plot = True):

    with labrad.connect() as cxn:
        bin_centers, binned_samples = main_three_pulses_with_cxn(cxn, 
                  nv_sig, apd_indices, num_reps, num_runs, num_bins, plot)

    return bin_centers, binned_samples

def main_three_pulses_with_cxn(cxn, nv_sig, apd_indices, num_reps, num_runs, num_bins, plot):
    
    if len(apd_indices) > 1:
        msg = 'Currently lifetime only supports single APDs!!'
        raise NotImplementedError(msg)
    
    tool_belt.reset_cfm(cxn)
    

    
    # %% Read the optical power forinit and illum pulse light. 
    # Aslo set the init pusle and illum pulse delays
    
    init_laser = 'initialize_laser'
    test_laser = 'test_laser'
    readout_laser = 'charge_readout_laser'
    init_laser_key = nv_sig[init_laser]
    test_laser_key = nv_sig[test_laser]
    readout_laser_key = nv_sig[readout_laser]
    
    # Initial Calculation and setup
    tool_belt.set_filter(cxn, nv_sig, init_laser)
    tool_belt.set_filter(cxn, nv_sig, test_laser)
    tool_belt.set_filter(cxn, nv_sig, readout_laser)
    
    
    init_laser_power = tool_belt.set_laser_power(
        cxn, nv_sig, init_laser
    )
    test_laser_power = tool_belt.set_laser_power(
        cxn, nv_sig, test_laser
    )
    readout_laser_power = tool_belt.set_laser_power(
        cxn, nv_sig, readout_laser
    )

    # Estimate the lenth of the sequance            
    file_name = 'time_resolved_readout_three_pulses.py'
    
    #### Load the measuremnt 
    init_laser_duration = nv_sig["{}_dur".format(init_laser)]
    test_laser_duration = nv_sig["{}_dur".format(test_laser)]
    readout_laser_duration = nv_sig["{}_dur".format(readout_laser)]
    readout_apd_duration = readout_laser_duration*2
    
    seq_args = [init_laser_duration, test_laser_duration, 
                                readout_apd_duration, readout_laser_duration,
                init_laser_key, test_laser_key, readout_laser_key,
                init_laser_power, test_laser_power, readout_laser_power, 
                apd_indices[0]]
    # print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    seq_time = ret_vals[0]


    # %% Report the expected run time

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * num_runs * seq_time_s  # s
    expected_run_time_m = expected_run_time / 60 # m
    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))
    # return

    # %% Bit more setup

    # Record the start time
    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()
    
    # opti_coords_list = []
    optimize.main_with_cxn(cxn, nv_sig, apd_indices)

    # return
    # %% Collect the data
    
    processed_tags = []

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        
        print(' \nRun index: {}'.format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break        
        
        # Expose the stream
        cxn.apd_tagger.start_tag_stream(apd_indices, apd_indices, False)
    
        # Find the gate channel
        # The order of channel_mapping is APD, APD gate open, APD gate close
        channel_mapping = cxn.apd_tagger.get_channel_mapping()
        gate_open_channel = channel_mapping[1]
        gate_close_channel = channel_mapping[2]
            
        cxn.pulse_streamer.stream_immediate(file_name, int(num_reps),
                                            seq_args_string)
        # Initialize state
        current_tags = []
        current_channels = []
        num_processed_reps = 0

        while num_processed_reps < num_reps:
            
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
            
            # Read the stream and convert from strings to int64s
            ret_vals = cxn.apd_tagger.read_tag_stream()
            buffer_timetags, buffer_channels = ret_vals
            buffer_timetags = numpy.array(buffer_timetags, dtype=numpy.int64)
            current_tags.extend(buffer_timetags.tolist())
            current_channels.extend(buffer_channels.tolist())
            
            # if new_tags == []:
            #     continue
#            print()
#            print(time.time()-dur_start)
            
            # MCC test
            # if len(new_tags) > 750000:
            #     print()
            #     print('Received {} tags out of 10^6 max'.format(len(new_tags)))
            #     print('Turn down the reps and turn up the runs so that the Time Tagger can catch up!')
            
#            dur_start = time.time()
                
            ret_vals = process_raw_buffer(buffer_timetags, buffer_channels,
                                   current_tags, current_channels,
                                   gate_open_channel, gate_close_channel)
        
            new_processed_tags, num_new_processed_reps = ret_vals
#            print(time.time()-dur_start)
            
            num_processed_reps += num_new_processed_reps
            
            processed_tags.extend(new_processed_tags)
            

        cxn.apd_tagger.stop_tag_stream()
        
        # %% Save the data we have incrementally for long measurements
        processed_tags = [int(el) for el in processed_tags]
        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'init_laser_duration': init_laser_duration,
                    'test_laser_duration': test_laser_duration,
                    'readout_laser_duration': readout_laser_duration,
                    'readout_apd_duration': readout_apd_duration,
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'run_ind': run_ind,
                    'processed_tags': processed_tags,
                    'processed_tags-units': 'ps',
                    }

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)
        

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Bin the data
    
    readout_time_ps = 1000*readout_apd_duration
    
#    start_readout_time_ps = 1000*start_readout_time
#    end_readout_time_ps = 1000*end_readout_time
    binned_samples, bin_edges = numpy.histogram(processed_tags, num_bins,
                                (0, readout_time_ps))
#    print(binned_samples)
    
    # Compute the centers of the bins
    bin_size = readout_apd_duration / num_bins
    bin_center_offset = bin_size / 2
    bin_centers = numpy.linspace(0, readout_apd_duration, num_bins) + bin_center_offset
#    print(bin_centers)

    # %% Plot
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        
    
        init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                          ['Config', 'Optics', nv_sig[init_laser]])
        test_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                          ['Config', 'Optics', nv_sig[test_laser]])
        readout_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                          ['Config', 'Optics', nv_sig[readout_laser]])
    
        ax.plot(bin_centers, binned_samples, 'r-')
        ax.set_title('Lifetime')
        ax.set_xlabel('Readout time (ns)')
        ax.set_ylabel('Counts')
        ax.set_title('{} initial pulse, {} test pulse, {} readout'.format(init_color,test_color, readout_color))
           
        params_text = '\n'.join(('Init pulse time: {} us'.format(init_laser_duration/10**3),
                                 'Test pulse time: {} us'.format(test_laser_duration/10**3),
                          'bin size: ' + '%.1f'%(bin_size) + 'ns'))
    
        ax.text(0.55, 0.85, params_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        fig.canvas.draw()
        fig.set_tight_layout(True)
        fig.canvas.flush_events()

    # %% Save the data

    endFunctionTime = time.time()
    time_elapsed = endFunctionTime - startFunctionTime
    timestamp = tool_belt.get_time_stamp()
    
    raw_data = {'timestamp': timestamp,
                'time_elapsed': time_elapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'init_laser_duration': init_laser_duration,
                'test_laser_duration': test_laser_duration,
                'readout_laser_duration': readout_laser_duration,
                'readout_apd_duration': readout_apd_duration,
                'num_bins': num_bins,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'binned_samples': binned_samples.tolist(),
                'bin_centers': bin_centers.tolist(),
                'processed_tags': processed_tags,
                'processed_tags-units': 'ps',
                }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    if plot:
        tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)
    
    return bin_centers, binned_samples


# %% Run the files

if __name__ == '__main__':
    file_path = 'pc_rabi/branch_master/time_resolved_readout/2022_04'
    
    #--- red init
    # 17 mW red readout
    sig_file = '2022_04_18-14_14_33-sandia-siv_R10_a130_r1_c1'
    ref_file = '2022_04_18-14_22_36-sandia-siv_R10_a130_r1_c1'
    # 17 mW green readout
    # sig_file = '2022_04_15-15_10_18-sandia-siv_R10_a130_r1_c1'
    # ref_file = '2022_04_15-15_15_38-sandia-siv_R10_a130_r1_c1'
    
    data = tool_belt.get_raw_data(sig_file, file_path)
    nv_sig = data['nv_sig']
    try:
        binned_samples_sig = numpy.array(data['binned_samples'])
    except Exception:
        binned_samples_sig = numpy.array(data['binned_samples_sig'])
    bin_centers = data['bin_centers']
    
    data = tool_belt.get_raw_data(ref_file, file_path)
    try:
        binned_samples_ref = numpy.array(data['binned_samples'])
    except Exception:
        binned_samples_ref = numpy.array(data['binned_samples_sig'])
    bin_centers = data['bin_centers']
    
    binned_samples_sub_red = binned_samples_sig - binned_samples_ref
    
    #--- green init
    # 17 mW red readout
    sig_file = '2022_04_18-14_36_40-sandia-siv_R10_a130_r1_c1'
    ref_file = '2022_04_18-14_44_15-sandia-siv_R10_a130_r1_c1'
    # 17 mW green readout
    # sig_file = '2022_04_18-12_57_01-sandia-siv_R10_a130_r1_c1'
    # ref_file = '2022_04_18-12_57_14-sandia-siv_R10_a130_r1_c1'
    
    data = tool_belt.get_raw_data(sig_file, file_path)
    nv_sig = data['nv_sig']
    try:
        binned_samples_sig = numpy.array(data['binned_samples'])
    except Exception:
        binned_samples_sig = numpy.array(data['binned_samples_sig'])
    bin_centers = data['bin_centers']
    
    data = tool_belt.get_raw_data(ref_file, file_path)
    try:
        binned_samples_ref = numpy.array(data['binned_samples'])
    except Exception:
        binned_samples_ref = numpy.array(data['binned_samples_sig'])
    bin_centers = data['bin_centers']
    
    binned_samples_sub_green = binned_samples_sig - binned_samples_ref
    
    #---
    binned_samples_sub = binned_samples_sub_red - binned_samples_sub_green
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    init_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['initialize_laser']])
    readout_color = tool_belt.get_registry_entry_no_cxn('wavelength',
                      ['Config', 'Optics', nv_sig['charge_readout_laser']])
    
    ax.plot(bin_centers, binned_samples_sub, 'b-')
    ax.set_title('Subtracted lifetime')
    ax.set_xlabel('Readout time (ns)')
    ax.set_ylabel('Counts')
    ax.set_title('{} initial pulse, {} readout'.format(init_color, readout_color))