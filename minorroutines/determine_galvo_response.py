# -*- coding: utf-8 -*-
"""

A routine to measure the response time of the galvo.

Input positions along x to scan between. The galve will then move as time resolved
readout records the counts.

Created on Tue Nov 10 13:36:12 2020

@author: agardill
"""


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import time
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import json
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
    num_closed_samples = len(gate_close_click_inds)
#    print()
#    print('Num open gate clicks: ' + str(len(gate_open_click_inds)))
#    print('Num close gate clicks: ' + str(len(gate_close_click_inds)))
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


def main ( nv_sig,x_range, num_steps, apd_indices, illumination_time,
                  num_reps, num_runs, num_bins, plot = True):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, 
                   nv_sig, x_range, num_steps,  apd_indices, illumination_time,
                  num_reps, num_runs, num_bins, plot)

    return 

def main_with_cxn(cxn, nv_sig, x_range, num_steps, apd_indices, illumination_time,
                  num_reps, num_runs, num_bins, plot):
    
    if len(apd_indices) > 1:
        msg = 'Currently lifetime only supports single APDs!!'
        raise NotImplementedError(msg)
    
    tool_belt.reset_cfm(cxn)

    # %% Define the times to be used in the sequence
    
    aom_ao_589_pwr = nv_sig['am_589_power']
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    
#    illumination_time = 10**6


    drift = tool_belt.get_drift()
    start_coords = nv_sig['coords']
    start_coords_drift = numpy.array(start_coords) + numpy.array(drift)
    x_center = start_coords_drift[0]
    y_center = start_coords_drift[1]
    bin_size = illumination_time / num_bins
    
    # %% Read the optical power forinit and illum pulse light. 
    # Aslo set the init pusle and illum pulse delays
    laser_delay = shared_params['515_laser_delay']
     
#    illum_pulse_delay = shared_params['515_laser_delay']


    # %% Analyze the sequence

    # pulls the file of the sequence from serves/timing/sequencelibrary
    file_name = os.path.basename('simple_readout.py')
    seq_args = [laser_delay, illumination_time, 
                aom_ao_589_pwr, apd_indices[0], 532]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
#    seq_time = ret_vals[0]

    # %% Bit more setup

    # Record the start time
#    startFunctionTime = time.time()
#    start_timestamp = tool_belt.get_time_stamp()
    
    opti_coords_list = []

    # %% Calculate galvo voltages to step through
    
    
    # %% Collect the data
    
    processed_tags = []

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        #move galvo and wait so hat we know we're at the point
        tool_belt.set_xyz(cxn,start_coords_drift)
        time.sleep(0.01)
        
        print(' \nRun index: {}'.format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=True)
        opti_coords_list.append(opti_coords)
        
        
        # Expose the stream
        cxn.apd_tagger.start_tag_stream(apd_indices, apd_indices, False)
    
        # Find the gate channel
        # The order of channel_mapping is APD, APD gate open, APD gate close
        channel_mapping = cxn.apd_tagger.get_channel_mapping()
        gate_open_channel = channel_mapping[1]
        gate_close_channel = channel_mapping[2]
            
        # Stream the sequence
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        
        cxn.galvo.load_x_scan(x_center, y_center, x_range,
                             num_steps,int( bin_size)) #is this really doing what we want?
                                                        # might need to send clock pulses for the galvo to advance?
        
        # 
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
            
#            dur_start = time.time()
#            new_tags, new_channels = cxn.apd_tagger.read_tag_stream()
            ret_vals_string = cxn.apd_tagger.read_tag_stream() 
            new_tags,new_channels = tool_belt.decode_time_tags(ret_vals_string)
            # new_channels.index(clock_pulse_channel)
            if new_tags == []:
                continue
#            print()
#            print(time.time()-dur_start)
            
            # MCC test
            if len(new_tags) > 750000:
                print()
                print('Received {} tags out of 10^6 max'.format(len(new_tags)))
                print('Turn down the reps and turn up the runs so that the Time Tagger can catch up!')
            
#            dur_start = time.time()
            ret_vals = process_raw_buffer(new_tags, new_channels,
                                   current_tags, current_channels,
                                   gate_open_channel, gate_close_channel)
            new_processed_tags, num_new_processed_reps = ret_vals
#            print(time.time()-dur_start)
            
            num_processed_reps += num_new_processed_reps
            
            processed_tags.extend(new_processed_tags)
            

        cxn.apd_tagger.stop_tag_stream()
        
        # %% Save the data we have incrementally for long measurements
#        processed_tags = [int(el) for el in processed_tags]
#        raw_data = {'start_timestamp': start_timestamp,
#                    'nv_sig': nv_sig,
#                    'nv_sig-units': tool_belt.get_nv_sig_units(),
#                    'init_color_ind': init_color_ind,
#                    'init_optical_power_pd': init_optical_power_pd,
#                    'init_optical_power_pd-units': 'V',
#                    'init_optical_power_mW': init_optical_power_mW,
#                    'init_optical_power_mW-units': 'mW',
#                    'illum_color_ind': illum_color_ind,
#                    'illum_optical_power_pd': illum_optical_power_pd,
#                    'illum_optical_power_pd-units': 'V',
#                    'illum_optical_power_mW': illum_optical_power_mW,
#                    'illum_optical_power_mW-units': 'mW',
#                    'illumination_time': illumination_time,
#                    'illumination_time-units': 'ns',
#                    'init_pulse_duration': init_pulse_duration,
#                    'init_pulse_duration-units': 'ns',
#                    'num_reps': num_reps,
#                    'num_runs': num_runs,
#                    'run_ind': run_ind,
#                    'opti_coords_list': opti_coords_list,
#                    'opti_coords_list-units': 'V',
#                    'processed_tags': processed_tags,
#                    'processed_tags-units': 'ps',
#                    }
#
#        # This will continuously be the same file path so we will overwrite
#        # the existing file with the latest version
#        file_path = tool_belt.get_file_path(__file__, start_timestamp,
#                                            nv_sig['name'], 'incremental')
#        tool_belt.save_raw_data(raw_data, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Bin the data
    
    readout_time_ps = 1000*illumination_time
    
#    start_readout_time_ps = 1000*start_readout_time
#    end_readout_time_ps = 1000*end_readout_time
    binned_samples, bin_edges = numpy.histogram(processed_tags, num_bins,
                                (0, readout_time_ps))
#    print(binned_samples)
    
    # Compute the centers of the bins
    bin_center_offset = bin_size / 2
    bin_centers = numpy.linspace(0, illumination_time, num_bins) + bin_center_offset
#    print(bin_centers)

    # %% Plot
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        ax.plot(numpy.array(bin_centers)/10**6, binned_samples, 'r-')
        ax.set_title('Lifetime')
        ax.set_xlabel('Readout time (ms)')
        ax.set_ylabel('Counts')
        ax.set_title('Time resolved readout while scanning galvo')
           
        params_text = 'bin size: ' + '%.1f'%(bin_size) + 'ns'
    
        ax.text(0.55, 0.85, params_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        fig.canvas.draw()
        fig.set_tight_layout(True)
        fig.canvas.flush_events()

    # %% Save the data

#    endFunctionTime = time.time()
#    time_elapsed = endFunctionTime - startFunctionTime
#    timestamp = tool_belt.get_time_stamp()
#    
#    raw_data = {'timestamp': timestamp,
#                'time_elapsed': time_elapsed,
#                'nv_sig': nv_sig,
#                'nv_sig-units': tool_belt.get_nv_sig_units(),
#                'init_color_ind': init_color_ind,
#                'init_optical_power_pd': init_optical_power_pd,
#                'init_optical_power_pd-units': 'V',
#                'init_optical_power_mW': init_optical_power_mW,
#                'init_optical_power_mW-units': 'mW',
#                'illum_color_ind': illum_color_ind,
#                'illum_optical_power_pd': illum_optical_power_pd,
#                'illum_optical_power_pd-units': 'V',
#                'illum_optical_power_mW': illum_optical_power_mW,
#                'illum_optical_power_mW-units': 'mW',
#                'readout_time': readout_time,
#                'readout_time-units': 'ns',
#                'init_pulse_duration': init_pulse_duration,
#                'init_pulse_duration-units': 'ns',
#                'num_bins': num_bins,
#                'num_reps': num_reps,
#                'num_runs': num_runs,
#                'opti_coords_list': opti_coords_list,
#                'opti_coords_list-units': 'V',
#                'binned_samples': binned_samples.tolist(),
#                'bin_centers': bin_centers.tolist(),
#                'processed_tags': processed_tags,
#                'processed_tags-units': 'ps',
#                }
#
#    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
#    if plot:
#        tool_belt.save_figure(fig, file_path)
#    tool_belt.save_raw_data(raw_data, file_path)
    
    return 


