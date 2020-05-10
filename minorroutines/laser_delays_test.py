# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:45:26 2020

A short file to test if the delays on all three lasers are correct with respect
to each other.

File collects photon counts from NV. Green light, then yellow, then red shine 
for 1 us, with 100 ns intervals between them.

@author: agardill
"""

import labrad
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import numpy
import os
import time
import matplotlib.pyplot as plt

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
    
    new_processed_tags = []
    
    # Loop over the number of closes we have since there are guaranteed to
    # be opens
    num_closed_samples = len(gate_close_click_inds)
    for list_ind in range(num_closed_samples):
        
        gate_open_click_ind = gate_open_click_inds[list_ind]
        gate_close_click_ind = gate_close_click_inds[list_ind]
        
        # Extract all the counts between these two indices as a single sample
        rep = current_tags[gate_open_click_ind+1:
                                    gate_close_click_ind]
        rep = numpy.array(rep, dtype=numpy.int64)
        # Make relative to gate open
        rep -= current_tags[gate_open_click_ind]
        new_processed_tags.extend(rep.astype(int).tolist())
        
    # Clear processed tags
    if len(gate_close_click_inds) > 0:
        leftover_start = gate_close_click_inds[-1]
        del current_tags[0: leftover_start+1]
        del current_channels[0: leftover_start+1]
        
    return new_processed_tags, num_closed_samples


# %% Mian function

def main(cxn, nv_sig, apd_indices):
    num_bins = 151
    num_reps = 10**4
    num_runs = 1
    
    # Input a set delay to check that measured delay is correct
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_532_delay = shared_params['515_laser_delay']
    laser_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_laser_delay']

    # %% Bit more setup

    # Record the start time
    startFunctionTime = time.time()
    
    opti_coords_list = []

    # %% Collect the data
    
    processed_tags = []

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        
#        print(' \nRun index: {}'.format(run_ind))

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
        seq_args = [laser_532_delay, laser_589_delay, laser_638_delay, 
                    apd_indices[0]]
        seq_args = [int(el) for el in seq_args]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        
        cxn.pulse_streamer.stream_immediate('laser_delays_test.py', int(num_reps),
                                            seq_args_string)
        
        # Initialize state
        current_tags = []
        current_channels = []
        num_processed_reps = 0

        while num_processed_reps < num_reps:

            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
    
            new_tags, new_channels = cxn.apd_tagger.read_tag_stream()
            new_tags = numpy.array(new_tags, dtype=numpy.int64)
            
            ret_vals = process_raw_buffer(new_tags, new_channels,
                                   current_tags, current_channels,
                                   gate_open_channel, gate_close_channel)
            new_processed_tags, num_new_processed_reps = ret_vals
            # MCC test
            if num_new_processed_reps > 750000:
                print('Processed {} reps out of 10^6 max'.format(num_new_processed_reps))
                print('Tell Matt that the time tagger is too slow!')
            
            num_processed_reps += num_new_processed_reps
            
            processed_tags.extend(new_processed_tags)
            

        cxn.apd_tagger.stop_tag_stream()

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Bin the data
    
    readout_time = (3 * 10**3 + 200)
    readout_time_ps = 1000*(3 * 10**3 + 200)
    
#    start_readout_time_ps = 1000*start_readout_time
#    end_readout_time_ps = 1000*end_readout_time
    binned_samples, bin_edges = numpy.histogram(processed_tags, num_bins,
                                (0, readout_time_ps))
#    print(binned_samples)
    
    # Compute the centers of the bins
    bin_size = readout_time / num_bins
    bin_center_offset = bin_size / 2
    bin_centers = numpy.linspace(0, readout_time, num_bins) + bin_center_offset

# %% Plot

    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    
    ax.plot(bin_centers, binned_samples, 'r-')
    ax.set_title('Laser delay test')
    ax.set_xlabel('Readout time (ns)')
    ax.set_ylabel('Counts')
    
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
                'num_bins': num_bins,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'opti_coords_list': opti_coords_list,
                'opti_coords_list-units': 'V',
                'binned_samples': binned_samples.tolist(),
                'bin_centers': bin_centers.tolist(),
                'processed_tags': processed_tags,
                'processed_tags-units': 'ps',
                }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)
    # %%
    
# %%

if __name__ == '__main__':
    
    apd_indices = [0]
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
                'name': '{}-ensemble'.format(sample_name),
                'expected_count_rate': 1000, 'nd_filter': 'nd_0',
                'pulsed_readout_dur': 1000, 'magnet_angle': 0,
                'resonance_LOW': 2.8059, 'rabi_LOW': 173.5, 'uwave_power_LOW': 9.0, 
                'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}
    
    nv_sig = ensemble
    
    with labrad.connect() as cxn:
        main(cxn, nv_sig, apd_indices)