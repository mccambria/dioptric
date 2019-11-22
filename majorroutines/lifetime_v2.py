# -*- coding: utf-8 -*-
"""
This is a program to record the lifetime (right now, specifically of the Er 
implanted materials fro mVictor brar's group).

It takes the same structure as a standard t1 measurement. We shine 532 nm 
light, wait some time, and then read out the counts WITHOUT shining 532 nm 
light.

I'm not sure what to normalize the signal to quite yet...

Created on Mon Nov 11 12:49:55 2019

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import time
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import labrad


# %% Functions


def process_raw_buffer(new_timestamps, new_channels,
                       current_timestamps, current_channels,
                       gate_open_channel, gate_close_channel):
    
    # Tack the new data onto the leftover data (leftovers are necessary if
    # the last read contained a gate open without a matching gate close)
    current_timestamps.extend(new_timestamps)
    current_channels.extend(new_channels)
    current_channels_array = numpy.array(current_channels)
    
    # The processing here will be bin_size agnostic. The data structure will
    # be, you guessed it, a list of lists. The deepest list contains the
    # photon timetags for a given sample
    
    # Find gate open clicks
    result = numpy.nonzero(current_channels_array == gate_open_channel)
    gate_open_click_inds = result[0].tolist()

    # Find gate close clicks
    result = numpy.nonzero(current_channels_array == gate_close_channel)
    gate_close_click_inds = result[0].tolist()
    
    new_samples = []
    
    # Loop over the number of closes we have since there are guaranteed to
    # be opens
    for list_ind in range(len(gate_close_click_inds)):
        
        gate_open_click_ind = gate_open_click_inds[list_ind]
        gate_close_click_ind = gate_close_click_inds[list_ind]
        
        # Extract all the counts between these two indices as a single sample
        # Include the gate open/closs for bin reference points
        sample = current_timestamps[gate_open_click_ind:
                                    gate_close_click_ind+1]
        new_samples.append(sample)
        
    # Handle potential leftovers
    if len(gate_close_click_inds) > 0:
        leftover_start = gate_close_click_inds[-1]
        del current_timestamps[0: leftover_start+1]
        del current_channels[0: leftover_start+1]
        
    return new_samples


def bin_samples(samples, num_bins):
    
    # The structure of samples is a list of lists. Each sublist contains the
    # timetag of the opening gate, followed by the timetags of the photon
    # counts. First we'll flatten the samples into a 1D list of timetags
    # relative to each sample's gate open
    flattened_samples = []
    for sample in samples:
        sample = numpy.array(sample, dtype=numpy.int64)
        sample -= sample[0]  # Make relative to gate open
        clipped_sample = sample[1:-1]  # Trim the reference points
        flattened_samples.extend(clipped_sample)
    
    binned_samples = []
    binned_samples, bin_edges = numpy.histogram(flattened_samples, num_bins,
                                                (sample[0], sample[-1]))
    return binned_samples


# %% Main


def main(nv_sig, apd_indices, readout_time,
         num_reps, num_runs, num_bins):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, readout_time,
                      num_reps, num_runs, num_bins)


def main_with_cxn(cxn, nv_sig, apd_indices, readout_time,
                  num_reps, num_runs, num_bins):
    
    if len(apd_indices) > 1:
        msg = 'Currently lifetime only supports single APDs!!'
        raise NotImplementedError(msg)
    
    tool_belt.reset_cfm(cxn)

    # %% Define the times to be used in the sequence
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    # In ns
    polarization_time = 30 * 10**3
    readout_time = int(readout_time)
#    inter_exp_wait_time = 500  # time between experiments

    aom_delay_time = shared_params['532_aom_delay']

    # %% Analyze the sequence

    # pulls the file of the sequence from serves/timing/sequencelibrary
    file_name = os.path.basename(__file__)
    seq_args = [readout_time, polarization_time, 
                aom_delay_time, apd_indices[0]]
    seq_args = [int(el) for el in seq_args]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    seq_time = ret_vals[0]

    # %% Report the expected run time

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * num_runs * seq_time_s  # s
    expected_run_time_m = expected_run_time / 60 # m
    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))
    
    # %% Bit more setup

    # Record the start time
    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()
    
    opti_coords_list = []

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):

        print(' \nRun index: {}'.format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
        opti_coords_list.append(opti_coords)
        
        # Expose the stream
        cxn.apd_tagger.start_tag_stream(apd_indices, apd_indices, False)
    
        # Find the gate channel
        # The order of channel_mapping is APD, APD gate open, APD gate close
        channel_mapping = cxn.apd_tagger.get_channel_mapping()
        gate_open_channel = channel_mapping[1]
        gate_close_channel = channel_mapping[2]
            
        # Stream the sequence
        seq_args = [readout_time, polarization_time, 
                    aom_delay_time, apd_indices[0]]
        seq_args = [int(el) for el in seq_args]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        
        cxn.pulse_streamer.stream_immediate(file_name, int(num_reps),
                                            seq_args_string)
        
        # Initialize state and data
        current_timestamps = []
        current_channels = []
        samples = []

        while len(samples) < num_reps:
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
    
            start = time.time()
            new_timetags, new_channels = cxn.apd_tagger.read_tag_stream()
            new_timetags = numpy.array(new_timetags, dtype=numpy.int64)
            print(time.time() - start)
            continue
            
            new_samples = process_raw_buffer(new_timetags, new_channels,
                                   current_timestamps, current_channels,
                                   gate_open_channel, gate_close_channel)
            
            samples.extend(new_samples)

        cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'readout_time': readout_time,
                    'readout_time-units': 'ns',
                    'num_reps': num_reps,
                    'num_runs': num_runs,
                    'opti_coords_list': opti_coords_list,
                    'opti_coords_list-units': 'V',
                    'samples': samples,
                    }

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Bin the data
    
    binned_samples = bin_samples(samples, num_bins)
    # Compute the centers of the bins
    bin_size = readout_time / num_bins
    bin_center_offset = bin_size / 2
    bin_centers = numpy.linspace(0, readout_time, num_bins) + bin_center_offset

    # %% Plot the t1 signal

    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))

    ax.plot(bin_centers, binned_samples, 'r-')
    ax.set_xlabel('Time after illumination (ns)')
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
                'readout_time': readout_time,
                'readout_time-units': 'ns',
                'num_bins': num_bins,
                'num_reps': num_reps,
                'num_runs': num_runs,
                'opti_coords_list': opti_coords_list,
                'opti_coords_list-units': 'V',
                'binned_samples': binned_samples.tolist(),
                'samples': samples,
                }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(fig, file_path)
    tool_belt.save_raw_data(raw_data, file_path)

# %%

def decayExp(t, amplitude, decay):
    return amplitude * numpy.exp(- t / decay)

def triple_decay(t, a1, d1, a2, d2, a3, d3):
    return decayExp(t, a1, d1) + decayExp(t, a2, d2) + decayExp(t, a3, d3)

# %% Fitting the data

def t1_exponential_decay(open_file_name):

    directory = 'E:/Shared Drives/Kolkowitz Lab Group/nvdata/lifetime/'

    # Open the specified file
    with open(directory + open_file_name + '.txt') as json_file:

        # Load the data from the file
        data = json.load(json_file)
        countsT1_array = numpy.array(data["sig_counts"])
        relaxation_time_range = data["relaxation_time_range"]
        num_steps = data["num_steps"]

    min_relaxation_time = relaxation_time_range[0] / 10**3
    max_relaxation_time = relaxation_time_range[1] / 10**3

    timeArray = numpy.linspace(min_relaxation_time, max_relaxation_time,
                              num=num_steps, dtype=numpy.int32)
    print(max_relaxation_time)
    amplitude = 500
    decay = 100 # us
    init_params = [amplitude, decay]
    
    init_params = [500, 10, 500, 100, 500, 500]
    
    countsT1 = numpy.average(countsT1_array, axis = 0)
#    popt,pcov = curve_fit(decayExp, timeArray, countsT1,
#                              p0=init_params)
    popt,pcov = curve_fit(triple_decay, timeArray, countsT1,
                              p0=init_params)

#    decay_time = popt[1]

    first = timeArray[0]
    last = timeArray[len(timeArray)-1]
    linspaceTime = numpy.linspace(first, last, num=1000)


    fig_fit, ax= plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(timeArray, countsT1,'bo',label='data')
    ax.plot(linspaceTime, triple_decay(linspaceTime,*popt),'r-',label='fit')
    ax.set_xlabel('Wait Time (us)')
    ax.set_ylabel('Counts (arb.)')
    ax.set_title('Lifetime')
    ax.legend()

#    text = "\n".join((r'$A_0 e^{-t / d}$',
#                      r'$A_0 = $' + '%.1f'%(popt[0]),
#                      r'$d = $' + "%.1f"%(decay_time) + " us"))
    text = "\n".join((r'$A_1 e^{-t / d_1} + A_2 e^{-t / d_2} + A_3 e^{-t / d_3}$',
                      r'$A_1 = $' + '%.1f'%(popt[0]),
                      r'$d_1 = $' + "%.1f"%(popt[1]) + " us",
                      r'$A_2 = $' + '%.1f'%(popt[2]),
                      r'$d_2 = $' + "%.1f"%(popt[3]) + " us",
                      r'$A_3 = $' + '%.1f'%(popt[4]),
                      r'$d_3 = $' + "%.1f"%(popt[5]) + " us"))


    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.65, 0.75, text, transform=ax.transAxes, fontsize=12,
                            verticalalignment="top", bbox=props)
    ax.set_yscale("log", nonposy='clip')
    
    fig_fit.canvas.draw()
    fig_fit.canvas.flush_events()

    file_path = directory + open_file_name
    tool_belt.save_figure(fig_fit, file_path+'-triple_fit_semilog')
#    fig.savefig(open_file_name + '-fit.' + save_file_type)

# %%
    

if __name__ == '__main__':
    file_name = '2019_11/2019_11_12-16_31_02-Y2O3-lifetime'
    
    t1_exponential_decay(file_name)