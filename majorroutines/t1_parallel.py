# -*- coding: utf-8 -*-
"""
T1 measurement routine.

This version of t1 allows the the readout and measurement of all nine possible
combinations of the preparation and readout of the states in relaxation
measurements.

Pass into the function an experiment array, and it will run each experiment
one run at a time. That way, in post processing, we can split the data up by
num_run and see smaller time scales if the values are changing.

The num_runs of each experiment MUST BE THE SAME.


Created on Thu Aug 16 10:01:04 2019

@author: agardill
"""

# %% Imports


import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import time
from random import shuffle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import labrad
from utils.tool_belt import States


# %% Main


def main(nv_sig, apd_indices, t1_exp_array, num_runs):

    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, t1_exp_array, num_runs)


def main_with_cxn(cxn, nv_sig, apd_indices, t1_exp_array, num_runs):

    tool_belt.reset_cfm(cxn)

    # %% Define the times to be used in the sequence
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    polarization_time = shared_params['polarization_dur']
    # time of illumination during which signal readout occurs
    signal_time = polarization_time
    # time of illumination during which reference readout occurs
    reference_time = polarization_time
    pre_uwave_exp_wait_time = shared_params['post_polarization_wait_dur']
    post_uwave_exp_wait_time = shared_params['pre_readout_wait_dur']
    # time between signal and reference without illumination
    sig_to_ref_wait_time = pre_uwave_exp_wait_time + post_uwave_exp_wait_time
    aom_delay_time = shared_params['532_aom_delay']
    rf_delay_time = shared_params['uwave_delay']
    gate_time = nv_sig['pulsed_readout_dur']
    
    # %% Setting initialize and readout states

    uwave_pi_pulse_high = round(nv_sig['rabi_HIGH'] / 2)
    uwave_pi_pulse_low = round(nv_sig['rabi_LOW'] / 2)
    uwave_freq_high = nv_sig['resonance_HIGH']
    uwave_freq_low = nv_sig['resonance_LOW']
    uwave_power_high = nv_sig['uwave_power_HIGH']
    uwave_power_low = nv_sig['uwave_power_LOW']
    
    # %% Extract the number of experiments
    
    num_exp = len(t1_exp_array)
    
    # %% define the file name
    
    file_name = os.path.basename(__file__)
    
    # %% Create lists, that the first index will refer to the experiment
    
    # Lists to fill with taus for each experiment
    tau_master_list = []
    tau_ind_master_list = []
    
    # Lists to store the sig/ref counts for each experiment
    sig_counts_master_list = []
    ref_counts_master_list  = []
    
    avg_sig_counts_master_list = []
    avg_ref_counts_master_list = []
    
    norm_sig_counts_master_list = []
    
    # List for the parameters (freq, pi pulse) for each experiment
    # Format: [[init_state, read_state], init_pi, init_freq, init_pwr,
    #                                   read_pi, read_freq, read_pwr]
    params_master_list = [[] for i in range(num_exp)]
        
    # Nested ists for opticoord and tau_ind for each experiment
    opti_coords_list = [[] for i in range(num_exp)]
    tau_ind_save_list = [[[[] for i in range(num_runs)]] for i in range(num_exp)]

    # Trivial list to add the seq times to obtain a total run time
    exp_time_list = []
    
    # %% Extract the needed information for each experiment and add them to lists
    
    for exp_ind in range(num_exp):
        # For each experiment, take the time values and create a linspace
        min_relaxation_time = int( t1_exp_array[exp_ind][1][0] )
        max_relaxation_time = int( t1_exp_array[exp_ind][1][1] )
        num_steps = t1_exp_array[exp_ind][2]
        num_reps = t1_exp_array[exp_ind][3]
        
        taus = numpy.linspace(min_relaxation_time, max_relaxation_time,
                          num=num_steps, dtype=numpy.int32)
        
        tau_master_list.append(taus)
        
        # also calculate the half length for each tau list to step through, 
        # and later shuffle
        if len(taus) % 2 == 0:
            half_length_taus = int( len(taus) / 2 )
        elif len(taus) % 2 == 1:
            half_length_taus = int( (len(taus) + 1) / 2 )

        # Then we must use this half length to calculate the list of integers to be
        # shuffled for each run
    
        tau_ind_list = list(range(0, half_length_taus))
        tau_ind_master_list.append(tau_ind_list)
        
        # Create empty arrays to fill with data, the indexing will be [exp_ind][num_run][num_steps]

        #append the list with each experiment's specific sized arrays        
        sig_count_line = numpy.empty([num_runs, num_steps], dtype=numpy.uint32)
        sig_count_line[:] = numpy.nan
        ref_count_line = numpy.copy(sig_count_line)
        
        sig_counts_master_list.append(sig_count_line)
        ref_counts_master_list.append(ref_count_line)
        
        # Create a list of the init and read params per experiment
        
        init_state = t1_exp_array[exp_ind][0][0]
        read_state = t1_exp_array[exp_ind][0][1]
        
        # Default values
        uwave_pi_pulse_init = 0
        uwave_freq_init = 2.87
        uwave_power_init = 9.0
        if init_state.value == States.HIGH.value:
            uwave_pi_pulse_init = uwave_pi_pulse_high
            uwave_freq_init = uwave_freq_high
            uwave_power_init = uwave_power_high
        elif init_state.value == tool_belt.States.LOW.value:
            uwave_pi_pulse_init = uwave_pi_pulse_low
            uwave_freq_init = uwave_freq_low
            uwave_power_init = uwave_power_low
    
        # Default values
        uwave_pi_pulse_read = 0
        uwave_freq_read = 2.87
        uwave_power_read = 9.0
        if read_state.value == States.HIGH.value:
            uwave_pi_pulse_read = uwave_pi_pulse_high
            uwave_freq_read = uwave_freq_high
            uwave_power_read = uwave_power_high
        elif read_state.value == States.LOW.value:
            uwave_pi_pulse_read = uwave_pi_pulse_low
            uwave_freq_read = uwave_freq_low
            uwave_power_read = uwave_power_low
        
        # Append the values for each experiment into the master list
        # Format: [[init_state, read_state], init_pi, init_freq, init_pwr,
        #                                   read_pi, read_freq, read_pwr,]
        params_master_list[exp_ind].append([init_state, read_state])
        params_master_list[exp_ind].append(uwave_pi_pulse_init)
        params_master_list[exp_ind].append(uwave_freq_init)
        params_master_list[exp_ind].append(uwave_power_init)
        params_master_list[exp_ind].append(uwave_pi_pulse_read)
        params_master_list[exp_ind].append(uwave_freq_read)
        params_master_list[exp_ind].append(uwave_power_read)
        
        # Analyze the sequence for each experiment
        seq_args = [min_relaxation_time, polarization_time, signal_time, reference_time,
                    sig_to_ref_wait_time, pre_uwave_exp_wait_time,
                    post_uwave_exp_wait_time, aom_delay_time, rf_delay_time,
                    gate_time, uwave_pi_pulse_low, uwave_pi_pulse_high, max_relaxation_time,
                    apd_indices[0], init_state.value, read_state.value]
        seq_args = [int(el) for el in seq_args]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
        seq_time = ret_vals[0]
        seq_time_s = seq_time / (10**9)  # s
        expected_run_time = num_steps * num_reps * num_runs * seq_time_s / 2  # s
        
        seq_time_s = seq_time / (10**9)  # s
        expected_run_time = num_steps * num_reps * num_runs * seq_time_s / 2  # s
        expected_run_time_m = expected_run_time / 60 # m
        
        exp_time_list.append(expected_run_time_m)
    # %% Ask user if they wish to run experiment based on run time

    # Add up the total time for the experiment
    total_exp_time = sum(exp_time_list)
    total_exp_time_h = total_exp_time / 60 # h
    
    # Ask to continue and timeout if no response in 2 seconds?
    print(' \nExpected run time for entire experiment: {:.1f} hours. '.format(total_exp_time_h))
#    return

    # %% Get the starting time of the function, to be used to calculate run time

    startFunctionTime = time.time()
    start_timestamp = tool_belt.get_time_stamp()

    # %% Collect the data

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()   

# %% Start one of the runs

    for run_ind in range(num_runs):

        print(' \nRun index: {}'.format(run_ind))
        
        # Set up the microwaves for the low and high states
        low_sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, States.LOW)
        low_sig_gen_cxn.set_freq(uwave_freq_low)
        low_sig_gen_cxn.set_amp(uwave_power_low)
        low_sig_gen_cxn.uwave_on()
        high_sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, States.HIGH)
        high_sig_gen_cxn.set_freq(uwave_freq_high)
        high_sig_gen_cxn.set_amp(uwave_power_high)
        high_sig_gen_cxn.uwave_on()
            
        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
        
        for exp_ind in range(num_exp):

            # Define the values for this experiment
            init_state = params_master_list[exp_ind][0][0]
            read_state = params_master_list[exp_ind][0][1]
            
            print('\nStarting experiment: ({}, {})'.format(init_state.name,
                                              read_state.name))
            
            num_reps = t1_exp_array[exp_ind][3]
                
            # Optimize
            opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
            opti_coords_list[exp_ind].append(opti_coords)

            # Load the APD
            cxn.apd_tagger.start_tag_stream(apd_indices)
    
            # Shuffle the list of tau indices so that it steps thru them randomly
            shuffle(tau_ind_master_list[exp_ind])
    
            for tau_ind in tau_ind_master_list[exp_ind]:
    
                # 'Flip a coin' to determine which tau (long/shrt) is used first
                rand_boolean = numpy.random.randint(0, high=2)
    
                if rand_boolean == 1:
                    tau_ind_first = tau_ind
                    tau_ind_second = -tau_ind - 1
                elif rand_boolean == 0:
                    tau_ind_first = -tau_ind - 1
                    tau_ind_second = tau_ind
    
                # add the tau indexxes used to a list to save at the end
                tau_ind_save_list[exp_ind][run_ind].append(tau_ind_first)
                tau_ind_save_list[exp_ind][run_ind].append(tau_ind_second)

                
                # Break out of the while if the user says stop
                if tool_belt.safe_stop():
                    break
    
                print(' \nFirst relaxation time: {}'.format(taus[tau_ind_first]))
                print('Second relaxation time: {}'.format(taus[tau_ind_second]))
    
                # Stream the sequence
                seq_args = [taus[tau_ind_first], polarization_time, signal_time, reference_time,
                            sig_to_ref_wait_time, pre_uwave_exp_wait_time,
                            post_uwave_exp_wait_time, aom_delay_time, rf_delay_time,
                            gate_time, uwave_pi_pulse_low, uwave_pi_pulse_high, taus[tau_ind_second],
                            apd_indices[0], init_state.value, read_state.value]
                seq_args = [int(el) for el in seq_args]
                seq_args_string = tool_belt.encode_seq_args(seq_args)
                
                cxn.pulse_streamer.stream_immediate(file_name, int(num_reps),
                                                    seq_args_string)
    
                # Each sample is of the form [*(<sig_shrt>, <ref_shrt>, <sig_long>, <ref_long>)]
                # So we can sum on the values for similar index modulus 4 to
                # parse the returned list into what we want.
                new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
                sample_counts = new_counts[0]
    
                count = sum(sample_counts[0::4])
                sig_counts_master_list[exp_ind, run_ind, tau_ind_first] = count
                print('First signal = ' + str(count))
    
                count = sum(sample_counts[1::4])
                ref_counts_master_list[exp_ind, run_ind, tau_ind_first] = count
                print('First Reference = ' + str(count))
    
                count = sum(sample_counts[2::4])
                sig_counts_master_list[exp_ind, run_ind, tau_ind_second] = count
                print('Second Signal = ' + str(count))
    
                count = sum(sample_counts[3::4])
                ref_counts_master_list[exp_ind, run_ind, tau_ind_second] = count
                print('Second Reference = ' + str(count))

            cxn.apd_tagger.stop_tag_stream()

        # %% Save the data we have incrementally for long measurements

        raw_data = {'start_timestamp': start_timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'gate_time': gate_time,
            'gate_time-units': 'ns',
            't1_exp_array': t1_exp_array,
            't1_exp_array-format': '[[min_relaxation_time, max_relaxation_time],\
                                                num_steps, num_reps]',
            'num_runs': num_runs,
            'params_master_list': params_master_list,
            'params_master_list-format': '[[init_state, read_state], \
                uwave_pi_pulse_init, uwave_freq_init, uwave_power_init, \
                uwave_pi_pulse_read, uwave_freq_read, uwave_power_read]',
            'params_master_list-units': '[[null, null], ns, GHz, dBm, \
                                                        ns, GHz, dBm]',
            'tau_master_list': tau_master_list,
            'tau_master_list-units': 'ns',
            'tau_ind_master_list': tau_ind_master_list,
            'opti_coords_list': opti_coords_list,
            'opti_coords_list-units': 'V',
            'sig_counts_master_list': sig_counts_master_list,
            'sig_counts_master_list-units': 'counts',
            'ref_counts_master_list': ref_counts_master_list,
            'ref_counts_master_list-units': 'counts'}
        
#        # This will continuously be the same file path so we will overwrite
#        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path)

    # %% Hardware clean up

    tool_belt.reset_cfm(cxn)

    # %% Average the counts over the iterations
    for exp_ind in range(num_exp):
        
        avg_sig_counts = numpy.average(sig_counts_master_list[num_exp], axis=0)
        avg_ref_counts = numpy.average(ref_counts_master_list[num_exp], axis=0)
        
        avg_sig_counts_master_list.append(avg_sig_counts)
        avg_ref_counts_master_list.append(avg_ref_counts)

        # Replace x/0=inf with 0
        try:
            norm_avg_sig = avg_sig_counts / avg_ref_counts
        except RuntimeWarning as e:
            print(e)
            inf_mask = numpy.isinf(norm_avg_sig)
            # Assign to 0 based on the passed conditional array
            norm_avg_sig[inf_mask] = 0
            
        norm_sig_counts_master_list.append(norm_avg_sig)

        # %% Plot the t1 signal
        
        init_state = params_master_list[exp_ind, 0, 0]
        read_state = params_master_list[exp_ind, 0, 1]
        
        raw_fig, axes_pack = plt.subplots(1, 2, figsize=(17, 8.5))
        
        ax = axes_pack[0]
        ax.plot(taus / 10**6, avg_sig_counts, 'r-', label = 'signal')
        ax.plot(taus / 10**6, avg_ref_counts, 'g-', label = 'reference')
        ax.set_xlabel('Relaxation time (ms)')
        ax.set_ylabel('Counts')
        ax.legend()
        
        ax = axes_pack[1]
        ax.plot(taus / 10**6, norm_avg_sig, 'b-')
        ax.set_title('T1 Measurement. Initial state: {}, readout state: {}'.format(init_state.name, read_state.name))
        ax.set_xlabel('Relaxation time (ms)')
        ax.set_ylabel('Contrast (arb. units)')
    
        raw_fig.canvas.draw()
        # fig.set_tight_layout(True)
        raw_fig.canvas.flush_events()
        
        # Save each figure
        timestamp = tool_belt.get_time_stamp()
        file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
        tool_belt.save_figure(raw_fig, file_path)
        
        # Sleep for 1.1 seconds so the files don't save over eachother
        time.sleep(1.1)

    # %% Save the data

    endFunctionTime = time.time()

    timeElapsed = endFunctionTime - startFunctionTime

    timestamp = tool_belt.get_time_stamp()

    raw_data = {'timestamp': timestamp,
            'timeElapsed': timeElapsed,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'gate_time': gate_time,
            'gate_time-units': 'ns',
            't1_exp_array': t1_exp_array,
            't1_exp_array-format': '[[min_relaxation_time, max_relaxation_time],\
                                                num_steps, num_reps]',
            'num_runs': num_runs,
            'params_master_list': params_master_list,
            'params_master_list-format': '[[init_state, read_state], \
                uwave_pi_pulse_init, uwave_freq_init, uwave_power_init, \
                uwave_pi_pulse_read, uwave_freq_read, uwave_power_read]',
            'params_master_list-units': '[[null, null], ns, GHz, dBm, \
                                                        ns, GHz, dBm]',
            'tau_master_list': tau_master_list,
            'tau_master_list-units': 'ns',
            'tau_ind_master_list': tau_ind_master_list,
            'opti_coords_list': opti_coords_list,
            'opti_coords_list-units': 'V',
            'norm_sig_counts_master_list': norm_sig_counts_master_list,
            'norm_sig_counts_master_list-units': 'arb',
            'avg_sig_counts_master_list': avg_sig_counts_master_list,
            'avg_sig_counts_master_list-units': 'counts',
            'avg_ref_counts_master_list': avg_ref_counts_master_list,
            'avg_ref_counts_master_list-units': 'counts',
            'sig_counts_master_list': sig_counts_master_list,
            'sig_counts_master_list-units': 'counts',
            'ref_counts_master_list': ref_counts_master_list,
            'ref_counts_master_list-units': 'counts'}
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])    
    tool_belt.save_raw_data(raw_data, file_path)
    
## %%
#
#def decayExp(t, offset, amplitude, decay):
#    return offset + amplitude * numpy.exp(-decay * t)
#
## %% Fitting the data
#
#def t1_exponential_decay(open_file_name, save_file_type):
#
#    directory = 'E:/Team Drives/Kolkowitz Lab Group/nvdata/t1_measurement/'
#
#    # Open the specified file
#    with open(directory + open_file_name + '.txt') as json_file:
#
#        # Load the data from the file
#        data = json.load(json_file)
#        countsT1 = data["norm_avg_sig"]
#        relaxation_time_range = data["relaxation_time_range"]
#        num_steps = data["num_steps"]
#        spin = data["spin_measured?"]
#
#    min_relaxation_time = relaxation_time_range[0]
#    max_relaxation_time = relaxation_time_range[1]
#
#    timeArray = numpy.linspace(min_relaxation_time, max_relaxation_time,
#                              num=num_steps, dtype=numpy.int32)
#
#    offset = 0.8
#    amplitude = 0.1
#    decay = 1/10000 # inverse ns
#
#    popt,pcov = curve_fit(decayExp, timeArray, countsT1,
#                              p0=[offset, amplitude, decay])
#
#    decay_time = 1 / popt[2]
#
#    first = timeArray[0]
#    last = timeArray[len(timeArray)-1]
#    linspaceTime = numpy.linspace(first, last, num=1000)
#
#
#    fig, ax= plt.subplots(1, 1, figsize=(10, 8))
#    ax.plot(timeArray / 10**6, countsT1,'bo',label='data')
#    ax.plot(linspaceTime / 10**6, decayExp(linspaceTime,*popt),'r-',label='fit')
#    ax.set_xlabel('Dark Time (ms)')
#    ax.set_ylabel('Contrast (arb. units)')
#    ax.set_title('T1 of ' + str(spin))
#    ax.legend()
#
#    text = "\n".join((r'$C + A_0 e^{-t / d}$',
#                      r'$C = $' + '%.1f'%(popt[0]),
#                      r'$A_0 = $' + '%.1f'%(popt[1]),
#                      r'$d = $' + "%.3f"%(decay_time / 10**6) + " ms"))
#
#
#    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
#    ax.text(0.70, 0.95, text, transform=ax.transAxes, fontsize=12,
#                            verticalalignment="top", bbox=props)
#
#    fig.canvas.draw()
#    fig.canvas.flush_events()
#
#    fig.savefig(open_file_name + 'replot.' + save_file_type)
