# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:30:14 2020

A file to optimize both the power and readout duration for the 589 laser

@author: agardill
"""

import utils.tool_belt as tool_belt
import numpy
import labrad
import matplotlib.pyplot as plt
import os

# %%

def main_measurement(nv_sig, apd_indices, num_reps):
    
    with labrad.connect() as cxn:
        reion_count, ion_count = main_measurement_w_cxn(cxn,
                            nv_sig, apd_indices, num_reps)
        
    return reion_count, ion_count

def main_measurement_w_cxn(cxn, nv_sig, apd_indices, num_reps):
    
    # Collect the needed parameters from nv_sig or shared_parameters
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    reionization_time = nv_sig['pulsed_reionization_dur']
    ionization_time = nv_sig['pulsed_ionization_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    wait_time = shared_params['post_polarization_wait_dur']
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    
    # set the nd filter
    cxn.filter_slider_ell9k.set_filter(nd_filter)
            
    # Analyze the sequence

    # pulls the file of the sequence from serves/timing/sequencelibrary
    file_name = os.path.basename(__file__)
    seq_args = [readout_time, reionization_time, ionization_time,\
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, \
            apd_indices[0], aom_ao_589_pwr]
    seq_args = [int(el) for el in seq_args]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    seq_time = ret_vals[0]

    # Report the expected run time
    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  # s
    expected_run_time_m = expected_run_time / 60 # m
    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))
    
    # Run the sequence
    cxn.apd_tagger.start_tag_stream(apd_indices)

    seq_args = [readout_time, reionization_time, ionization_time,\
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, \
            apd_indices[0], aom_ao_589_pwr]
#    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)
    
    new_counts = cxn.apd_tagger.read_counter_separate_gates(num_reps)
#    print(len(new_counts))
#    sample_counts = new_counts[0]
    
    # Counts w/out red are even - get every second element starting from 0
    reion_gate_counts = new_counts[0::2]
    reion_count = int(numpy.average(reion_gate_counts))

    # Counts w/ red are odd - sample_counts every second element starting from 1
    ion_gate_counts = new_counts[1::2]
    ion_count = int(numpy.average(ion_gate_counts))
    
    cxn.apd_tagger.stop_tag_stream()
    
    
    return reion_count, ion_count
# %%
    
def optimize_readout_power(nv_sig, apd_indices, num_reps, yellow_power_list):
    # Create some lists to fill
    power_list = []
    g_y_counts_list = []
    r_y_counts_list = []
    # Get the starting timestamp for incriment data
    start_timestamp = tool_belt.get_time_stamp()
    
    for p in range(len(yellow_power_list)):
        nv_sig['am_589_power'] = yellow_power_list[p]
        # Measure the optical powers
        ret_vals = tool_belt.measure_g_r_y_power(nv_sig['am_589_power'], 
                                                 nv_sig['nd_filter'])
        
        green_optical_power_pd, green_optical_power_mW, \
                    red_optical_power_pd, red_optical_power_mW, \
                    yellow_optical_power_pd, yellow_optical_power_mW = ret_vals
        
        print(' \n589 AM set to: ' + str(nv_sig['am_589_power']))
        
        # Run the sequence
        reion_count, ion_count = main_measurement(nv_sig, apd_indices, num_reps)
                
        power_list.append(yellow_optical_power_mW)
        g_y_counts_list.append(reion_count)        
        r_y_counts_list.append(ion_count)
        
        #save data incrimentally
        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    '589_power_list': yellow_power_list,
                    'num_reps': num_reps,
                    'g_y_counts_list': g_y_counts_list,
                    'g_y_counts_list-units': 'counts*ns',
                    'r_y_counts_list': r_y_counts_list,
                    'r_y_counts_list-units': 'counts*ns',
                    'power_list': power_list,
                    'power_list-units': 'mW'
                    }

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path + 'opt_power')
        
    # Plot
    SNR = (numpy.array(g_y_counts_list) - numpy.array(r_y_counts_list)) / \
                numpy.sqrt(numpy.array(g_y_counts_list))
    text = 'Illumnation time: ' + '%.1f'%(nv_sig['pulsed_SCC_readout_dur']/10**3) + ' us'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    ind_fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))

    ax.text(0.55, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.plot(power_list, g_y_counts_list, 'g', label = 'Green/Yellow')
    ax.plot(power_list, r_y_counts_list, 'r', label = 'Red/Yellow')
    ax.set_xlabel('589 power (mW)')
    ax.set_ylabel('Counts')
    ax.legend()
    
    snr_fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    ax.text(0.55, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.plot(power_list, SNR)
    ax.set_xlabel('589 power (mW)')
    ax.set_ylabel('SNR')
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                '589_power_list': yellow_power_list,
                'green_optical_power_pd': green_optical_power_pd,
                'green_optical_power_pd-units': 'V',
                'green_optical_power_mW': green_optical_power_mW,
                'green_optical_power_mW-units': 'mW',
                'red_optical_power_pd': red_optical_power_pd,
                'red_optical_power_pd-units': 'V',
                'red_optical_power_mW': red_optical_power_mW,
                'red_optical_power_mW-units': 'mW',
                'num_reps': num_reps,
                'power_list': power_list,
                'power_list-units': 'mW',
                'g_y_counts_list': g_y_counts_list,
                'g_y_counts_list-units': 'counts',
                'r_y_counts_list': r_y_counts_list,
                'r_y_counts_list-units': 'counts'
                }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(ind_fig, str(file_path + '-opt_power_ind_fig'))
    tool_belt.save_figure(snr_fig, str(file_path + '-opt_power_snr_fig'))
    tool_belt.save_raw_data(raw_data, str(file_path + '-opt_power'))
    print('Run complete!')
    
# %% Untested 3/27
    
def optimize_readout_time(nv_sig, apd_indices, num_reps, readout_time_list):
    g_y_counts_list = []
    r_y_counts_list = []
    start_timestamp = tool_belt.get_time_stamp()
    # Measure the optical powers
    ret_vals = tool_belt.measure_g_r_y_power(nv_sig['am_589_power'], 
                                             nv_sig['nd_filter'])
    
    green_optical_power_pd, green_optical_power_mW, \
                red_optical_power_pd, red_optical_power_mW, \
                yellow_optical_power_pd, yellow_optical_power_mW = ret_vals
    
    for t in range(len(readout_time_list)):
        illumination_time = readout_time_list[t]
        nv_sig['pulsed_SCC_readout_dur'] = illumination_time
        print(' \n589 pulse length set to: ' + str(illumination_time/10**6) + ' ms')
        reion_count, ion_count =main_measurement(nv_sig, apd_indices, num_reps)
                
        g_y_counts_list.append(reion_count)        
        r_y_counts_list.append(ion_count)
        
        #save data incrimentally
        raw_data = {'start_timestamp': start_timestamp,
                    'nv_sig': nv_sig,
                    'nv_sig-units': tool_belt.get_nv_sig_units(),
                    'readout_time_list': readout_time_list,
                    'readout_time_list-units': 'ns',
                    'num_reps': num_reps,
                    'g_y_counts_list': g_y_counts_list,
                    'g_y_counts_list-units': 'counts*ns',
                    'r_y_counts_list': r_y_counts_list,
                    'r_y_counts_list-units': 'counts*ns'
                    }
        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tool_belt.get_file_path(__file__, start_timestamp,
                                            nv_sig['name'], 'incremental')
        tool_belt.save_raw_data(raw_data, file_path + 'opt_power')
        
    # Plot
    SNR = (numpy.array(g_y_counts_list) - numpy.array(r_y_counts_list)) / \
                numpy.sqrt(numpy.array(g_y_counts_list))
    text = 'Illumnation time: ' + '%.1f'%(illumination_time/10**3) + ' us'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    ind_fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))

    ax.text(0.55, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.plot(numpy.array(readout_time_list)/10**6, g_y_counts_list, 'g', label = 'Green/Yellow')
    ax.plot(numpy.array(readout_time_list)/10**6, r_y_counts_list, 'r', label = 'Red/Yellow')
    ax.set_xlabel('589 nm pulse duration (ms)')
    ax.set_ylabel('Counts')
    ax.legend()
    
    snr_fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    ax.text(0.55, 0.85, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.plot(numpy.array(readout_time_list)/10**6, SNR)
    ax.set_xlabel('589 nm pulse duration (ms)')
    ax.set_ylabel('SNR')
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'green_optical_power_pd': green_optical_power_pd,
                'green_optical_power_pd-units': 'V',
                'green_optical_power_mW': green_optical_power_mW,
                'green_optical_power_mW-units': 'mW',
                'red_optical_power_pd': red_optical_power_pd,
                'red_optical_power_pd-units': 'V',
                'red_optical_power_mW': red_optical_power_mW,
                'red_optical_power_mW-units': 'mW',
                'num_reps': num_reps,
                'readout_time_list': readout_time_list,
                'readout_time_list-units': 'ns',
                'g_y_counts_list': g_y_counts_list,
                'g_y_counts_list-units': 'counts*ns',
                'r_y_counts_list': r_y_counts_list,
                'r_y_counts_list-units': 'counts*ns'
                }
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_figure(ind_fig, str(file_path + '-opt_length_ind_fig'))
    tool_belt.save_figure(snr_fig, str(file_path + '-opt_length_snr_fig'))
    tool_belt.save_raw_data(raw_data, str(file_path + '-opt_length'))
    print('Run complete!')
    
# %%

if __name__ == '__main__':
    apd_indices = [0]
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 10**6, 'am_589_power': 0.5, 
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10**6, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 173.5, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}
    nv_sig = ensemble
    
    power_list = numpy.linspace(0.1, 0.7, 13).tolist()
#    power_list = [0.4, 0.6]
#    num_runs = 10
#    readout_time_list = [5*10**5, 10**6, 2*10**6, 3*10**6, 4*10**6, 5*10**6, 
#                         6*10**6, 7*10**6, 8*10**6, 9*10**6, 10**7]
#    readout_time_list = [5*10**5, 10**6]
    num_reps = 5*10**4
#    num_bins = 1000
#    readout_time_array =10**5 * numpy.linspace(1,9,9)

    optimize_readout_power(nv_sig, apd_indices, num_reps, power_list)  
#    optimize_readout_time(nv_sig, apd_indices, num_reps, readout_time_list)
  
    