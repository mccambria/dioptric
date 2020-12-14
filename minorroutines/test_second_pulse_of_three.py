# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:40:44 2020

This file contans the main function of running a sequence with three pulses,
of arbitrary colors, where we run the three pulses twice: once without the 
second pulse and one with the second pulse. This is similar to the function of 
SCC_optimize_pulses_wout_uwaves, however allows us to specify all the pulse
colors, so we can readout in red or green.

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
import labrad
import copy

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# %%
def plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None):
    # turn the list into an array, so we can convert into us
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(test_pulse_dur_list / 10**3, sig_counts_avg, 'ro', 
           label = 'W/ 638 nm pulse')
    ax.plot(test_pulse_dur_list / 10**3, ref_counts_avg, 'ko', 
           label = 'W/out 638 nm pulse')
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title(title)
    ax.legend()
    
    ax = axes[1]    
    ax.plot(test_pulse_dur_list / 10**3, snr_list, 'ro')
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('SNR')
    ax.set_title(title)
    if text:
        ax.text(0.50, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig

def plot_power_sweep(power_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None):
    power_list = numpy.array(power_list)
    
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(power_list * 10**3, sig_counts_avg, 'ro', 
           label = 'W/ 638 nm pulse')
    ax.plot(power_list * 10**3, ref_counts_avg, 'ko', 
           label = 'W/out 638 nm pulse')
    ax.set_xlabel('Test pulse power (uW)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title(title)
    ax.legend()
    
    ax = axes[1]    
    ax.plot(power_list * 10**3, snr_list, 'ro')
    ax.set_xlabel('Test pulse power (uW)')
    ax.set_ylabel('SNR')
    ax.set_title(title)
    if text:
        ax.text(0.50, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig

def compile_raw_data_length_sweep(nv_sig, green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, yellow_optical_power_pd, 
                     yellow_optical_power_mW, init_pulse_time, init_color, test_pulse_time, test_color, 
                     readout_color, test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list):

    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'init_color': init_color,
                'test_color': test_color,
                'readout_color': readout_color,
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
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'init_pulse_time': init_pulse_time,
            'init_pulse_time-units': 'ns',
            'test_pulse_time': test_pulse_time,
            'test_pulse_time-units': 'ns',            
            'test_pulse_dur_list': test_pulse_dur_list.tolist(),
            'test_pulse_dur_list-units': 'ns',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',            
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'ref_counts_avg': ref_counts_avg,
            'ref_counts_avg-units': 'counts',
            'snr_list': snr_list,
            'snr_list-units': 'arb'
            }
    return timestamp, raw_data

def compile_raw_data_power_sweep(nv_sig, green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, yellow_optical_power_pd, 
                     yellow_optical_power_mW, init_pulse_time, init_color, test_pulse_time, test_color, 
                     readout_time, readout_color, power_list, optical_power_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list):

    power_list = numpy.array(power_list)
        
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'init_color': init_color,
                'test_color': test_color,
                'readout_color': readout_color,
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
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'init_pulse_time': init_pulse_time,
            'init_pulse_time-units': 'ns',
            'test_pulse_time': test_pulse_time,
            'test_pulse_time-units': 'ns',     
            'readout_time': readout_time,
            'readout_time-units': 'ns',         
            'power_list': power_list.tolist(),
            'power_list-units': '0-1 V',
            'optical_power_list': optical_power_list,
            'optical_power_list-units': 'mW',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',            
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'ref_counts_avg': ref_counts_avg,
            'ref_counts_avg-units': 'counts',
            'snr_list': snr_list,
            'snr_list-units': 'arb'
            }
    return timestamp, raw_data
#%% Main
def main_data_collect(nv_sig, apd_indices, num_reps, init_pulse_time, test_pulse_time, readout_time,
                      init_color, test_color, readout_color):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_data_collect_with_cxn(cxn, nv_sig, apd_indices, 
                           num_reps, init_pulse_time, test_pulse_time, readout_time,init_color, test_color, readout_color)
        
    return sig_counts, ref_counts
def main_data_collect_with_cxn(cxn, nv_sig, apd_indices, num_reps, init_pulse_time, test_pulse_time, readout_time, init_color, test_color, 
                                      readout_color):

    tool_belt.reset_cfm(cxn)

    # Initial Setup
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
        
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    
    wait_time = shared_params['post_polarization_wait_dur']

    # Set up our data lists
    opti_coords_list = []

    # Estimate the lenth of the sequance            
    file_name = 'test_second_pulse_of_three.py'
    seq_args = [readout_time, init_pulse_time, test_pulse_time, 
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, 
            init_color, test_color, readout_color, 
            apd_indices[0], aom_ao_589_pwr]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    # Collect data

    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)
    opti_coords_list.append(opti_coords)
    
    
    cxn.filter_slider_ell9k.set_filter(nd_filter)

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
    sample_counts = new_counts[0]

    # signal counts are even - get every second element starting from 0
    sig_counts = sample_counts[0::2]

    # ref counts are odd - sample_counts every second element starting from 1
    ref_counts = sample_counts[1::2]
    
    cxn.apd_tagger.stop_tag_stream()
    
    return sig_counts, ref_counts

# %%

def optimize_readout_pulse_length(nv_sig, init_pulse_time, init_color, test_pulse_time, test_color, readout_color, test_pulse_dur_list  = [10*10**3, 
                               50*10**3, 100*10**3,500*10**3, 
                               1*10**6, 2*10**6, 3*10**6, 4*10**6, 5*10**6, 
                               6*10**6, 7*10**6, 8*10**6, 9*10**6, 1*10**7,
                               2*10**7,3*10**7,4*10**7,5*10**7]):
    apd_indices = [0]
    num_reps = 1000

    
        
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        print('Readout set to {} ms'.format(test_pulse_length/10**6))
        
        # Collect the counts
        sig_count, ref_count = main_data_collect(nv_sig, apd_indices, num_reps, 
                     init_pulse_time, test_pulse_time, test_pulse_length,
                      init_color, test_color, readout_color)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    if readout_color == 532:
        optical_power = green_optical_power_mW
    elif readout_color == 589:
        optical_power = yellow_optical_power_mW
    elif readout_color == 638:
        optical_power = red_optical_power_mW
            
    # Plot
    title = 'Sweep pulse length for {} nm\n{} nm init pulse, {} nm test pulse'.format(readout_color, init_color, test_color)
    text = 'Readout pulse power set to ' + '%.0f'%(optical_power*10**3) + ' uW'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title, text = text)
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
                'init_color': init_color,
                'test_color': test_color,
                'readout_color': readout_color,
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
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'init_pulse_time': init_pulse_time,
            'init_pulse_time-units': 'ns',
            'test_pulse_time': test_pulse_time,
            'test_pulse_time-units': 'ns',            
            'test_pulse_dur_list': test_pulse_dur_list.tolist(),
            'test_pulse_dur_list-units': 'ns',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',            
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'ref_counts_avg': ref_counts_avg,
            'ref_counts_avg-units': 'counts',
            'snr_list': snr_list,
            'snr_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-readout_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-readout_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %% Run the files
    
if __name__ == '__main__':
    sample_name = 'goeppert-mayer'
    
    
    nv18_2020_11_10 = { 'coords':[0.179, 0.247, 5.26], 
            'name': '{}-nv18_2020_11_10'.format(sample_name),
            'expected_count_rate': 60, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 4*10**6, 'am_589_power': 0.2, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':20, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}  

    init_color = 638
    test_color = 532
    readout_color = 638
    
    init_pulse_time = 10**3
    test_pulse_time = 10**3

    # Run the program
    optimize_readout_pulse_length(nv18_2020_11_10, init_pulse_time, init_color, 
                                  test_pulse_time, test_color, readout_color)
    
