# -*- coding: utf-8 -*-
"""
Created on mon Apr 8 10:45:09 2020

This file runs a sequence of R/G/(R)/Y. The second red pulse is applied in
the signal and not applied for the reference. The difference in these two 
cases relates to our ability to distinguish the charge state.

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
                     yellow_optical_power_mW, test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list):

    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
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
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
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
                     yellow_optical_power_mW, power_list, optical_power_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list):

    power_list = numpy.array(power_list)
        
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
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
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
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, num_reps):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices, num_reps)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices, num_reps):

    tool_belt.reset_cfm(cxn)

# Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    ion_time = nv_sig['pulsed_ionization_dur']
    reion_time = nv_sig['pulsed_reionization_dur']
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
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
    file_name = 'SCC_optimize_pulses_wout_uwaves.py'
    seq_args = [readout_time,init_ion_time, reion_time, ion_time,
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, 
            apd_indices[0], aom_ao_589_pwr]
#    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  #s
    expected_run_time_m = expected_run_time / 60 # m

    # Ask to continue and timeout if no response in 2 seconds?

#    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))

    # Collect data

    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)
    opti_coords_list.append(opti_coords)

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

def optimize_reion_pulse_length(nv_sig, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
        test_pulse_dur_list = [0,5*10**3, 10*10**3, 20*10**3, 30*10**3, 
                        40*10**3, 50*10**3, 100*10**3,200*10**3,300*10**3,
                        400*10**3,500*10**3, 600*10**3, 700*10**3, 800*10**3, 
                        900*10**3, 1*10**6, 2*10**6, 3*10**6 ]
#        test_pulse_dur_list = [0]
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_reionization_dur'] = test_pulse_length
        # shine the red laser before each measurement
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(2) 
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
        
    # Plot
    title = 'Sweep pulse length for 515 nm'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title)
    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-reion_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-reion_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_init_ion_pulse_length(nv_sig, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
        test_pulse_dur_list = [0,5*10**3, 10*10**3, 20*10**3, 30*10**3, 
                        40*10**3, 50*10**3, 100*10**3,200*10**3,300*10**3,
                        400*10**3,500*10**3, 600*10**3, 700*10**3, 800*10**3, 
                        900*10**3, 1*10**6, 2*10**6, 3*10**6 ]
#        test_pulse_dur_list = [0]
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_initial_ion_dur'] = test_pulse_length
        # shine the red laser before each measurement
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(2) 
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
        
    # Plot
    title = 'Sweep pulse length for initial 638 nm'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title)
    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-init_ion_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-init_ion_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%
    
def optimize_init_ion_and_reion_pulse_length(nv_sig, test_pulse_dur_list = None):
    '''
    This function will test init red pulse and green lengths w/ and w/out 
    second red ion pulse.
    
    Both pulse lengths are set to the same value
    '''
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
        test_pulse_dur_list = [0,5*10**3, 10*10**3, 20*10**3, 30*10**3, 
                        40*10**3, 50*10**3, 100*10**3,200*10**3,300*10**3,
                        400*10**3,500*10**3, 600*10**3, 700*10**3, 800*10**3, 
                        900*10**3, 1*10**6, 2*10**6, 3*10**6 ]
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_reionization_dur'] = test_pulse_length
        nv_sig['pulsed_initial_ion_dur'] = test_pulse_length
        # shine the red laser before each measurement
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(2) 
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    # Plot
    title = 'Sweep pulse length for initial 638 nm and 515 nm'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title)

    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-init_ion_and_reion_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-init_ion_and_reion_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_ion_pulse_length(nv_sig, test_pulse_dur_list = [     0,  100, 500, 10**3,  
                                 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6, 5*10**6]):
                              #numpy.linspace(0, 2*10**5, 11)):
    apd_indices = [0]
    num_reps = 1000
#    num_reps = 500
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    print('')
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_ionization_dur'] = test_pulse_length
        print('Ionization pusle length set to {} us'.format(test_pulse_length / 10**3))
        # shine the red laser before each measurement
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(2) 
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    
    # Plot
    title = 'Sweep pulse length for 638 nm'
    text = '\n'.join(('Readout time set to {} ms'.format(nv_sig['pulsed_SCC_readout_dur']/10**6),
                      'Readout power set to ' + '%.1f'%(yellow_optical_power_mW * 10**3) + ' uW'))
    
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title, text= text)

    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, 
                     sig_counts_avg, ref_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-ion_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-ion_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_readout_pulse_length(nv_sig, test_pulse_dur_list  = [10*10**3, 
                               50*10**3, 100*10**3,500*10**3, 
                               1*10**6, 2*10**6, 3*10**6, 4*10**6, 5*10**6, 
                               6*10**6, 7*10**6, 8*10**6, 9*10**6, 1*10**7,
                               2*10**7,3*10**7,4*10**7,5*10**7]):
    apd_indices = [0]
    num_reps = 1000

    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_SCC_readout_dur'] = int(test_pulse_length)
        print('Readout set to {} ms'.format(test_pulse_length/10**6))
        # shine the red laser before each measurement
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(2) 
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    
    # Plot
    title = 'Sweep pulse length for 589 nm'
    text = 'Yellow pulse power set to ' + '%.0f'%(yellow_optical_power_mW*10**3) + ' uW'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title, text = text)
    
    # Save
    timestamp, raw_data = compile_raw_data_length_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     test_pulse_dur_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-readout_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-readout_pulse_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_readout_pulse_power(nv_sig, power_list = None):
    apd_indices = [0]
    num_reps = 1000

    if not power_list:
        power_list = numpy.linspace(0.1,0.8,15).tolist()
       
    # create some lists for data
    optical_power_list = []
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for power in power_list:
        nv_sig['am_589_power'] = power
        
        # measure laser powers:
        green_optical_power_pd, green_optical_power_mW, \
                red_optical_power_pd, red_optical_power_mW, \
                yellow_optical_power_pd, yellow_optical_power_mW = \
                tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        optical_power_list.append(yellow_optical_power_mW)
        # shine the red laser before each measurement
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(2) 
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps)
        
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        snr = tool_belt.calc_snr(sig_count, ref_count)
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(-snr)
    
    # Plot
    title = 'Sweep pulse power for 589 nm'
    text = 'Yellow pulse length set to ' + str(nv_sig['pulsed_SCC_readout_dur']/10**6) + ' ms'
    fig = plot_power_sweep(optical_power_list, sig_counts_avg, ref_counts_avg, 
                          snr_list, title, text = text)
    # Save
    timestamp, raw_data = compile_raw_data_power_sweep(nv_sig, 
                     green_optical_power_pd, green_optical_power_mW, 
                     red_optical_power_pd, red_optical_power_mW, 
                     yellow_optical_power_pd, yellow_optical_power_mW, 
                     power_list, optical_power_list, num_reps, 
                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list)

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-readout_pulse_pwr')

    tool_belt.save_figure(fig, file_path + '-readout_pulse_pwr')
    
    print(' \nRoutine complete!')
    return
# %% Run the files
    
if __name__ == '__main__':
    sample_name = 'johnson'
    
    
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
#    nv_sig = NVA
    
#    test_pulse_dur_list = [   
#        0.,  100.,  200.,  300.,  400.,  500.,  600.,   800.,  1000.,  
#        2000.,  3000.,  4000.,   6000.,   8000.,  10000.,
#        20000.,  30000.,  40000.,    60000.,   80000.,  100000.]
    test_pulse_dur_list = numpy.array([1, 5, 10, 15, 20 ,25, 30, 35, 40, 45])*10**6
    readout_power = numpy.linspace(0.1,0.8, 8)
#    readout_power = [0.3]
    ion_time = numpy.array([0, 0.5, 1, 10, 25, 75, 150, 200])*10**3
#    readout_time = [10**7]

    # Run the program
#    optimize_ion_pulse_length(nv_sig)
#    optimize_reion_pulse_length(nv_sig)
#    optimize_init_ion_pulse_length(nv_sig)
#    
#    for power in readout_power:
#        nv_sig['am_589_power'] = power
#        for ti in ion_time:
#            print(' \nReadout power set to {} V'.format(power))
#            print('Ionization time set to {} us'.format(ti/10**3))
#            nv_sig['pulsed_ionization_dur'] = int(ti)            
#            optimize_readout_pulse_length(nv_sig, test_pulse_dur_list = test_pulse_dur_list)
            
#    optimize_init_ion_and_reion_pulse_length(nv_sig)
#    optimize_readout_pulse_length(nv_sig)
    for nd in ['nd_0', 'nd_0.5', 'nd_1.0', 'nd_1.5']:
        for p in numpy.linspace(0.2, 0.7, 6):
            nv_sig = copy.deepcopy(nv18_2020_11_10)
            nv_sig['nd_filter'] = nd
            nv_sig['am_589_power'] = p
            optimize_readout_pulse_length(nv_sig)
        
#    optimize_readout_pulse_power(nv18_2020_11_10)
    
