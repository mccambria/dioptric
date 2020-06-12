# -*- coding: utf-8 -*-
"""
Created on mon Apr 8 10:45:09 2020

This file will test the charge state with variable dark times between either
a green then yellow readout or a red and yellow readout.

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
import labrad

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# %%
def plot_time_sweep(test_pulse_dur_list, sig_count_list, title, text = None):
    # turn the list into an array, so we can convert into us
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    
    fig, ax = plt.subplots(1,1, figsize = (8.5, 8.5)) 
    ax.plot(test_pulse_dur_list / 10**3, sig_count_list, 'bo')
    ax.set_xlabel('Dark time (us)')
    ax.set_ylabel('Counts (single shot)')
    ax.set_title(title)
#    ax.legend()
    
    ax.set_title(title)
    if text:
        ax.text(0.50, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig

#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, num_reps, initial_pulse_time, dark_time, init_pulse_color):

    with labrad.connect() as cxn:
        sig_counts = main_with_cxn(cxn, nv_sig, apd_indices, num_reps, initial_pulse_time, dark_time,  init_pulse_color)
        
    return sig_counts
def main_with_cxn(cxn, nv_sig, apd_indices, num_reps, initial_pulse_time, dark_time, init_pulse_color):

    tool_belt.reset_cfm(cxn)

# Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
        
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    
#    wait_time = shared_params['post_polarization_wait_dur']

    if init_pulse_color == 532:
        init_pulse_delay =laser_515_delay
    elif init_pulse_color == 638:
        init_pulse_delay =laser_638_delay

    # Estimate the lenth of the sequance            
    file_name = 'time_resolved_readout.py'
    seq_args = [readout_time, readout_time, initial_pulse_time, dark_time, 
                init_pulse_delay, aom_589_delay, 
                aom_ao_589_pwr, apd_indices[0],
                init_pulse_color, 589]
    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(file_name, seq_args_string)

#    seq_time = ret_vals[0]
#
#    seq_time_s = seq_time / (10**9)  # s
#    expected_run_time = num_reps * seq_time_s  #s
#    expected_run_time_m = expected_run_time / 60 # m

    # Ask to continue and timeout if no response in 2 seconds?

#    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))

    # Collect data

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
#    print(new_counts)
    sample_counts = new_counts[0]

    # signal counts are even - get every second element starting from 0
    sig_counts = numpy.average(sample_counts)
    
    cxn.apd_tagger.stop_tag_stream()
    
    return sig_counts
# %%

def do_dark_time_w_red(nv_sig, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
#        test_pulse_dur_list = [10**3,5*10**3, 10**4,2*10**4,5*10**4,10**5,2*10**5, 5*10**5, 10**6, 5*10**6,
#                               10**7, 5*10**7]
        test_pulse_dur_list = [10**3,2*10**3, 3*10**3, 4*10**3, 5*10**3,6*10**3, 7*10**3,
                               8*10**3,9*10**3,10**4,2*10**4, 3*10**4, 4*10**4, 10**5]
    initial_pulse_time = 10**7
    # measure laser powers:
#    green_optical_power_pd, green_optical_power_mW, \
#            red_optical_power_pd, red_optical_power_mW, \
#            yellow_optical_power_pd, yellow_optical_power_mW = \
#            tool_belt.measure_g_r_y_power( 
#                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    # create some lists for data
    sig_count_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
#        # shine the red laser before each measurement
#        with labrad.connect() as cxn:
#            cxn.pulse_streamer.constant([7], 0.0, 0.0)
#            time.sleep(2) 
#        print(test_pulse_length)
        sig_count = main(nv_sig, apd_indices, num_reps,initial_pulse_time, test_pulse_length, 638)
        
#        ref_count = [int(el) for el in ref_count]
        
        sig_count_list.append(sig_count)
#        ref_count_raw.append(ref_count)
        
#        snr = tool_belt.calc_snr(sig_count, ref_count)
#        sig_counts_avg.append(numpy.average(sig_count))
#        ref_counts_avg.append(numpy.average(ref_count))
#        snr_list.append(-snr)
        
    # Plot
    title = 'Sweep dark time after {} ms red pulse'.format(initial_pulse_time/10**6)
    fig = plot_time_sweep(test_pulse_dur_list, sig_count_list, title)
    # Save
    timestamp = tool_belt.get_time_stamp()
#    sig_count_list = [int(el) for el in sig_count_list]    
    
    raw_data = {'timestamp': timestamp,
#                'time_elapsed': time_elapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'init_color_ind': '638 nm',
                'initial_pulse_time': initial_pulse_time,
                'initial_pulse_time-units': 'ns',
#                'num_bins': num_bins,
                'num_reps': num_reps,
#                'num_runs': num_runs,
                'test_pulse_dur_list': test_pulse_dur_list,
                'test_pulse_dur_list-units': 'ns',
                'sig_count_list': sig_count_list,
                'sig_count_list-units': 'counts',
                }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-dark_time_w_638')

    tool_belt.save_figure(fig, file_path + '-dark_time_w_638')
    
    print(' \nRoutine complete!')
    return

# %%

def do_dark_time_w_green(nv_sig, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 1000
    if not test_pulse_dur_list:
        test_pulse_dur_list = [10**3,5*10**3, 10**4,2*10**4,5*10**4,10**5,2*10**5, 5*10**5, 10**6, 5*10**6,
                               10**7, 5*10**7]
    initial_pulse_time = 10**6
    # measure laser powers:
#    green_optical_power_pd, green_optical_power_mW, \
#            red_optical_power_pd, red_optical_power_mW, \
#            yellow_optical_power_pd, yellow_optical_power_mW = \
#            tool_belt.measure_g_r_y_power( 
#                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    # create some lists for data
    sig_count_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
#        # shine the red laser before each measurement
#        with labrad.connect() as cxn:
#            cxn.pulse_streamer.constant([7], 0.0, 0.0)
#            time.sleep(2) 
#        print(test_pulse_length)
        sig_count = main(nv_sig, apd_indices, num_reps,initial_pulse_time, test_pulse_length, 532)
        
#        ref_count = [int(el) for el in ref_count]
        
        sig_count_list.append(sig_count)
#        ref_count_raw.append(ref_count)
        
#        snr = tool_belt.calc_snr(sig_count, ref_count)
#        sig_counts_avg.append(numpy.average(sig_count))
#        ref_counts_avg.append(numpy.average(ref_count))
#        snr_list.append(-snr)
        
    # Plot
    title = 'Sweep dark time after {} ms green pulse'.format(initial_pulse_time/10**6)
    fig = plot_time_sweep(test_pulse_dur_list, sig_count_list, title)
    # Save
    timestamp = tool_belt.get_time_stamp()
#    sig_count_list = [int(el) for el in sig_count_list]    
    
    raw_data = {'timestamp': timestamp,
#                'time_elapsed': time_elapsed,
                'nv_sig': nv_sig,
                'nv_sig-units': tool_belt.get_nv_sig_units(),
                'init_color_ind': '532 nm',
                'initial_pulse_time': initial_pulse_time,
                'initial_pulse_time-units': 'ns',
#                'num_bins': num_bins,
                'num_reps': num_reps,
#                'num_runs': num_runs,
                'test_pulse_dur_list': test_pulse_dur_list,
                'test_pulse_dur_list-units': 'ns',
                'sig_count_list': sig_count_list,
                'sig_count_list-units': 'counts',
                }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-dark_time_w_532')

    tool_belt.save_figure(fig, file_path + '-dark_time_w_532')
    
    print(' \nRoutine complete!')
    return

# %% Run the files
    
if __name__ == '__main__':
    sample_name = 'bachman-A1'
    ensemble_B1 = { 'coords':[-0.404, 0.587, 5.39],
            'name': '{}-A6'.format(sample_name),
            'expected_count_rate': 6600, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7555,"rabi_LOW": 385.1, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9878,"rabi_HIGH": 582.3,"uwave_power_HIGH": 10.0} 
    nv_sig = ensemble_B1
    
#    do_dark_time_w_green(nv_sig)
    do_dark_time_w_red(nv_sig)
    
