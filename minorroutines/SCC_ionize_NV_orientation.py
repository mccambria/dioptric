# -*- coding: utf-8 -*-
"""
Created on mon Apr 10 10:45:09 2020

This file works to ionize the Nvs in all but one Nv orientation. The idea is
to start with the NVs in NV-. We apply a pi pulse to the NVs in one 
orientation and immediately a red pusle to ionize the other NVs. We then repeat 
this N times, and should essentially only have the NVs in one orientation 
fluorescing.

The file repeats returns a signal and reference counts. The reference counts
is the case where the pi pusle is absent, but the repeated ionization is still 
occuring.

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import json
import time
import matplotlib.pyplot as plt
import labrad
from utils.tool_belt import States
#from scipy.optimize import curve_fit

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

reset_pulse_time = 2#s
# %%

#def sqrt_fnct(x, alpha):
#    return numpy.sqrt(x/alpha)
#
#def plot_n_sweep(num_ion_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None):
#    
#    fig, axes = plt.subplots(1,1, figsize = (10, 8.5)) 
#    ax = axes[0]
#    ax.plot(num_ion_list, sig_counts_avg, 'ro')
#    ax.set_xlabel('Number of repetitions')
#    ax.set_ylabel('Counts (single shot measurement)')
#    ax.set_title(title)
#    ax.legend()
#  
#    return fig

 
#def compile_raw_data(nv_sig, green_optical_power_pd, green_optical_power_mW, 
#                     red_optical_power_pd, red_optical_power_mW, yellow_optical_power_pd, 
#                     yellow_optical_power_mW, shelf_power, num_ion_list, num_reps, 
#                     sig_count_raw, ref_count_raw, sig_counts_avg, snr_list):
#    
#    if type(num_ion_list) != list:
#        num_ion_list.tolist()
#        
#    timestamp = tool_belt.get_time_stamp()
#    raw_data = {'timestamp': timestamp,
#            'nv_sig': nv_sig,
#            'nv_sig-units': tool_belt.get_nv_sig_units(),
#            'green_optical_power_pd': green_optical_power_pd,
#            'green_optical_power_pd-units': 'V',
#            'green_optical_power_mW': green_optical_power_mW,
#            'green_optical_power_mW-units': 'mW',
#            'red_optical_power_pd': red_optical_power_pd,
#            'red_optical_power_pd-units': 'V',
#            'red_optical_power_mW': red_optical_power_mW,
#            'red_optical_power_mW-units': 'mW',
#            'yellow_optical_power_pd': yellow_optical_power_pd,
#            'yellow_optical_power_pd-units': 'V',
#            'yellow_optical_power_mW': yellow_optical_power_mW,
#            'yellow_optical_power_mW-units': 'mW',
#            'shelf_optical_power': shelf_power,
#            'shelf_optical_power-units': 'mW',
#            'num_ion_list': num_ion_list,
#            'num_reps':num_reps,
#            'sig_count_raw': sig_count_raw,
#            'sig_count_raw-units': 'counts',
#            'ref_count_raw': ref_count_raw,
#            'ref_count_raw-units': 'counts',            
#            'sig_counts_avg': sig_counts_avg,
#            'sig_counts_avg-units': 'counts',
#            'sig_counts_avg': sig_counts_avg,
#            'sig_counts_avg-units': 'counts',
#            'snr_list': snr_list,
#            'snr_list-units': 'arb'
#            }
#    return timestamp, raw_data


#%% Main
# Function to actually run sequence and collect counts
def main(nv_sig, apd_indices, num_reps, state):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices,
                            num_reps, state)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices, num_reps, state):

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    num_ionizations = nv_sig['ionization_rep']
    yellow_pol_time =  nv_sig['yellow_pol_dur']
    yellow_pol_pwr = nv_sig['am_589_pol_power']
    shelf_time = nv_sig['pulsed_shelf_dur']
    shelf_pwr = nv_sig['am_589_shelf_power']
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
    ionization_time = nv_sig['pulsed_ionization_dur']
    reionization_time = nv_sig['pulsed_reionization_dur']

    
    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    rabi_period = nv_sig['rabi_{}'.format(state.name)]
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    rf_delay = shared_params['uwave_delay']
    
    wait_time = shared_params['post_polarization_wait_dur']

    num_reps = int(num_reps)
    # Set up our data lists
    opti_coords_list = []

    # Estimate the lenth of the sequance            
    file_name = 'SCC_ionize_NV_orientation.py'
    seq_args = [readout_time, yellow_pol_time, shelf_time, init_ion_time, reionization_time, ionization_time, uwave_pi_pulse,
        wait_time, num_ionizations, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
        apd_indices[0], aom_ao_589_pwr, yellow_pol_pwr, shelf_pwr,  state.value]
    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  #s
    expected_run_time_m = expected_run_time / 60 # m

    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))

    # Collect data

    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=True)
    opti_coords_list.append(opti_coords)
    
    # Turn on the microwaves
    sig_gen_cxn = tool_belt.get_signal_generator_cxn(cxn, state)
    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()
    
    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)

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

def optimize_num_ionizations(nv_sig):
    '''
    This function will apply different repetitions of the pi/ionization combo
    to distinguish between NVs in the single NV orientation
    '''
    apd_indices = [0]
    num_reps = 10**3
#    if not num_ion_list:
    #    num_ion_list = numpy.linspace(0,5, 6)
#    num_ion_list = numpy.linspace(0,150, 76)
    num_ion_list = numpy.linspace(0,30, 16)
    
    # measure laser powers (yellow one is measured at readout power:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    # measure shelf laser power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    norm_diff_list = []
    
    # Step through the pulse lengths for the test laser
    for num_ionizations in num_ion_list:
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(reset_pulse_time) 
        nv_sig['ionization_rep'] = num_ionizations
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        diff = numpy.average(sig_counts) - numpy.average(ref_counts)
        norm_diff = diff / numpy.average(ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        norm_diff_list.append(norm_diff)
 
    #plot
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(num_ion_list, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(num_ion_list, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Number of repetitions')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title('Number of pi-pulse/ionization repetitions')
    ax.legend()
    ax = axes[1]
    ax.plot(num_ion_list, norm_diff_list, 'ro')
    ax.set_xlabel('Number of repetitions')
    ax.set_ylabel('(sig - ref) / ref')
    ax.set_title('Number of pi-pulse/ionization repetitions')
    
    
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
            'yellow_optical_power_pd': yellow_optical_power_pd,
            'yellow_optical_power_pd-units': 'V',
            'yellow_optical_power_mW': yellow_optical_power_mW,
            'yellow_optical_power_mW-units': 'mW',
            'shelf_optical_power': shelf_power,
            'shelf_optical_power-units': 'mW',
            'num_ion_list': num_ion_list.tolist(),
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'ref_counts_avg': ref_counts_avg,
            'ref_counts_avg-units': 'counts',
            'norm_diff_list': norm_diff_list,
            'norm_diff_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-ion_rep')

    tool_belt.save_figure(fig, file_path + '-ion_rep')
    
    print(' \nRoutine complete!')
    return

# %%
    
def optimize_reion_and_init_ion_length(nv_sig):
    '''
    This function will sweep the pulse length for the reionization and initial
    ionization pulse
    '''
    apd_indices = [0]
    num_reps = 10**3
    test_pulse_dur_list = numpy.linspace(0,10*10**3,21).tolist()
    
    # measure laser powers (yellow one is measured at readout power:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    # measure shelf laser power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # measure yellow pol power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_pol_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    yel_pol_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    norm_diff_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        # shine the red laser for a few seconds before the sequence
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(reset_pulse_time)        
        
        nv_sig['pulsed_reionization_dur'] = test_pulse_length
        nv_sig['pulsed_initial_ion_dur'] = test_pulse_length
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        diff = numpy.average(sig_counts) - numpy.average(ref_counts)
        norm_diff = diff / numpy.average(ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        norm_diff_list.append(norm_diff)
 
    #plot
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Pulse length (us)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title('Sweep reionization and initial ionization pulse length')
    ax.legend()
    ax = axes[1]
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, norm_diff_list, 'ro')
    ax.set_xlabel('Pulse length (us)')
    ax.set_ylabel('(sig - ref) / ref')
    ax.set_title('Sweep reionization and initial ionization pulse length')
#    text = 'Yellow pol. pulse power = ' + '%.0f'%(yel_pol_power * 10**3) + ' uW' 
#    ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
#            verticalalignment='top', bbox=props)
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'reset_pulse_time': reset_pulse_time,
            'reset_pulse_time-units': 's',
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
            'shelf_optical_power': shelf_power,
            'shelf_optical_power-units': 'mW',
            'yel_pol_power': yel_pol_power,
            'yel_pol_power-units': 'mW',
            'test_pulse_dur_list': test_pulse_dur_list,
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
            'norm_diff_list': norm_diff_list,
            'norm_diff_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-reion_init_ion_dur')

    tool_belt.save_figure(fig, file_path + '-reion_init_ion_dur')
    
    print(' \nRoutine complete!')
    return

# %%
    
def optimize_init_ion_length(nv_sig):
    '''
    This function will sweep the pulse length for the ibnitial ion pulse
    '''
    apd_indices = [0]
    num_reps = 10**3
    test_pulse_dur_list = numpy.linspace(0,100*10**3,21).tolist()
    
    # measure laser powers (yellow one is measured at readout power:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    # measure shelf laser power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # measure yellow pol power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_pol_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    yel_pol_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
#    snr_list = []
    norm_diff_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        # shine the red laser for a few seconds before the sequence
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(reset_pulse_time)        
        
        nv_sig['pulsed_initial_ion_dur'] = test_pulse_length
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        diff = numpy.average(sig_counts) - numpy.average(ref_counts)
        norm_diff = diff / numpy.average(ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        norm_diff_list.append(norm_diff)
    #plot
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Pulse length (us)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title('Sweep initial ionization pulse length')
    ax.legend()
    ax = axes[1]
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, norm_diff_list, 'ro')
    ax.set_xlabel('Pulse length (us)')
    ax.set_ylabel('(sig - ref) / ref')
    ax.set_title('Sweep initial ionization pulse length')
#    text = 'Yellow pol. pulse power = ' + '%.0f'%(yel_pol_power * 10**3) + ' uW' 
#    ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
#            verticalalignment='top', bbox=props)
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'reset_pulse_time': reset_pulse_time,
            'reset_pulse_time-units': 's',
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
            'shelf_optical_power': shelf_power,
            'shelf_optical_power-units': 'mW',
            'yel_pol_power': yel_pol_power,
            'yel_pol_power-units': 'mW',
            'test_pulse_dur_list': test_pulse_dur_list,
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
            'norm_diff_list': norm_diff_list,
            'norm_diff_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-init_ion_dur')

    tool_belt.save_figure(fig, file_path + '-init_ion_dur')
    
    print(' \nRoutine complete!')
    return
# %%
    
def optimize_reion_length(nv_sig):
    '''
    This function will sweep the pulse length for the reionization pulse
    '''
    apd_indices = [0]
    num_reps = 10**3
    test_pulse_dur_list = numpy.linspace(0,15*10**3,16).tolist()
    
    # measure laser powers (yellow one is measured at readout power:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    # measure shelf laser power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # measure yellow pol power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_pol_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    yel_pol_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
#    snr_list = []
    norm_diff_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        # shine the red laser for a few seconds before the sequence
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(reset_pulse_time)        
        
        nv_sig['pulsed_reionization_dur'] = test_pulse_length
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        diff = numpy.average(sig_counts) - numpy.average(ref_counts)
        norm_diff = diff / numpy.average(ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        norm_diff_list.append(norm_diff)
        
    #plot
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Pulse length (us)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title('Sweep reionization pulse length')
    ax.legend()
    ax = axes[1]
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, norm_diff_list, 'ro')
    ax.set_xlabel('Pulse length (us)')
    ax.set_ylabel('(sig - ref) / ref')
    ax.set_title('Sweep reionization pulse length')
#    text = 'Yellow pol. pulse power = ' + '%.0f'%(yel_pol_power * 10**3) + ' uW' 
#    ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
#            verticalalignment='top', bbox=props)
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'reset_pulse_time': reset_pulse_time,
            'reset_pulse_time-units': 's',
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
            'shelf_optical_power': shelf_power,
            'shelf_optical_power-units': 'mW',
            'yel_pol_power': yel_pol_power,
            'yel_pol_power-units': 'mW',
            'test_pulse_dur_list': test_pulse_dur_list,
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
            'norm_diff_list': norm_diff_list,
            'norm_diff_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-reion_dur')

    tool_belt.save_figure(fig, file_path + '-reion_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_shelf_pulse_length(nv_sig):
    '''
    This function will sweep the pulse length for the shelf pulse
    '''
    apd_indices = [0]
    num_reps = 10**3
    test_pulse_dur_list = numpy.linspace(0,400,21).tolist()
    
    # measure laser powers (yellow one is measured at readout power:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    # measure shelf laser power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    norm_diff_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_shelf_dur'] = test_pulse_length
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(reset_pulse_time)   
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        diff = numpy.average(sig_counts) - numpy.average(ref_counts)
        norm_diff = diff / numpy.average(ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        norm_diff_list.append(norm_diff)
 
    #plot
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(test_pulse_dur_list, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(test_pulse_dur_list, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Pulse length (ns)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title('Sweep shelf pulse length')
    ax.legend()
    ax = axes[1]
    ax.plot(test_pulse_dur_list, norm_diff_list, 'ro')
    ax.set_xlabel('Pulse length (ns)')
    ax.set_ylabel('(sig - ref) / ref')
    ax.set_title('Sweep shelf pulse length')
    text = 'Shelf pulse power = ' + '%.0f'%(shelf_power * 10**3) + ' uW' 
    ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'reset_pulse_time': reset_pulse_time,
            'reset_pulse_time-units': 's',
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
            'shelf_optical_power': shelf_power,
            'shelf_optical_power-units': 'mW',
            'test_pulse_dur_list': test_pulse_dur_list,
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
            'norm_diff_list': norm_diff_list,
            'norm_diff_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-shelf_dur')

    tool_belt.save_figure(fig, file_path + '-shelf_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_shelf_power(nv_sig):
    '''
    This function will sweep the pulse power for the shelf pulse
    '''
    apd_indices = [0]
    num_reps = 10**3
    power_list = numpy.linspace(0.1,0.8,15).tolist()
    
    # measure laser powers (yellow one is measured at readout power:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    # measure yellow pol power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_pol_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    yel_pol_power = tool_belt.calc_optical_power_mW(589, optical_power)
            
    # create some lists for data
    opt_power_list = []
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    norm_diff_list = []
    
    # Step through the pulse lengths for the test laser
    for power in power_list:
        nv_sig['am_589_shelf_power'] = power
        # measure shelf laser power
        optical_power = tool_belt.opt_power_via_photodiode(589, 
                                        AO_power_settings = nv_sig['am_589_shelf_power'], 
                                        nd_filter = nv_sig['nd_filter'])
        shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
        opt_power_list.append(shelf_power)
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(reset_pulse_time)   
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        diff = numpy.average(sig_counts) - numpy.average(ref_counts)
        norm_diff = diff / numpy.average(ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        norm_diff_list.append(norm_diff)
 
    #plot
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(numpy.array(opt_power_list)*10**3, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(numpy.array(opt_power_list)*10**3, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Pulse power (uW)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title('Sweep shelf pulse power')
    ax.legend()
    ax = axes[1]
    ax.plot(numpy.array(opt_power_list)*10**3, norm_diff_list, 'ro')
    ax.set_xlabel('Pulse power (uW)')
    ax.set_ylabel('(sig - ref) / ref')
    ax.set_title('Sweep shelf pulse power')
    text = 'Shelf pulse length = ' + '%.0f'%(nv_sig['pulsed_shelf_dur'] / 10**3) + ' us' 
    ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'reset_pulse_time': reset_pulse_time,
            'reset_pulse_time-units': 's',
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
            'yel_pol_power': yel_pol_power,
            'yel_pol_power-units': 'mW',
            'power_list': power_list,
            'power_list-units': '0-1 V',
            'opt_power_list': opt_power_list,
            'opt_power_list-units': 'mW',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'ref_counts_avg': ref_counts_avg,
            'ref_counts_avg-units': 'counts',
            'norm_diff_list': norm_diff_list,
            'norm_diff_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-shelf_pwr')

    tool_belt.save_figure(fig, file_path + '-shelf_pwr')
    
    print(' \nRoutine complete!')
    return
# %%

def optimize_yellow_pol_length(nv_sig):
    '''
    This function will sweep the pulse length for the yellow pol. pulse
    '''
    apd_indices = [0]
    num_reps = 10**3
    test_pulse_dur_list = numpy.linspace(0,10*10**3,11).tolist()
    
    # measure laser powers (yellow one is measured at readout power:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    # measure shelf laser power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # measure yellow pol power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_pol_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    yel_pol_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    norm_diff_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['yellow_pol_dur'] = test_pulse_length
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(reset_pulse_time)  
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        diff = numpy.average(sig_counts) - numpy.average(ref_counts)
        norm_diff = diff / numpy.average(ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        norm_diff_list.append(norm_diff)
 
    #plot
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Pulse length (us)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title('Sweep yellow pol pulse length')
    ax.legend()
    ax = axes[1]
    ax.plot(numpy.array(test_pulse_dur_list)/10**3, norm_diff_list, 'ro')
    ax.set_xlabel('Pulse length (us)')
    ax.set_ylabel('(sig - ref) / ref')
    ax.set_title('Sweep yellow pol pulse length')
    text = 'Yellow pol. pulse power = ' + '%.0f'%(yel_pol_power * 10**3) + ' uW' 
    ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'reset_pulse_time': reset_pulse_time,
            'reset_pulse_time-units': 's',
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
            'shelf_optical_power': shelf_power,
            'shelf_optical_power-units': 'mW',
            'yel_pol_power': yel_pol_power,
            'yel_pol_power-units': 'mW',
            'test_pulse_dur_list': test_pulse_dur_list,
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
            'norm_diff_list': norm_diff_list,
            'norm_diff_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-yel_pol_dur')

    tool_belt.save_figure(fig, file_path + '-yel_pol_dur')
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_yellow_pol_power(nv_sig):
    '''
    This function will sweep the pulse power for the yellow pol. pulse
    '''
    apd_indices = [0]
    num_reps = 10**3
    power_list = numpy.linspace(0.1,0.8,15).tolist()
    
    # measure laser powers (yellow one is measured at readout power:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    # measure shelf laser power
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
    
    # create some lists for data
    opt_power_list = []
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    norm_diff_list = []
    
    # Step through the pulse lengths for the test laser
    for power in power_list:
        nv_sig['am_589_pol_power'] = power
        # measure yellow pol power
        optical_power = tool_belt.opt_power_via_photodiode(589, 
                                        AO_power_settings = nv_sig['am_589_pol_power'], 
                                        nd_filter = nv_sig['nd_filter'])
        yel_pol_power = tool_belt.calc_optical_power_mW(589, optical_power)
        opt_power_list.append(yel_pol_power)
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(reset_pulse_time)
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        diff = numpy.average(sig_counts) - numpy.average(ref_counts)
        norm_diff = diff / numpy.average(ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        norm_diff_list.append(norm_diff)
 
    #plot
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(numpy.array(opt_power_list)*10**3, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(numpy.array(opt_power_list)*10**3, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Pulse power (uW)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title('Sweep yellow pol pulse power')
    ax.legend()
    ax = axes[1]
    ax.plot(numpy.array(opt_power_list)*10**3, norm_diff_list, 'ro')
    ax.set_xlabel('Pulse power (uW)')
    ax.set_ylabel('(sig - ref) / ref')
    ax.set_title('Sweep yellow pol pulse power')
    text = 'Yellow pol. pulse length = ' + '%.0f'%(nv_sig['yellow_pol_dur'] / 10**3) + ' us' 
    ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Save
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'reset_pulse_time': reset_pulse_time,
            'reset_pulse_time-units': 's',
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
            'shelf_optical_power': shelf_power,
            'shelf_optical_power-units': 'mW',
            'power_list': power_list,
            'power_list-units': '0-1 V',
            'opt_power_list': opt_power_list,
            'opt_power_list-units': 'mW',
            'num_reps':num_reps,
            'sig_count_raw': sig_count_raw,
            'sig_count_raw-units': 'counts',
            'ref_count_raw': ref_count_raw,
            'ref_count_raw-units': 'counts',
            'sig_counts_avg': sig_counts_avg,
            'sig_counts_avg-units': 'counts',
            'ref_counts_avg': ref_counts_avg,
            'ref_counts_avg-units': 'counts',
            'norm_diff_list': norm_diff_list,
            'norm_diff_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-yel_pol_pwr')

    tool_belt.save_figure(fig, file_path + '-yel_pol_pwr')
    
    print(' \nRoutine complete!')
    return
# %%
    
if __name__ == '__main__':
    apd_indices = [0]
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.2, 
            'yellow_pol_dur': 2*10**3, 'am_589_pol_power': 0.20,
            'pulsed_initial_ion_dur': 50*10**3,
            'pulsed_shelf_dur': 100, 'am_589_shelf_power': 0.20,
            'pulsed_ionization_dur': 400, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10*10**3, 'cobalt_532_power': 8,
            'ionization_rep': 7,
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}  
    nv_sig = ensemble
    
#    ion_pulse_list = [400, 450, 500, 550, 600]
    # Run the program
#    for ion_time in ion_pulse_list:
#        nv_sig['pulsed_ionization_dur'] = ion_time
#    optimize_num_ionizations(nv_sig)
#    optimize_reion_and_init_ion_length(nv_sig)
#    optimize_init_ion_length(nv_sig)
#    optimize_reion_length(nv_sig)
#    optimize_shelf_pulse_length(nv_sig)
#    optimize_shelf_power(nv_sig)
#    optimize_yellow_pol_length(nv_sig)
    optimize_yellow_pol_power(nv_sig)
    
    
# %%
#    directory = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/SCC_ionize_NV_orientation/'
#    folder= 'branch_Spin_to_charge/2020_04/'
#    
#    pi_pulse_file = '2020_04_10-18_30_15-hopper-ensemble-ion_rep'
#    no_pi_pulse_file = '2020_04_13-13_04_36-hopper-ensemble-ion_rep'
#
#    # Open the specified file
#    with open(directory + folder + pi_pulse_file + '.txt') as json_file:
#        # Load the data from the file
#        data = json.load(json_file)
#        sig_pi_counts = data["counts_raw"]
#    with open(directory + folder + no_pi_pulse_file + '.txt') as json_file:
#        # Load the data from the file
#        data = json.load(json_file)
#        ref_no_pi_counts = data["counts_raw"]
#    num_ion_list = numpy.linspace(0,1000, 101)
#    sig_counts_sum = []
#    ref_counts_sum = []
#    snr_list = []
#    for i in range(len(num_ion_list)):
#        sig_count = sig_pi_counts[i]
#        ref_count = ref_no_pi_counts[i]
#        snr = tool_belt.calc_snr(sig_count, ref_count)
#        snr_list.append(snr)
        
#    fig, ax = plt.subplots(1,1, figsize = (10, 8.5)) 
#    ax.plot(num_ion_list, snr_list, 'ro')
#    ax.set_xlabel('Number of repetitions')
#    ax.set_ylabel('SNR')
#    ax.set_title('Number of pi-pulse/ionization repetitions')
    