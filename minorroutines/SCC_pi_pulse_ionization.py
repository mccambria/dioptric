# -*- coding: utf-8 -*-
"""
Created on mon Apr 6 10:45:09 2020

This file will perform the sequence G/R/Y, two times, one with a pi pulse
occuring during the rd ionization process and the second without the pi pulse.

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import json
import matplotlib.pyplot as plt
import labrad
from utils.tool_belt import States
import minorroutines.photonstatistics as ps
import time

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# %%
def get_Probability_distribution(aList):

    def get_unique_value(aList):
        unique_value_list = []
        for i in range(0,len(aList)):
            if aList[i] not in unique_value_list:
                unique_value_list.append(aList[i])
        return unique_value_list
    unique_value = get_unique_value(aList)
    relative_frequency = []
    for i in range(0,len(unique_value)):
        relative_frequency.append(aList.count(unique_value[i])/ (len(aList)))

    return unique_value, relative_frequency


def calc_snr(sig_count, ref_count):
    #v1
#    dif = numpy.array(ref_count) - numpy.array(sig_count)
#    noise = numpy.sqrt(ref_count)
#    snr = dif / noise
#    snr = numpy.average(snr)
    # v2
    sig_count_avg = numpy.average(sig_count)
    ref_count_avg = numpy.average(ref_count)
    dif = sig_count_avg - ref_count_avg
    noise = numpy.sqrt(ref_count_avg)
    snr = dif / noise
    
    return snr

def plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None):

    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(test_pulse_dur_list / 10**3, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(test_pulse_dur_list / 10**3, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
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
        ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig

def plot_power_sweep(power_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None):

    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(numpy.array(power_list) * 10**3, sig_counts_avg, 'ro', 
           label = 'W/ pi-pulse')
    ax.plot(numpy.array(power_list) * 10**3, ref_counts_avg, 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Test pulse power (uW)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title(title)
    ax.legend()
    
    ax = axes[1]    
    ax.plot(numpy.array(power_list) * 10**3, snr_list, 'ro')
    ax.set_xlabel('Test pulse power (uW)')
    ax.set_ylabel('SNR')
    ax.set_title(title)
    if text:
        ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig
 

#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, num_reps, state, plot = True):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices,  
                           num_reps, state, plot)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices,
                  num_reps, state, plot):

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
    ionization_time = nv_sig['pulsed_ionization_dur']
    reionization_time = nv_sig['pulsed_reionization_dur']
    shelf_time = nv_sig['pulsed_shelf_dur']
    shelf_power = nv_sig['am_589_shelf_power']
    
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
    file_name = 'SCC_pi_pulse_ionization_ion_edit.py'
    seq_args = [readout_time, init_ion_time, reionization_time, ionization_time, uwave_pi_pulse,
        shelf_time , wait_time, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
        apd_indices[0], aom_ao_589_pwr, shelf_power, state.value]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  #s
    expected_run_time_m = expected_run_time / 60 # m

    # Ask to continue and timeout if no response in 2 seconds?

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

#    seq_args = [readout_time,init_ion_time, reionization_time, ionization_time, uwave_pi_pulse,
#        shelf_time, wait_time, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
#        apd_indices[0], aom_ao_589_pwr,  shelf_power, state.value]
#    print(seq_args)
#    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
    sample_counts = new_counts[0]
#    print(len(sample_counts))
    # signal counts are even - get every second element starting from 0
    sig_counts = sample_counts[0::2]

    # ref counts are odd - sample_counts every second element starting from 1
    ref_counts = sample_counts[1::2]
    
    cxn.apd_tagger.stop_tag_stream()
    
    # Plot
    if plot:
        unique_value1, relative_frequency1 = get_Probability_distribution(list(ref_counts))
        unique_value2, relative_frequency2 = get_Probability_distribution(list(sig_counts))
    
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        ax.plot(unique_value2, relative_frequency2, 'ro', label='pi-pulse')
        ax.plot(unique_value1, relative_frequency1, 'ko', label='pi-pulse absent')
        ax.set_xlabel('number of photons (n)')
        ax.set_ylabel('P(n)')
        ax.legend()
      
        
        fig2, ax2 = plt.subplots(1,1, figsize = (10, 8.5))
    
    #    time_axe_sig = ps.get_time_axe(seq_time_s*2, readout_time*10**-9,sig_counts)
        sig_counts_cps = ps.get_photon_counts(readout_time*10**-9, sig_counts)
        sig_len=len(sig_counts_cps)
    
    #    time_axe_ref = numpy.array(ps.get_time_axe(seq_time_s*2, readout_time*10**-9,ref_counts)) + seq_time_s
        ref_counts_cps = ps.get_photon_counts(readout_time*10**-9, ref_counts)
        ref_len=len(ref_counts_cps)
    
        ax2.plot(numpy.linspace(0,sig_len-1, sig_len), numpy.array(sig_counts_cps)/10**3, 'r', label='pi-pulse')
        ax2.plot(numpy.linspace(0,ref_len-1, ref_len), numpy.array(ref_counts_cps)/10**3, 'k', label='pi-pulse absent')
        ax2.set_xlabel('Rep number')
        ax2.set_ylabel('photon counts (kcps)')
        ax2.legend()
        
        timestamp = tool_belt.get_time_stamp()
        file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
        tool_belt.save_figure(fig, file_path + '-red_pulse_dur-dist')  
        tool_belt.save_figure(fig2, file_path + '-red_pulse_dur-counts')  
        
    return sig_counts, ref_counts

# %%

def optimize_initial_ion_pulse_length(nv_sig):
    '''
    This function will test an initial red pulse, that starts the NV in NV0
    '''
    apd_indices = [0]
    num_reps = 10**3
#    test_pulse_dur_list = numpy.linspace(0,10**6,11)
    test_pulse_dur_list = numpy.array([0.,  10000.,  20000.,  30000.,  40000.,  50000.,  60000.,
        70000.,  80000.,  90000., 100000.,
        200000.,  300000.,  400000.,  500000.,
        600000.,  700000.,  800000.,  900000., 1000000.])
    
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
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps,
                                    States.LOW, plot = False)
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        avg_snr = calc_snr(sig_count, ref_count)
        
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(avg_snr)
 
    #plot
    title = 'Sweep pusle length for initial 638 nm'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg,
                          snr_list, title)
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
    tool_belt.save_raw_data(raw_data, file_path + '-init_red_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-init_red_pulse_dur')
#    tool_belt.save_figure(fig_snr, file_path + '-pulse_dur_snr') 
    
    print(' \nRoutine complete!')
    return

# %%

def optimize_ion_pulse_length(nv_sig):
    '''
    This function will test red pulse lengths between 0 and 600 ns on the LOW
    NV state.
    '''
    apd_indices = [0]
    num_reps = 10**3
    test_pulse_dur_list = numpy.linspace(0,1500,16)
#    test_pulse_dur_list = numpy.linspace(0,100,2)
    
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
        nv_sig['pulsed_ionization_dur'] = test_pulse_length
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps,
                                    States.LOW, plot = False)
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        avg_snr = calc_snr(sig_count, ref_count)
        
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(avg_snr)
 
    #plot
    title = 'Sweep pusle length for 638 nm'
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg,
                          snr_list, title)
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
    tool_belt.save_raw_data(raw_data, file_path + '-red_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-red_pulse_dur')
#    tool_belt.save_figure(fig_snr, file_path + '-pulse_dur_snr') 
    
    print(' \nRoutine complete!')
    return
# %%
def optimize_shelf_pulse_length(nv_sig):
    '''
    This function will test yellow shelf pulse lengths between 0 and 200 ns on the LOW
    NV state.
    '''
    apd_indices = [0]
    num_reps = 10**3
    test_pulse_dur_list = numpy.linspace(0,200,11)
#    test_pulse_dur_list = numpy.linspace(0,200,3)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
#    cxn.pulse_streamer.constant([], 0.0, nv_sig['am_589_shelf_power'])
    
    # create some lists for data
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # measure the power of the test pulse
    optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
                                    nd_filter = nv_sig['nd_filter'])
    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)

    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_shelf_dur'] = test_pulse_length
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps,
                                    States.LOW, plot = False)
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        avg_snr = calc_snr(sig_count, ref_count)
        
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(avg_snr)
 
    #plot
    title = 'Sweep pusle length for 589 nm shelf'
    text = 'Shelf pulse power = ' + '%.0f'%(shelf_power * 10**3) + ' mW' 
    fig = plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = text)
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
    tool_belt.save_raw_data(raw_data, file_path + '-shelf_pulse_dur')

    tool_belt.save_figure(fig, file_path + '-shelf_pulse_dur')
    
    print(' \nRoutine complete!')
    return
#%%
def optimize_shelf_pulse_power(nv_sig):
    '''
    This function will test yellow shelf pulse powers between 0 and ~ 200 mW 
    on the LOW NV state.
    '''
    apd_indices = [0]
    num_reps = 10**3
    test_pulse_power_list = numpy.linspace(0,0.8,9)
#    test_pulse_power_list = numpy.linspace(0,0.8,2)
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
#    cxn.pulse_streamer.constant([], 0.0, nv_sig['am_589_shelf_power'])
    
    # create some lists for data
    power_list = []
    sig_count_raw = []
    ref_count_raw = []
    sig_counts_avg = []
    ref_counts_avg = []
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_power in test_pulse_power_list:
        # Get the optical power (mW) of the shelf pulse
        optical_power = tool_belt.opt_power_via_photodiode(589, 
                                    AO_power_settings = test_pulse_power, 
                                    nd_filter = nv_sig['nd_filter'])
        shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
#        print(shelf_power)
        power_list.append(shelf_power)

        nv_sig['am_589_shelf_power'] = test_pulse_power
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps,
                                    States.LOW, plot = False)
        sig_count = [int(el) for el in sig_count]
        ref_count = [int(el) for el in ref_count]
        
        sig_count_raw.append(sig_count)
        ref_count_raw.append(ref_count)
        
        avg_snr = calc_snr(sig_count, ref_count)
        
        sig_counts_avg.append(numpy.average(sig_count))
        ref_counts_avg.append(numpy.average(ref_count))
        snr_list.append(avg_snr)
 
    #plot
    title = 'Sweep power for 589 nm shelf'
    text = 'Shelf pulse length = {} ns'.format(nv_sig['pulsed_shelf_dur'])
    fig = plot_power_sweep(power_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = text)
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
            'test_pulse_power_list': test_pulse_power_list.tolist(),
            'test_pulse_power_list-units': '0-1 V',
            'power_list': power_list,
            'power_list-units': 'mW',
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
    tool_belt.save_raw_data(raw_data, file_path + '-shelf_pulse_pwr')

    tool_belt.save_figure(fig, file_path + '-shelf_pulse_pwr')
    
    print(' \nRoutine complete!')
    return
# %%
    
if __name__ == '__main__':
    apd_indices = [0]
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0.5',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 200*10**3,
            'pulsed_shelf_dur': 50, 'am_589_shelf_power': 0.3,
            'pulsed_ionization_dur': 800, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 3*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}  
    nv_sig = ensemble
    
    # Run the program
    optimize_initial_ion_pulse_length(nv_sig)
#    optimize_ion_pulse_length(nv_sig)
#    optimize_shelf_pulse_length(nv_sig)
#    optimize_shelf_pulse_power(nv_sig)
    
#    sig_counts, ref_counts = main(nv_sig, apd_indices, 10**3, States.LOW)
    
    # %% Replot data
#    directory = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/SCC_pi_pulse_ionization/'
#    folder= 'branch_Spin_to_charge/2020_04/'
#    
#    open_file_name = '2020_04_06-19_56_57-hopper-ensemble-red_pulse_dur'
#
#    # Open the specified file
#    with open(directory + folder + open_file_name + '.txt') as json_file:
#
#        # Load the data from the file
#        data = json.load(json_file)
#        sig_count_raw = data["sig_count_raw"]
#        ref_count_raw = data["ref_count_raw"]
#        sig_counts_avg = data["sig_counts_avg"]
#        ref_counts_avg = data["ref_counts_avg"]
#        test_pulse_dur_list = numpy.array(data["test_pulse_dur_list"])
#    sig_counts_sum = []
#    ref_counts_sum = []
#    snr_list = []
#    for i in range(len(test_pulse_dur_list)):
#        sig_count = sig_count_raw[i]
#        ref_count = ref_count_raw[i]
#        snr = calc_snr(sig_count, ref_count)
#        snr_list.append(snr)
#    plot_time_sweep(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, snr_list)
    