# -*- coding: utf-8 -*-
"""
Created on mon Apr 6 10:45:09 2020

This file will perform the sequence R/G/R/Y, two times, one with a pi pulse
occuring before the red ionization process and the second without the pi pulse.

You can use it to find the optimum ionization pulse, as well as the 
reionization pulse and spin shelf pulse.

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
# import json
# import time
import copy
import matplotlib.pyplot as plt
import labrad
from utils.tool_belt import States
from random import shuffle
import scipy.stats as stats
# import minorroutines.photonstatistics as ps
# from scipy.optimize import curve_fit

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# %%

def sqrt_fnct(x, alpha):
    return numpy.sqrt(x/alpha)
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

def plot_snr_v_dur(dur_list, sig_counts_avg, ref_counts_avg, 
                   sig_counts_ste, ref_counts_ste, 
                   snr_list, title, text = None):
    # turn the list into an array, so we can convert into us
    dur_list = numpy.array(dur_list)
    
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.errorbar(dur_list / 10**3, sig_counts_avg, yerr=sig_counts_ste, fmt= 'ro', 
           label = 'W/ pi-pulse')
    ax.errorbar(dur_list / 10**3, ref_counts_avg, yerr=ref_counts_ste, fmt= 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title(title)
    ax.legend()
    
    ax = axes[1]    
    ax.plot(dur_list / 10**3, snr_list, 'ro')
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('SNR')
    ax.set_title(title)
    if text:
        ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
  
    return fig


#%% Main
# Function to actually run sequence and collect counts
def measure(nv_sig, apd_indices, num_reps, state, plot = True):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = measure_with_cxn(cxn, nv_sig, apd_indices,  
                           num_reps, state, plot)
        
    return sig_counts, ref_counts
def measure_with_cxn(cxn, nv_sig, apd_indices,
                  num_reps, state, plot):

    tool_belt.reset_cfm(cxn)

    # Optimize
    opti_coords_list = []
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices)
    opti_coords_list.append(opti_coords)

    # Initial Calculation and setup
    tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv-_reionization_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv0_ionization_laser")
    # tool_belt.set_filter(cxn, nv_sig, "spin_shelf_laser")
        
    readout_time = nv_sig['charge_readout_dur']
    ionization_time = nv_sig['nv0_ionization_dur']
    reionization_time = nv_sig['nv-_reionization_dur']
    shelf_time = nv_sig['spin_shelf_dur']
    
    readout_power = tool_belt.set_laser_power(
        cxn, nv_sig, "charge_readout_laser"
    )
    reion_power = tool_belt.set_laser_power(
        cxn, nv_sig, "nv-_reionization_laser"
    )
    ion_power = tool_belt.set_laser_power(
        cxn, nv_sig, "nv0_ionization_laser"
    )
    shelf_power = tool_belt.set_laser_power(
        cxn, nv_sig, "spin_shelf_laser"
    )

    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    rabi_period = float(nv_sig['rabi_{}'.format(state.name)])
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    
    
    green_laser_name = nv_sig['nv-_reionization_laser']
    red_laser_name = nv_sig['nv0_ionization_laser']
    yellow_laser_name = nv_sig['charge_readout_laser']
    sig_gen_name = tool_belt.get_signal_generator_name_no_cxn(state)
    
    num_reps = int(num_reps)

    # Estimate the lenth of the sequance            
    file_name = 'rabi_scc.py'        
    seq_args = [readout_time, reionization_time, ionization_time, uwave_pi_pulse,
        shelf_time ,  uwave_pi_pulse, 
        green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
        apd_indices[0], reion_power, ion_power, shelf_power, readout_power]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
      
    # print(seq_args)
    # return
    
    seq_time = int(ret_vals[0])

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  #s
    # expected_run_time_m = expected_run_time / 60 # m

    # print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))


    # Collect data

    
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
    # print(sample_counts)

    # signal counts are even - get every second element starting from 0
    sig_counts = sample_counts[0::2]
    # print(sig_counts)

    # ref counts are odd - sample_counts every second element starting from 1
    ref_counts = sample_counts[1::2]
    # print(ref_counts)
    
    cxn.apd_tagger.stop_tag_stream()
    tool_belt.reset_cfm(cxn)

    return sig_counts, ref_counts

# %%

def determine_ionization_dur(nv_sig, apd_indices, num_reps, ion_durs=None):
    '''
    This function will test red pulse lengths between 0 and 600 ns on the LOW
    NV state.
    '''
    state = States.LOW
    if ion_durs is None:
        ion_durs = numpy.linspace(0,1000,11)
  
    num_steps = len(ion_durs)
    
    # create some arrays for data
    sig_counts_array = numpy.zeros(num_steps)
    sig_counts_ste_array = numpy.copy(sig_counts_array)
    ref_counts_array = numpy.copy(sig_counts_array)
    ref_counts_ste_array = numpy.copy(sig_counts_array)
    snr_array = numpy.copy(sig_counts_array)
    

    dur_ind_master_list = []
    
    dur_ind_list = list(range(0, num_steps))
    shuffle(dur_ind_list)
    
    # Step through the pulse lengths for the test laser
    for ind in dur_ind_list:
        t = ion_durs[ind]
        dur_ind_master_list.append(ind)
        print('Ionization dur: {} ns'.format(t))
        nv_sig_copy = copy.deepcopy(nv_sig)
        nv_sig_copy['nv0_ionization_dur'] = t
        sig_counts, ref_counts = measure(nv_sig_copy, apd_indices, num_reps,
                                    state, plot = False)
        # print(sig_counts)
        
        sig_count_avg = numpy.average(sig_counts)
        sig_counts_ste = stats.sem(sig_counts)
        ref_count_avg = numpy.average(ref_counts)
        ref_counts_ste = stats.sem(ref_counts)
            
        sig_counts_array[ind] = sig_count_avg
        sig_counts_ste_array[ind] = sig_counts_ste
        ref_counts_array[ind] = ref_count_avg
        ref_counts_ste_array[ind] = ref_counts_ste
        
        avg_snr = tool_belt.calc_snr(sig_counts, ref_counts)
        snr_array[ind] = avg_snr
 
    #plot
    title = 'Sweep ionization pulse duration'
    fig = plot_snr_v_dur(ion_durs, sig_counts_array, ref_counts_array, 
                         sig_counts_ste_array, ref_counts_ste_array,
                          snr_array, title)
    # Save
    
    ion_durs = numpy.array(ion_durs)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'ion_durs': ion_durs.tolist(),
            'ion_durs-units': 'ns',
            'num_reps':num_reps,            
            'sig_counts_array': sig_counts_array.tolist(),
            'sig_counts_ste_array': sig_counts_ste_array.tolist(),
            'ref_counts_array': ref_counts_array.tolist(),
            'ref_counts_ste_array': ref_counts_ste_array.tolist(),
            'snr_list': snr_array.tolist(),
            'dur_ind_master_list': dur_ind_master_list
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'] + '-ion_pulse_dur')
    
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)
    
    print(' \nRoutine complete!')
    return

# # %%
# def determine_reion_dur(nv_sig):
#     '''
#     This function will test green pulse lengths on the LOW NV state.
#     '''
#     apd_indices = [0]
#     num_reps = 10**4
# #    test_pulse_dur_list = [0,5*10**3, 10*10**3, 20*10**3, 30*10**3, 40*10**3, 50*10**3, 
# #                           100*10**3,200*10**3,300*10**3,400*10**3,500*10**3,
# #                           600*10**3, 700*10**3, 800*10**3, 900*10**3, 
# #                           1*10**6, 2*10**6, 3*10**6 ]
#     test_pulse_dur_list = numpy.linspace(0,5*10**5,11)
    
#     # measure laser powers:
#     # green_optical_power_pd, green_optical_power_mW, \
#     #         red_optical_power_pd, red_optical_power_mW, \
#     #         yellow_optical_power_pd, yellow_optical_power_mW = \
#     #         tool_belt.measure_g_r_y_power( 
#     #                               nv_sig['am_589_power'], nv_sig['nd_filter'])
    
#     # create some lists for data
#     sig_count_raw = []
#     ref_count_raw = []
#     sig_counts_avg = []
#     ref_counts_avg = []
#     snr_list = []
    
#     # Step through the pulse lengths for the test laser
#     for test_pulse_length in test_pulse_dur_list:
#         nv_sig['nv-_reionization_dur'] = test_pulse_length
#         sig_count, ref_count = measure(nv_sig, apd_indices, num_reps,
#                                     States.LOW, plot = False)
#         sig_count = [int(el) for el in sig_count]
#         ref_count = [int(el) for el in ref_count]
        
#         sig_count_raw.append(sig_count)
#         ref_count_raw.append(ref_count)
        
#         avg_snr = tool_belt.calc_snr(sig_count, ref_count)
        
#         sig_counts_avg.append(numpy.average(sig_count))
#         ref_counts_avg.append(numpy.average(ref_count))
#         snr_list.append(avg_snr)
 
#     #plot
#     title = 'Sweep pusle length for 532 nm'
#     fig = plot_snr_v_dur(test_pulse_dur_list, sig_counts_avg, ref_counts_avg,
#                           snr_list, title)
#     # Save    
#     timestamp = tool_belt.get_time_stamp()
#     raw_data = {'timestamp': timestamp,
#             'nv_sig': nv_sig,
#             'ion_durs': ion_durs.tolist(),
#             'ion_durs-units': 'ns',
#             'num_reps':num_reps,
#             'sig_count_raw': sig_count_raw,
#             'sig_count_raw-units': 'counts',
#             'ref_count_raw': ref_count_raw,
#             'ref_count_raw-units': 'counts',            
#             'sig_counts_avg': sig_counts_avg,
#             'sig_counts_avg-units': 'counts',
#             'sig_counts_avg': sig_counts_avg,
#             'sig_counts_avg-units': 'counts',
#             'snr_list': snr_list,
#             'snr_list-units': 'arb'
#             }

#     file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
#     tool_belt.save_raw_data(raw_data, file_path + '-reion_pulse_dur')

#     tool_belt.save_figure(fig, file_path + '-reion_pulse_dur')
    
#     print(' \nRoutine complete!')
#     return

# # %%
# def determine_shelf_dur(nv_sig):
#     '''
#     This function will test yellow shelf pulse lengths between 0 and 200 ns on the LOW
#     NV state.
#     '''
#     apd_indices = [0]
#     num_reps = 10**3
#     test_pulse_dur_list = numpy.linspace(0,500,6).tolist()
# #    test_pulse_dur_list = numpy.linspace(0,200,3)
    
#     # measure laser powers:
#     # green_optical_power_pd, green_optical_power_mW, \
#     #         red_optical_power_pd, red_optical_power_mW, \
#     #         yellow_optical_power_pd, yellow_optical_power_mW = \
#     #         tool_belt.measure_g_r_y_power( 
#     #                               nv_sig['am_589_power'], nv_sig['nd_filter'])
    
# #    cxn.pulse_streamer.constant([], 0.0, nv_sig['am_589_shelf_power'])
    
#     # create some lists for data
#     sig_count_raw = []
#     ref_count_raw = []
#     sig_counts_avg = []
#     ref_counts_avg = []
#     snr_list = []
    
#     # measure the power of the test pulse
#     # optical_power = tool_belt.opt_power_via_photodiode(589, 
#     #                                 AO_power_settings = nv_sig['am_589_shelf_power'], 
#     #                                 nd_filter = nv_sig['nd_filter'])
#     # shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)

#     # Step through the pulse lengths for the test laser
#     for test_pulse_length in test_pulse_dur_list:
#         nv_sig['spin_shelf_dur'] = test_pulse_length
#         sig_count, ref_count = measure(nv_sig, apd_indices, num_reps,
#                                     States.LOW, plot = False)
#         sig_count = [int(el) for el in sig_count]
#         ref_count = [int(el) for el in ref_count]
        
#         sig_count_raw.append(sig_count)
#         ref_count_raw.append(ref_count)
        
#         avg_snr = tool_belt.calc_snr(sig_count, ref_count)
        
#         sig_counts_avg.append(numpy.average(sig_count))
#         ref_counts_avg.append(numpy.average(ref_count))
#         snr_list.append(avg_snr)
 
#     #plot
#     title = 'Sweep pusle length for 589 nm shelf'
#     # text = 'Shelf pulse power = ' + '%.0f'%(shelf_power * 10**3) + ' uW' 
#     fig = plot_snr_v_dur(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None)
    
#     # Save
#     timestamp = tool_belt.get_time_stamp()
#     raw_data = {'timestamp': timestamp,
#             'nv_sig': nv_sig,
#             'ion_durs': ion_durs.tolist(),
#             'ion_durs-units': 'ns',
#             'num_reps':num_reps,
#             'sig_count_raw': sig_count_raw,
#             'sig_count_raw-units': 'counts',
#             'ref_count_raw': ref_count_raw,
#             'ref_count_raw-units': 'counts',            
#             'sig_counts_avg': sig_counts_avg,
#             'sig_counts_avg-units': 'counts',
#             'sig_counts_avg': sig_counts_avg,
#             'sig_counts_avg-units': 'counts',
#             'snr_list': snr_list,
#             'snr_list-units': 'arb'
#             }

#     file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
#     tool_belt.save_raw_data(raw_data, file_path + '-shelf_pulse_dur')

#     tool_belt.save_figure(fig, file_path + '-shelf_pulse_dur')
    
#     print(' \nRoutine complete!')
#     return


# %%
# def test_rabi(nv_sig):
#     from random import shuffle
#     apd_indices = [0]
#     num_reps = 10**3
# #    test_pulse_dur_list = numpy.linspace(0,200,11).tolist()
#     test_pulse_dur_list = numpy.linspace(0,400,21).tolist()
# #    test_pulse_dur_ind_list = [list(range(0, len(test_pulse_dur_list)))]
#     # measure laser powers:
# #    green_optical_power_pd, green_optical_power_mW, \
# #            red_optical_power_pd, red_optical_power_mW, \
# #            yellow_optical_power_pd, yellow_optical_power_mW = \
# #            tool_belt.measure_g_r_y_power( 
# #                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
#     green_optical_power_pd = None
#     green_optical_power_mW = None
#     yellow_optical_power_pd = None
#     yellow_optical_power_mW = None
#     red_optical_power_pd = None
#     red_optical_power_mW = None
#     shelf_power = None
    
#     # create some lists for data
#     sig_count_raw = []
#     ref_count_raw = []
#     sig_counts_avg = []
#     ref_counts_avg = []
#     snr_list = []
    
#     # measure the power of the test pulse
# #    optical_power = tool_belt.opt_power_via_photodiode(589, 
# #                                    AO_power_settings = nv_sig['am_589_shelf_power'], 
# #                                    nd_filter = nv_sig['nd_filter'])
# #    shelf_power = tool_belt.calc_optical_power_mW(589, optical_power)
#     shuffle(test_pulse_dur_list)
#     # Step through the pulse lengths for the test laser
#     for test_pulse_length in test_pulse_dur_list:
#         nv_sig['rabi_LOW'] = test_pulse_length
#         sig_count, ref_count = measure(nv_sig, apd_indices, num_reps,
#                                     States.LOW, plot = False)
#         sig_count = [int(el) for el in sig_count]
#         ref_count = [int(el) for el in ref_count]
        
#         sig_count_raw.append(sig_count)
#         ref_count_raw.append(ref_count)
        
#         avg_snr = tool_belt.calc_snr(sig_count, ref_count)
        
#         sig_counts_avg.append(numpy.average(sig_count))
#         ref_counts_avg.append(numpy.average(ref_count))
#         snr_list.append(avg_snr)
 
#     #plot
    
#     title = 'Test pi pulse length'
    
#     fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
#     ax = axes[0]
#     ax.plot(numpy.array(test_pulse_dur_list), sig_counts_avg, 'ro', 
#            label = 'W/ pi-pulse')
#     ax.plot(numpy.array(test_pulse_dur_list) , ref_counts_avg, 'ko', 
#            label = 'W/out pi-pulse')
#     ax.set_xlabel('Pi Pulse dur (ns)')
#     ax.set_ylabel('Counts (single shot measurement)')
#     ax.set_title(title)
#     ax.legend()
    
#     ax = axes[1]    
#     ax.plot(numpy.array(test_pulse_dur_list), numpy.array(sig_counts_avg) / numpy.array(ref_counts_avg), 'r')
#     ax.set_xlabel('Pi Pulse dur (ns)')
#     ax.set_ylabel('normalized counts')
#     ax.set_title(title)
    

#     fig = plot_snr_v_dur(test_pulse_dur_list, sig_counts_avg, ref_counts_avg, snr_list, title, text = None)
    
    
#     # Save
#     timestamp = tool_belt.get_time_stamp()
#     raw_data = {'timestamp': timestamp,
#             'nv_sig': nv_sig,
#             'ion_durs': ion_durs.tolist(),
#             'ion_durs-units': 'ns',
#             'num_reps':num_reps,
#             'sig_count_raw': sig_count_raw,
#             'sig_count_raw-units': 'counts',
#             'ref_count_raw': ref_count_raw,
#             'ref_count_raw-units': 'counts',            
#             'sig_counts_avg': sig_counts_avg,
#             'sig_counts_avg-units': 'counts',
#             'sig_counts_avg': sig_counts_avg,
#             'sig_counts_avg-units': 'counts',
#             'snr_list': snr_list,
#             'snr_list-units': 'arb'
#             }

#     file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
#     tool_belt.save_raw_data(raw_data, file_path + '-rabi_test')

#     tool_belt.save_figure(fig, file_path + '-rabi_test')
    
#     print(' \nRoutine complete!')
#     return


# %%
def test_esr(nv_sig, apd_indices, num_reps, state = States.LOW):
    '''
    This function will test red pulse lengths between 0 and 600 ns on the LOW
    NV state.
    '''
    from random import shuffle
    
    freq_center = nv_sig['resonance_LOW']
    freq_range = 0.12
    
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    freqs =  [freq_center]#numpy.linspace(freq_low, freq_high, 11).tolist()

    num_steps = len(freqs)
    
    # create some arrays for data
    sig_counts_array = numpy.zeros(num_steps)
    sig_counts_ste_array = numpy.copy(sig_counts_array)
    ref_counts_array = numpy.copy(sig_counts_array)
    ref_counts_ste_array = numpy.copy(sig_counts_array)
    snr_array = numpy.copy(sig_counts_array)
    

    freq_ind_master_list = []
    
    freq_ind_list = list(range(0, num_steps))
    shuffle(freq_ind_list)
    
    # Step through the pulse lengths for the test laser
    for f in freq_ind_list:
        print('Freq : {} GHz'.format(freqs[f]))
        nv_sig_copy = copy.deepcopy(nv_sig)
        nv_sig_copy['resonance_LOW'] = freqs[f]
        freq_ind_master_list.append(f)
        sig_counts, ref_counts = measure(nv_sig_copy, apd_indices, num_reps,
                                    state, plot = False)
        # print(sig_counts)
        
        sig_count_avg = numpy.average(sig_counts)
        sig_counts_ste = stats.sem(sig_counts)
        ref_count_avg = numpy.average(ref_counts)
        ref_counts_ste = stats.sem(ref_counts)
            
        sig_counts_array[f] = sig_count_avg
        sig_counts_ste_array[f] = sig_counts_ste
        ref_counts_array[f] = ref_count_avg
        ref_counts_ste_array[f] = ref_counts_ste
        
        avg_snr = tool_belt.calc_snr(sig_counts, ref_counts)
        snr_array[f] = avg_snr
 
    #plot
    title = 'SCC ESR'
    # fig = plot_snr_v_dur(freqs, sig_counts_array, ref_counts_array, 
    #                      sig_counts_ste_array, ref_counts_ste_array,
    #                       snr_array, title)
    
    freqs = numpy.array(freqs)
    
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.errorbar(freqs, sig_counts_array, yerr=sig_counts_ste, fmt= 'ro', 
           label = 'W/ pi-pulse')
    ax.errorbar(freqs, ref_counts_array, yerr=ref_counts_ste, fmt= 'ko', 
           label = 'W/out pi-pulse')
    ax.set_xlabel('Microwave freq (GHz)')
    ax.set_ylabel('Counts (single shot measurement)')
    ax.set_title(title)
    ax.legend()
    
    ax = axes[1]    
    ax.plot(freqs, snr_array, 'ro')
    ax.set_xlabel('Microwave freq (GHz)')
    ax.set_ylabel('SNR')
    ax.set_title(title)
    # if text:
    #     ax.text(0.55, 0.90, text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)
    # Save
    
    freqs = numpy.array(freqs)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'freqs': freqs.tolist(),
            'freqs-units': 'GHz',
            'num_reps':num_reps,            
            'sig_counts_array': sig_counts_array.tolist(),
            'sig_counts_ste_array': sig_counts_ste_array.tolist(),
            'ref_counts_array': ref_counts_array.tolist(),
            'ref_counts_ste_array': ref_counts_ste_array.tolist(),
            'snr_list': snr_array.tolist(),
            'freq_ind_master_list': freq_ind_master_list
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'] + '-scc_esr')
    
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)
    
    print(' \nRoutine complete!')
    return
    
#     from random import shuffle
#     apd_indices = [0]
#     num_reps =10**3
#     freq_center = nv_sig['resonance_LOW']
#     freq_range = 0.12
    
#     half_freq_range = freq_range / 2
#     freq_low = freq_center - half_freq_range
#     freq_high = freq_center + half_freq_range
#     freqs = [freq_center]# numpy.linspace(freq_low, freq_high, 25).tolist()

#     green_optical_power_pd = None
#     green_optical_power_mW = None
#     yellow_optical_power_pd = None
#     yellow_optical_power_mW = None
#     red_optical_power_pd = None
#     red_optical_power_mW = None
#     shelf_power = None
    
#     # create some lists for data
#     sig_count_raw = []
#     ref_count_raw = []
#     sig_counts_avg = []
#     ref_counts_avg = []
#     snr_list = []
    
#     shuffle(freqs)
#     # Step through the pulse lengths for the test laser
#     for f in freqs:
#         nv_sig['resonance_LOW'] = f
#         sig_count, ref_count = measure(nv_sig, apd_indices, num_reps,
#                                     States.LOW, plot = False)
#         sig_count = [int(el) for el in sig_count]
#         ref_count = [int(el) for el in ref_count]
        
#         sig_count_raw.append(sig_count)
#         ref_count_raw.append(ref_count)
        
#         avg_snr = tool_belt.calc_snr(sig_count, ref_count)
        
#         sig_counts_avg.append(numpy.average(sig_count))
#         ref_counts_avg.append(numpy.average(ref_count))
#         snr_list.append(avg_snr)
 
#     #plot
#     title = 'Test pi pulse frequency'    
    
#     fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
#     ax = axes[0]
#     ax.plot(freqs , sig_counts_avg, 'ro', 
#            label = 'W/ pi-pulse')
#     ax.plot(freqs , ref_counts_avg, 'ko', 
#            label = 'W/out pi-pulse')
#     ax.set_xlabel('Pi Pulse freq (GHz)')
#     ax.set_ylabel('Counts (single shot measurement)')
#     ax.set_title(title)
#     ax.legend()
    
#     ax = axes[1]    
#     ax.plot(freqs, numpy.array(sig_counts_avg) / numpy.array(ref_counts_avg), 'ro')
#     ax.set_xlabel('Pi Pulse freq (GHz)')
#     ax.set_ylabel('normalized counts')
#     ax.set_title(title)
    
#     # Save
#     timestamp = tool_belt.get_time_stamp()
#     raw_data = {'timestamp': timestamp,
#             'nv_sig': nv_sig,
#             'ion_durs': ion_durs.tolist(),
#             'ion_durs-units': 'ns',
#             'num_reps':num_reps,
#             'sig_count_raw': sig_count_raw,
#             'sig_count_raw-units': 'counts',
#             'ref_count_raw': ref_count_raw,
#             'ref_count_raw-units': 'counts',            
#             'sig_counts_avg': sig_counts_avg,
#             'sig_counts_avg-units': 'counts',
#             'sig_counts_avg': sig_counts_avg,
#             'sig_counts_avg-units': 'counts',
#             'snr_list': snr_list,
#             'snr_list-units': 'arb'
#             }

#     file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
#     tool_belt.save_raw_data(raw_data, file_path + '-esr_test')

#     tool_belt.save_figure(fig, file_path + '-esr_test')
    
#     print(' \nRoutine complete!')
#     return


# %%
    
if __name__ == '__main__':
    apd_indices = [1]
    # apd_indices = [0,1]

    nd_yellow = "nd_0"
    green_power =8000
    nd_green = 'nd_0.4'
    red_power = 120
    sample_name = "rubin"
    green_laser = "integrated_520"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"



    nv_sig = {
            "coords":[-0.853, -0.593, 6.16],
        "name": "{}-nv1_2022_08_10".format(sample_name,),
        "disable_opt":False,
        "ramp_voltages": False,
        "expected_count_rate":10,
        "correction_collar": 0.12,



          "spin_laser":green_laser,
          "spin_laser_power": green_power,
         "spin_laser_filter": nd_green,
          "spin_readout_dur": 350,
          "spin_pol_dur": 1000.0,

          "imaging_laser":green_laser,
        "imaging_laser_power": green_power,
         "imaging_laser_filter": nd_green,
          "imaging_readout_dur": 1e7,

         # "initialize_laser": green_laser,
         #   "initialize_laser_power": green_power,
         #   "initialize_laser_dur":  1e3,
         # "CPG_laser": green_laser,
         #   "CPG_laser_power":red_power,
         #   "CPG_laser_dur": int(1e6),



        # "nv-_prep_laser": green_laser,
        # "nv-_prep_laser-power": None,
        # "nv-_prep_laser_dur": 1e3,
        # "nv0_prep_laser": red_laser,
        # "nv0_prep_laser-power": None,
        # "nv0_prep_laser_dur": 1e3,
        
        "nv-_reionization_laser": green_laser,
        "nv-_reionization_laser_power": green_power,
        "nv-_reionization_dur": 1e3,
        
        "nv0_ionization_laser": red_laser,
        "nv0_ionization_laser_power": None,
        "nv0_ionization_dur": 30000,
        
        "spin_shelf_laser": yellow_laser,
        "spin_shelf_laser_power": 0.0,
        "spin_shelf_dur": 0,
        
         "charge_readout_laser": yellow_laser,
          "charge_readout_laser_power": 0.2, 
          "charge_readout_laser_filter": "nd_1.0",
          "charge_readout_dur": 100e6, 

        # "collection_filter": "715_lp",#see only SiV (some NV signal)
        # "collection_filter": "740_bp",#SiV emission only (no NV signal)
        "collection_filter": "715_sp+630_lp", # NV band only
        "magnet_angle": 156,
        "resonance_LOW":2.6061,
        "rabi_LOW":0,#96,
        "uwave_power_LOW": 15,  # 15.5 max
        "resonance_HIGH":3.1345,
        "rabi_HIGH":88.9,
        "uwave_power_HIGH": 10,
    }  # 14.5 max
    
    num_reps = 600#500
    # Run the program
    # determine_ionization_dur(nv_sig, apd_indices, num_reps, [600, 650, 700, 750, 800, 850, 900, 950, 1000])
    # determine_reion_dur(nv_sig)
    # determine_shelf_dur(nv_sig)
    test_esr(nv_sig, apd_indices, num_reps)
        


    
    