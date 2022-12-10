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
import numpy
import os
import copy
import matplotlib.pyplot as plt
import labrad
from utils.tool_belt import States
import time
from random import shuffle
import scipy.stats as stats
import majorroutines.optimize as optimize

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
def measure(nv_sig, num_reps, state, plot = True):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = measure_with_cxn(cxn, nv_sig, num_reps, state, plot)
        
    return sig_counts, ref_counts
def measure_with_cxn(cxn, nv_sig, num_reps, state, plot):

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    tool_belt.reset_cfm(cxn)
    
    tagger_server = tool_belt.get_server_tagger(cxn)
    pulsegen_server = tool_belt.get_server_pulse_gen(cxn)


    # Initial Calculation and setup
    tool_belt.set_filter(cxn, nv_sig, "charge_readout_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv-_reionization_laser")
    tool_belt.set_filter(cxn, nv_sig, "nv0_ionization_laser")
    # tool_belt.set_filter(cxn, nv_sig, "spin_shelf_laser")
        
    readout_time = nv_sig['charge_readout_dur']
    ionization_time = nv_sig['nv0_ionization_dur']
    print(ionization_time)
    reionization_time = nv_sig['nv-_reionization_dur']
    shelf_time = 0 #nv_sig['spin_shelf_dur']
    
    readout_power = tool_belt.set_laser_power(
        cxn, nv_sig, "charge_readout_laser"
    )
    reion_power = tool_belt.set_laser_power(
        cxn, nv_sig, "nv-_reionization_laser"
    )
    ion_power = tool_belt.set_laser_power(
        cxn, nv_sig, "nv0_ionization_laser"
    )
    #  = tool_belt.set_laser_power(
    #     cxn, nv_sig, "spin_shelf_laser"
    # )
    shelf_power = 0

    uwave_freq = nv_sig['resonance_{}'.format(state.name)]
    uwave_power = nv_sig['uwave_power_{}'.format(state.name)]
    rabi_period = float(nv_sig['rabi_{}'.format(state.name)])
    uwave_pi_pulse = tool_belt.get_pi_pulse_dur(rabi_period)
    
    
    green_laser_name = nv_sig['nv-_reionization_laser']
    red_laser_name = nv_sig['nv0_ionization_laser']
    yellow_laser_name = nv_sig['charge_readout_laser']
    sig_gen_cxn = tool_belt.get_server_sig_gen(cxn, state)   
    sig_gen_name = sig_gen_cxn.name
    
    num_reps = int(num_reps)
    opti_coords_list = []

    # Estimate the lenth of the sequance            
    file_name = 'rabi_scc.py'        
    seq_args = [readout_time, reionization_time, ionization_time, uwave_pi_pulse,
        shelf_time ,  uwave_pi_pulse, 
        green_laser_name, yellow_laser_name, red_laser_name, sig_gen_name,
        reion_power, ion_power, shelf_power, readout_power]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(file_name, seq_args_string)
      
    print(seq_args)
    # return
    
    seq_time = int(ret_vals[0])

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * seq_time_s  #s
    # expected_run_time_m = expected_run_time / 60 # m

    # print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))


    # Collect data

    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig)
    opti_coords_list.append(opti_coords)
    
    # Turn on the microwaves
    sig_gen_cxn.set_freq(uwave_freq)
    sig_gen_cxn.set_amp(uwave_power)
    sig_gen_cxn.uwave_on()
    
    # Load the APD
    tagger_server.start_tag_stream()

    pulsegen_server.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = tagger_server.read_counter_separate_gates(1)
    sample_counts = new_counts[0]
    # print(sample_counts)

    # signal counts are even - get every second element starting from 0
    sig_counts = sample_counts[0::2]
    # print(sig_counts)

    # ref counts are odd - sample_counts every second element starting from 1
    ref_counts = sample_counts[1::2]
    # print(ref_counts)
    
    tagger_server.stop_tag_stream()
    tool_belt.reset_cfm(cxn)

    return sig_counts, ref_counts

# %%

def determine_ionization_dur(nv_sig, num_reps, ion_durs=None):
    '''
    This function will test red pulse lengths between 0 and 600 ns on the LOW
    NV state.
    '''
    state = States.LOW
    if ion_durs is None:
        # ion_durs = numpy.array([340])#
        ion_durs = numpy.linspace(20,212,7)
        # ion_durs = numpy.linspace(20,532,17)
  
    num_steps = len(ion_durs)
    
    # create some arrays for data
    sig_counts_array = numpy.zeros(num_steps)
    sig_counts_eachshot_array = numpy.zeros([num_steps,num_reps])
    ref_counts_eachshot_array = numpy.zeros([num_steps,num_reps])
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
        sig_counts, ref_counts = measure(nv_sig_copy, num_reps,
                                    state, plot = False)
        # print(sig_counts)
        
        sig_count_avg = numpy.average(sig_counts)
        sig_counts_ste = stats.sem(sig_counts)
        ref_count_avg = numpy.average(ref_counts)
        ref_counts_ste = stats.sem(ref_counts)
        
        sig_counts_eachshot_array[ind] = sig_counts
        ref_counts_eachshot_array[ind] = ref_counts
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
            'sig_counts_eachshot_array': sig_counts_eachshot_array.tolist(),
            'ref_counts_eachshot_array': ref_counts_eachshot_array.tolist(),
            'sig_counts_ste_array': sig_counts_ste_array.tolist(),
            'ref_counts_array': ref_counts_array.tolist(),
            'ref_counts_ste_array': ref_counts_ste_array.tolist(),
            'snr_list': snr_array.tolist(),
            'dur_ind_master_list': dur_ind_master_list
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'] + '-ion_pulse_dur')
    
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)
    
    # nv_sig = raw_data['nv_sig']
    
    # readout_dur = int(nv_sig['charge_readout_dur'])
    # readout_power = int(nv_sig['charge_readout_laser_power']*1000)
    # init_time = int(nv_sig['nv-_reionization_dur'])

    # timestamp = tool_belt.get_time_stamp()
    # # time.sleep(2)
    
    
    # save_path1 = tool_belt.get_file_path(__file__, timestamp,'-hists'+'_{}'.format(readout_dur)+
    #                                      'ns_{}'.format(readout_power)+'mV')
    
    # ion_durs = numpy.array(raw_data['ion_durs'])
    # sig_counts = numpy.array(raw_data['sig_counts_eachshot_array'])[0]
    # ref_counts = numpy.array(raw_data['ref_counts_eachshot_array'])[0]
    
    
    # raw_fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    # # axes.cla()
    # nbins=[50,100,250,500,1000]
    # nb=0
    # for ax in axes:
    # # plt.hist(binned_data,bins=int(max(binned_data)),histtype='step',linewidth=2)
    #     width=nbins[nb]
    #     binned_data_sig = sig_counts[:(sig_counts.size // width) * width].reshape(-1, width).mean(axis=1)
    #     binned_data_ref = ref_counts[:(ref_counts.size // width) * width].reshape(-1, width).mean(axis=1)
    #     ax.hist(binned_data_sig,bins=6,histtype='step',linewidth=2,label='$m_s$=-1')
    #     ax.hist(binned_data_ref,bins=6,histtype='step',linewidth=2,label='$m_s$=0')
    #     ax.set_xlabel('avg counts in bins of {}'.format(width))
    #     nb+=1
    # axes[0].legend(loc='upper right')
    # axes[2].set_title('{}'.format(readout_dur/1000)+'us     {}'.format(readout_power/1000)+'V')
    # tool_belt.save_figure(raw_fig, save_path1)
    # plt.show()
    
    # print(' \nRoutine complete!')
    return

# %%
# def determine_reion_dur(nv_sig,apd_indices,num_reps, reion_durs):
#     '''
#     This function will test green pulse length for charge state reinitialization
#     NV state.
#     '''
#     state = States.LOW
  
#     num_steps = len(reion_durs)
    
#     # create some arrays for data
#     sig_counts_array = numpy.zeros(num_steps)
#     sig_counts_ste_array = numpy.copy(sig_counts_array)
#     ref_counts_array = numpy.copy(sig_counts_array)
#     ref_counts_ste_array = numpy.copy(sig_counts_array)
#     snr_array = numpy.copy(sig_counts_array)
    

#     dur_ind_master_list = []
    
#     dur_ind_list = list(range(0, num_steps))
#     shuffle(dur_ind_list)
    
#     # Step through the pulse lengths for the test laser
#     for ind in dur_ind_list:
#         t = reion_durs[ind]
#         dur_ind_master_list.append(ind)
#         print('Ionization dur: {} ns'.format(t))
#         nv_sig_copy = copy.deepcopy(nv_sig)
#         nv_sig_copy['nv0_ionization_dur'] = t
#         sig_counts, ref_counts = measure(nv_sig_copy, apd_indices, num_reps,
#                                     state, plot = False)
#         # print(sig_counts)
        
#         sig_count_avg = numpy.average(sig_counts)
#         sig_counts_ste = stats.sem(sig_counts)
#         ref_count_avg = numpy.average(ref_counts)
#         ref_counts_ste = stats.sem(ref_counts)
            
#         sig_counts_array[ind] = sig_count_avg
#         sig_counts_ste_array[ind] = sig_counts_ste
#         ref_counts_array[ind] = ref_count_avg
#         ref_counts_ste_array[ind] = ref_counts_ste
        
#         avg_snr = tool_belt.calc_snr(sig_counts, ref_counts)
#         snr_array[ind] = avg_snr
 
#     #plot
#     title = 'Sweep ionization pulse duration'
#     fig = plot_snr_v_dur(reion_durs, sig_counts_array, ref_counts_array, 
#                          sig_counts_ste_array, ref_counts_ste_array,
#                           snr_array, title)
#     # Save
    
#     reion_durs = numpy.array(reion_durs)
#     timestamp = tool_belt.get_time_stamp()
#     raw_data = {'timestamp': timestamp,
#             'nv_sig': nv_sig,
#             'reion_durs': reion_durs.tolist(),
#             'reion_durs-units': 'ns',
#             'num_reps':num_reps,            
#             'sig_counts_array': sig_counts_array.tolist(),
#             'sig_counts_ste_array': sig_counts_ste_array.tolist(),
#             'ref_counts_array': ref_counts_array.tolist(),
#             'ref_counts_ste_array': ref_counts_ste_array.tolist(),
#             'snr_list': snr_array.tolist(),
#             'dur_ind_master_list': dur_ind_master_list
#             }

#     file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'] + '-ion_pulse_dur')
    
#     tool_belt.save_raw_data(raw_data, file_path)
#     tool_belt.save_figure(fig, file_path)
    
#     print(' \nRoutine complete!')
#     return


# %%
    
if __name__ == '__main__':
    
    # file = "2022_11_15-11_37_37-johnson-search-ion_pulse_dur"
    
    assign_state = False
    photon_thresh = 4
    # for file in os.listdir("E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_Carr/branch_opx-setup/determine_scc_pulse_params/2022_11/readout_time+power_sweep"):
    for file in ['2022_11_15-09_45_21-johnson-search-ion_pulse_dur',
                 '2022_12_10-08_56_16-johnson-search-ion_pulse_dur']:
        print(file)
        
        data = tool_belt.get_raw_data(file)
        nv_sig = data['nv_sig']
        
        readout_dur = int(nv_sig['charge_readout_dur'])
        readout_power = int(nv_sig['charge_readout_laser_power']*1000)
        init_time = int(nv_sig['nv-_reionization_dur'])

        timestamp = tool_belt.get_time_stamp()
        # time.sleep(2)
        
        
        save_path1 = tool_belt.get_file_path(__file__, timestamp,'-hists'+'_{}'.format(readout_dur)+
                                             'ns_{}'.format(readout_power)+'mV')
        
        ion_durs = numpy.array(data['ion_durs'])
        sig_counts = numpy.array(data['sig_counts_eachshot_array'])[0]
        ref_counts = numpy.array(data['ref_counts_eachshot_array'])[0]
        
        if assign_state:
            
            sig_counts[numpy.where(sig_counts<=photon_thresh)] = 0
            sig_counts[numpy.where(sig_counts>=photon_thresh)] = 1
            
            ref_counts[numpy.where(ref_counts<=photon_thresh)] = 0
            ref_counts[numpy.where(ref_counts>=photon_thresh)] = 1
        
        raw_fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        # axes.cla()
        averaging_times = numpy.array([100e3,500e3,1000e3]) #us
        nbins = numpy.array(averaging_times/(readout_dur/1e3),dtype=numpy.int32)
        nb=0
        for ax in axes:
        # plt.hist(binned_data,bins=int(max(binned_data)),histtype='step',linewidth=2)
            width=nbins[nb]
            binned_data_sig = sig_counts[:(sig_counts.size // width) * width].reshape(-1, width).sum(axis=1)
            binned_data_ref = ref_counts[:(ref_counts.size // width) * width].reshape(-1, width).sum(axis=1)
            ax.hist(binned_data_sig,bins=10,histtype='step',linewidth=2,label='$m_s$=-1')
            ax.hist(binned_data_ref,bins=10,histtype='step',linewidth=2,label='$m_s$=0')
            ax.set_xlabel('Binned Counts (summed). Averaging Time = {} ms'.format(averaging_times[nb]/1000))
            nb+=1
        axes[0].legend(loc='upper right')
        axes[2].set_title('{}'.format(readout_dur/1000)+'us    {}'.format(readout_power/1000)+'V')
        tool_belt.save_figure(raw_fig, save_path1)
        plt.show()
    
    







