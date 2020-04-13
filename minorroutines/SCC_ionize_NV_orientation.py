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
import matplotlib.pyplot as plt
import labrad
from utils.tool_belt import States
#from scipy.optimize import curve_fit

#%%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

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
def main(nv_sig, apd_indices, num_ionizations, num_reps, state):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices,  
                           num_ionizations, num_reps, state)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices,
                  num_ionizations, num_reps, state):

    tool_belt.reset_cfm(cxn)

    # Initial Calculation and setup
    yellow_pol_time = 10**3
    yellow_pol_pwr = 0.3
    shelf_time = nv_sig['pulsed_shelf_dur']
    shelf_pwr = nv_sig['am_589_shelf_power']
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    init_ion_time = nv_sig['pulsed_initial_ion_dur']
#    init_ion_time = 0
    ionization_time = nv_sig['pulsed_ionization_dur']
    reionization_time = nv_sig['pulsed_reionization_dur']
#    reionization_time = 10**7
    
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
#    seq_args = [readout_time, yellow_pol_time, shelf_time, init_ion_time, reionization_time, ionization_time, uwave_pi_pulse,
#        wait_time, num_ionizations, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
#        apd_indices[0], aom_ao_589_pwr, yellow_pol_pwr, shelf_pwr,  state.value]
    seq_args = [readout_time, yellow_pol_time, init_ion_time, reionization_time, ionization_time, uwave_pi_pulse,
        wait_time, num_ionizations, laser_515_delay, aom_589_delay, laser_638_delay, rf_delay,
        apd_indices[0], aom_ao_589_pwr, yellow_pol_pwr,  state.value]
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
    num_ion_list = numpy.linspace(0,160, 81)
    
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
    snr_list = []
    
    # Step through the pulse lengths for the test laser
    for num_ionizations in num_ion_list:
        sig_counts, ref_counts = main(nv_sig, apd_indices, num_ionizations, num_reps,
                                    States.LOW)
        sig_counts = [int(el) for el in sig_counts]
        ref_counts = [int(el) for el in ref_counts]
        
        sig_count_raw.append(sig_counts)
        ref_count_raw.append(ref_counts)
        
        avg_snr = tool_belt.calc_snr(sig_counts, ref_counts)
        
        sig_counts_avg.append(numpy.average(sig_counts))
        ref_counts_avg.append(numpy.average(ref_counts))
        snr_list.append(avg_snr)
 
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
    ax.plot(num_ion_list, snr_list, 'ro')
    ax.set_xlabel('Number of repetitions')
    ax.set_ylabel('SNR')
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
            'snr_list': snr_list,
            'snr_list-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-ion_rep')

    tool_belt.save_figure(fig, file_path + '-ion_rep')
    
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
            'pulsed_SCC_readout_dur': 1*10**7, 'am_589_power': 0.3, 
            'pulsed_initial_ion_dur': 200*10**3,
            'pulsed_shelf_dur': 50, 'am_589_shelf_power': 0.3,
            'pulsed_ionization_dur': 200, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 200*10**3, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 187.8, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}   
    nv_sig = ensemble
    
    # Run the program
    optimize_num_ionizations(nv_sig)
    
    
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
    