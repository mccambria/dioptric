# -*- coding: utf-8 -*-
"""
Created on mon Mar 30 10:45:09 2020

This file will change the pulse duration or power of an intermediate laser 
pulse, the final pusle beign a readout with 589 nm. This can be done to 
estimate an optimal red or green pulse power and length.

For example, we can shine green light to polarize to NV-, then shine red light
while sweeping the length, and readout the counts with a low yellow light. In
this case, we'd want to observe a minimum amount of counts to optimize the red's
ability to ionize th eNv into NV0.

test_pulse_color_ind can be 532 or 638. This is the second pulse in the seq

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import matplotlib.pyplot as plt
import labrad

#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices,
          num_runs, num_reps, test_pulse_color_ind, plot = True):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices,  
                           num_runs, num_reps, 
                           test_pulse_color_ind, plot)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices,
                  num_runs, num_reps, 
                  test_pulse_color_ind, plot):

    tool_belt.reset_cfm(cxn)

# %% Initial Calculation and setup
    readout_time = nv_sig['pulsed_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    cobalt_532_pwr = nv_sig['cobalt_532_pwr']
    cobalt_638_pwr = nv_sig['cobalt_638_pwr']
    
    if test_pulse_color_ind == 532:
        test_pulse_length = nv_sig['pulsed_reionization_dur']
    elif test_pulse_color_ind == 638:
        test_pulse_length = nv_sig['pulsed_ionization_dur']
        
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    
    wait_time = shared_params['post_polarization_wait_dur']
    
    if test_pulse_color_ind == 532:
        initial_pulse = nv_sig['pulsed_ionization_dur']
        print(' \nTesting GREEN, set to {} mW, at {} us long'.format(cobalt_532_pwr, test_pulse_length / 10**3))
    elif test_pulse_color_ind == 638:
        initial_pulse = nv_sig['pulsed_reionization_dur']
        print(' \nTesting RED, set to {} mW, at {} us long'.format(cobalt_638_pwr, test_pulse_length / 10**3))
        
    test_pulse = test_pulse_length

    # Set up our data lists
#    counts = []
#    ref_counts = []
#    sig_counts=[]
    opti_coords_list = []

    # %% Read the optical power for red, yellow, and green light
#    green_optical_power_pd = tool_belt.opt_power_via_photodiode(532)
#
#    red_optical_power_pd = tool_belt.opt_power_via_photodiode(638,
#                                            AO_power_settings = ao_638_pwr)
#
#    yellow_optical_power_pd = tool_belt.opt_power_via_photodiode(589,
#           AO_power_settings = aom_ao_589_pwr, nd_filter = nd_filter)
#
#    # Convert V to mW optical power
#    green_optical_power_mW = \
#            tool_belt.calc_optical_power_mW(532, green_optical_power_pd)
#
#    red_optical_power_mW = \
#            tool_belt.calc_optical_power_mW(638, red_optical_power_pd)
#
#    yellow_optical_power_mW = \
#            tool_belt.calc_optical_power_mW(589, yellow_optical_power_pd)

#    readout_power = yellow_optical_power_mW

#%% Estimate the lenth of the sequance            
    file_name = os.path.basename(__file__)
    seq_args = [readout_time, initial_pulse, test_pulse,
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, 
            apd_indices[0], aom_ao_589_pwr, test_pulse_color_ind]
#    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_reps * num_runs * seq_time_s + (0.5 * num_runs)  #s
    expected_run_time_m = expected_run_time / 60 # m

    # Ask to continue and timeout if no response in 2 seconds?

    print(' \nExpected run time: {:.1f} minutes. '.format(expected_run_time_m))

#    return

#%% Collect data


    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=True)
    opti_coords_list.append(opti_coords)


    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)

    seq_args = [readout_time, initial_pulse, test_pulse,
        wait_time, laser_515_delay, aom_589_delay, laser_638_delay, 
        apd_indices[0], aom_ao_589_pwr, test_pulse_color_ind]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)
    
    # option 1:
#        new_counts = cxn.apd_tagger.read_counter_simple(num_reps)
#
#        counts.extend(new_counts)
#
#    cxn.apd_tagger.stop_tag_stream()
#    sig_counts = counts[0:len(counts):2]
#    ref_counts = counts[1:len(counts):2]
    
    # option 2

    new_counts = cxn.apd_tagger.read_counter_separate_gates(num_reps)
#    print(new_counts)
    
    # signal counts are even - get every second element starting from 0
    sig_gate_counts = new_counts[0::2]
    sig_count = int(numpy.average(sig_gate_counts))

    # ref counts are odd - sample_counts every second element starting from 1
    ref_gate_counts = new_counts[1::2]
    ref_count = int(numpy.average(ref_gate_counts))
    
    cxn.apd_tagger.stop_tag_stream()
    
#    print(sig_count)
#    print(ref_count)


#%% plot the data

#    if plot:
#        fig2, ax2 = plt.subplots(1,1, figsize = (10, 8.5))
#    
#        photon_counts_sig = ps.get_photon_counts(readout_time*10**-9, sig_counts)
#        sig_len=len(photon_counts_sig)
#    
#        photon_counts_ref = ps.get_photon_counts(readout_time*10**-9, ref_counts)
#        ref_len=len(photon_counts_ref)
#    
#        ax2.plot(numpy.linspace(0,sig_len-1, sig_len), numpy.array(photon_counts_sig)/10**3, 'r', label='Test pulse')
#        ax2.plot(numpy.linspace(0,ref_len-1, ref_len), numpy.array(photon_counts_ref)/10**3, 'k', label='Test pulse absent')
#        ax2.set_xlabel('Rep number')
#        ax2.set_ylabel('photon counts (kcps)')
#        if test_pulse_color_ind == 532:
#            ax2.set_title('Pulse sequence Green - Red - Yellow')
#        if test_pulse_color_ind == 638:
#            ax2.set_title('Pulse sequence Red - Green - Yellow')
#        ax2.legend()
#    
#        text = '\n'.join(('Readout time (589 nm)'+'%.3f'%(readout_time/10**3) + 'us',
#                         'Readout power (589 nm)'+'%.3f'%(readout_power*1000) + 'uW'))
#    
#        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#        ax2.text(0.55, 0.85, text, transform=ax2.transAxes, fontsize=12,
#                verticalalignment='top', bbox=props)


#%% Save data
#    timestamp = tool_belt.get_time_stamp()
#
#    # turn the list of unique_values into pure integers, for saving
#    sig_counts = [int(el) for el in sig_counts]
#    ref_counts = [int(el) for el in ref_counts]
#
#    raw_data = {'timestamp': timestamp,
#            'nv_sig': nv_sig,
#            'nv_sig-units': tool_belt.get_nv_sig_units(),
#            'test_pulse_color_ind': test_pulse_color_ind,
#            'ao_638_pwr': ao_638_pwr,
#            'ao_638_pwr-units': 'V',
#            'cobalt_532_pwr': cobalt_532_pwr,
#            'cobalt_532_pwr-units': 'mW',
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
#            'readout_time':readout_time,
#            'readout_time_unit':'ns',
#            'test_pulse_length': test_pulse_length,
#            'test_pulse_length-units': 'ns',
#            'num_runs': num_runs,
#            'num_reps':num_reps,
#            'sig_counts': sig_counts,
#            'sig_counts-units': 'counts',
#            'ref_counts': ref_counts,
#            'ref_counts-units': 'counts'
#            }
#
#    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
#    tool_belt.save_raw_data(raw_data, file_path)

#    if plot:
#        tool_belt.save_figure(fig2, file_path + '-counts')
    
    return sig_count, ref_count

# %%

def optimize_pulse_length(nv_sig, test_pulse_dur_list, test_pulse_color_ind):
    apd_indices = [0]
    num_runs = 1
    num_reps = 10**4
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
            
    if type(test_pulse_dur_list) != list:
        test_pulse_dur_list = test_pulse_dur_list.tolist()
        
    sig_count_list = []
    ref_count_list = []
#    norm_count_list = []
    
    for test_pulse_length in test_pulse_dur_list:
        nv_sig['pulsed_reionization_dur'] = int(test_pulse_length)
        sig_count, ref_count = main(nv_sig, apd_indices,
                                   num_runs, num_reps, test_pulse_color_ind)
        
#        norm_counts = numpy.array(sig_counts) / numpy.array(ref_counts)
#    
#        avg_sig_count = numpy.average(sig_counts)
#        avg_ref_count = numpy.average(ref_counts)
#        avg_norm_counts = numpy.average(norm_counts)
        
        sig_count_list.append(sig_count)
        ref_count_list.append(ref_count)
#        avg_norm_count_list.append(avg_norm_counts)
        
    norm_count_array = numpy.array(sig_count_list) / numpy.array(ref_count_list)
                                    
    fig, ax = plt.subplots(1,1, figsize = (10, 8.5))    
    ax.plot(numpy.array(test_pulse_dur_list) / 10**3, sig_count_list, 'go', label = 'Test pulse')
    ax.plot(numpy.array(test_pulse_dur_list) / 10**3, ref_count_list, 'ko', label = 'W/out test pulse')
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('Avg counts')
    ax.set_title('Sweep pusle length for {} nm'.format(test_pulse_color_ind))
    ax.legend()
    
    fig_norm, ax = plt.subplots(1,1, figsize = (10, 8.5))    
    ax.plot(numpy.array(test_pulse_dur_list) / 10**3, norm_count_array, 'bo')
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('Normalized counts')
    ax.set_title('Sweep pusle length for {} nm'.format(test_pulse_color_ind))

    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'test_pulse_color_ind': test_pulse_color_ind,
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
            'test_pulse_dur_list': test_pulse_dur_list,
            'test_pulse_dur_list-units': 'ns',
            'num_runs': num_runs,
            'num_reps':num_reps,
            'sig_count_list': sig_count_list,
            'sig_count_list-units': 'counts',
            'ref_count_list': ref_count_list,
            'ref_count_list-units': 'counts',
            'norm_count_array': norm_count_array.tolist(),
            'norm_count_array-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-pulse_dur')

    tool_belt.save_figure(fig, file_path + '-pulse_dur_sig_ref')
    tool_belt.save_figure(fig_norm, file_path + '-pulse_dur_norm') 
    
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
            'pulsed_SCC_readout_dur': 10**7, 'am_589_power': 0.3, 
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10**6, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 173.5, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}
    
    
    
    nv_sig = ensemble
    num_runs = 1
    
#    test_pulse_length = 10**4 # ns
    test_pulse_dur_list = numpy.linspace(100, 600, 6)
#    test_pulse_dur_list = [1000, 2000, 3000, 4000, 5000]
    
    num_reps = 10**4
    test_pulse_color_ind = 638
#    main(nv_sig, apd_indices, ao_638_pwr, cobalt_532_pwr, test_pulse_length, 
#          num_runs, num_reps, test_pulse_color_ind)
    
    optimize_pulse_length(nv_sig, test_pulse_dur_list, test_pulse_color_ind)