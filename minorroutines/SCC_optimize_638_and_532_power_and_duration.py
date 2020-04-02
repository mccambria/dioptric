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
def main(nv_sig, apd_indices, num_reps, test_pulse_length, 
         test_pulse_color_ind):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices,  
                           num_reps, test_pulse_length, 
                           test_pulse_color_ind)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices,
                  num_reps, test_pulse_length, test_pulse_color_ind):

    tool_belt.reset_cfm(cxn)

# Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    aom_ao_589_pwr = nv_sig['am_589_power']
    nd_filter = nv_sig['nd_filter']
    cobalt_532_pwr = nv_sig['cobalt_532_power']
    cobalt_638_pwr = nv_sig['cobalt_638_power']
    
    if test_pulse_color_ind == 532:
        initial_pulse = nv_sig['pulsed_ionization_dur']
        print(' \nTesting GREEN, set to {} mW, at {} us long'.format(cobalt_532_pwr, test_pulse_length / 10**3))
    elif test_pulse_color_ind == 638:
        initial_pulse = nv_sig['pulsed_reionization_dur']
        print(' \nTesting RED, set to {} mW, at {} us long'.format(cobalt_638_pwr, test_pulse_length / 10**3))
        
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
    file_name = os.path.basename(__file__)
    seq_args = [readout_time, initial_pulse, test_pulse_length,
            wait_time, laser_515_delay, aom_589_delay, laser_638_delay, 
            apd_indices[0], aom_ao_589_pwr, test_pulse_color_ind]
#    print(seq_args)
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

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)

    seq_args = [readout_time, initial_pulse, test_pulse_length,
        wait_time, laser_515_delay, aom_589_delay, laser_638_delay, 
        apd_indices[0], aom_ao_589_pwr, test_pulse_color_ind]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_immediate(file_name, num_reps, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
    sample_counts = new_counts[0]
#    print(len(sample_counts))
    # signal counts are even - get every second element starting from 0
    sig_gate_counts = sample_counts[0::2]
    sig_count = int(sum(sig_gate_counts))

    # ref counts are odd - sample_counts every second element starting from 1
    ref_gate_counts = sample_counts[1::2]
    ref_count = int(sum(ref_gate_counts))
    
    cxn.apd_tagger.stop_tag_stream()
    
    return sig_count, ref_count

# %%

def optimize_pulse_length(nv_sig, test_pulse_dur_list, test_pulse_color_ind):
    apd_indices = [0]
    num_reps = 500
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    
    # make sure the list is a list, for saving purposes
    if type(test_pulse_dur_list) != list:
        test_pulse_dur_list = test_pulse_dur_list.tolist()
    
    # create some lists for data
    sig_count_list = []
    ref_count_list = []
    
    # Step through the pulse lengths for the test laser
    for test_pulse_length in test_pulse_dur_list:
        sig_count, ref_count = main(nv_sig, apd_indices, num_reps,
                                    test_pulse_length, test_pulse_color_ind)
        
        sig_count_list.append(sig_count)
        ref_count_list.append(ref_count)
     
    # Calculate snr
    snr = abs((numpy.array(sig_count_list) - numpy.array(ref_count_list)) / \
              numpy.sqrt(ref_count_list)) # should it be ref? This mimics opt_589 file
    
    # Plot
    if test_pulse_color_ind == 532:
        fmt = 'go'
    elif test_pulse_color_ind == 638:
        fmt = 'ro'
    fig, axes = plt.subplots(1,2, figsize = (17, 8.5)) 
    ax = axes[0]
    ax.plot(numpy.array(test_pulse_dur_list) / 10**3, sig_count_list, fmt, 
            label = 'W/ {} nm pulse'.format(test_pulse_color_ind))
    ax.plot(numpy.array(test_pulse_dur_list) / 10**3, ref_count_list, 'ko', 
            label = 'W/out {} nm pulse'.format(test_pulse_color_ind))
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('Counts')
    ax.set_title('Sweep pusle length for {} nm'.format(test_pulse_color_ind))
    ax.legend()
    
    ax = axes[1]    
    ax.plot(numpy.array(test_pulse_dur_list) / 10**3, snr, fmt)
    ax.set_xlabel('Test pulse length (us)')
    ax.set_ylabel('SNR')
    ax.set_title('Sweep pusle length for {} nm'.format(test_pulse_color_ind))

    # Save
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
            'num_reps':num_reps,
            'sig_count_list': sig_count_list,
            'sig_count_list-units': 'counts',
            'ref_count_list': ref_count_list,
            'ref_count_list-units': 'counts',
            'snr': snr.tolist(),
            'snr-units': 'arb'
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-pulse_dur')

    tool_belt.save_figure(fig, file_path + '-pulse_dur')
#    tool_belt.save_figure(fig_snr, file_path + '-pulse_dur_snr') 
    
    print(' \nRoutine complete!')
    return

# %%
    
if __name__ == '__main__':
    sample_name = 'hopper'
    ensemble = { 'coords': [0.0, 0.0, 5.00],
            'name': '{}-ensemble'.format(sample_name),
            'expected_count_rate': 1000, 'nd_filter': 'nd_0.5',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 10**7, 'am_589_power': 0.25, 
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 160, 
            'pulsed_reionization_dur': 10**7, 'cobalt_532_power': 8, 
            'magnet_angle': 0,
            'resonance_LOW': 2.8059, 'rabi_LOW': 173.5, 'uwave_power_LOW': 9.0, 
            'resonance_HIGH': 2.9366, 'rabi_HIGH': 247.4, 'uwave_power_HIGH': 10.0}  
    nv_sig = ensemble

    # TEST PULSE LISTS
    
#    test_pulse_length = 10**4 # ns
    test_pulse_dur_list = numpy.linspace(100, 600, 6) # Red
#    test_pulse_dur_list = [5*10**2, 10**3, 10**4, 10**5, 1*10**6, 3*10**6, 5*10**6, 
#                           10**7, 2*10**7, 3*10**7, 4*10**7, 5*10**7]
    
    # Run the program
    test_pulse_color_ind = 638
    optimize_pulse_length(nv_sig, test_pulse_dur_list, test_pulse_color_ind)