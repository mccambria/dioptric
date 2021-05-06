# -*- coding: utf-8 -*-
"""
Created on mon Apr 8 10:45:09 2020

This file ru ns a sequence that places the NV in NV- (NV0), then applies a test
pulse of some length, then reads out the NV with yellow.

The point of this measurement is to determine how fast the NV charge states
change under illumination. 

USE WITH 515 DM MOD

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
import labrad
import copy


#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, num_reps, test_color, test_time):

    with labrad.connect() as cxn:
        green_counts, red_counts = main_with_cxn(cxn, nv_sig, apd_indices,
                                 num_reps, test_color, test_time)
        
    return green_counts, red_counts
def main_with_cxn(cxn, nv_sig, apd_indices, num_reps, test_color, test_time):

    tool_belt.reset_cfm_wout_uwaves(cxn)

# Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    red_prep_time = nv_sig['pulsed_ionization_dur']
    green_prep_time = nv_sig['pulsed_reionization_dur']
    am_589_power = nv_sig['am_589_power']
    am_515_power = nv_sig['ao_515_pwr']
    nd_filter = nv_sig['nd_filter']
    
    apd_index = 0
            
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_DM_laser_delay'] 
    
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
        
    
    green_counts = []
    red_counts = []

    cxn.filter_slider_ell9k.set_filter('nd_0')
    optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)
    run_start_time = time.time()
    
    for i in range(num_reps):
        # Move the ND filter to 0
        cxn.filter_slider_ell9k.set_filter('nd_0')
        time.sleep(0.2)
        #optimize every 2 min or so
        # So first check the time. If the time that has passed since the last
        # optimize is longer that 2 min, optimize again
        current_time = time.time()
        if current_time - run_start_time >= 2*60:
            optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532)
            run_start_time = current_time
                
        # GREEN INITIAL PULSE          
        pulse_file_name = 'simple_pulse.py'            
        seq_args = [laser_515_delay, green_prep_time, am_589_power, am_515_power, 532 ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)
    
        # Set the Nd filter
        cxn.filter_slider_ell9k.set_filter(nd_filter)
        time.sleep(0.1)    
    
        # test pulse                    
        seq_args = [laser_515_delay, test_time, am_589_power, am_515_power, test_color ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)
        
        # readout 
        readout_file_name = 'simple_readout.py'  
        seq_args = [aom_589_delay, readout_time, am_589_power, am_515_power, apd_index, 589 ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_load(readout_file_name,  seq_args_string)
        
        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        # Clear the buffer
        cxn.apd_tagger.clear_buffer()
        # Run the sequence
        cxn.pulse_streamer.stream_immediate(readout_file_name, 1, seq_args_string)
    
        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        green_counts.append(sample_counts[0])
        
        
        # RED INITIAL PULSE   
        pulse_file_name = 'simple_pulse.py'            
        seq_args = [laser_638_delay, red_prep_time, am_589_power, am_515_power, 638 ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)
        
        # test pulse                    
        seq_args = [laser_515_delay, test_time, am_589_power, am_515_power, test_color ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_immediate(pulse_file_name, 1, seq_args_string)
        
        # readout 
        readout_file_name = 'simple_readout.py'  
        seq_args = [aom_589_delay, readout_time, am_589_power, am_515_power, apd_index, 589 ]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        cxn.pulse_streamer.stream_load(readout_file_name,  seq_args_string)
        
        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        # Clear the buffer
        cxn.apd_tagger.clear_buffer()
        # Run the sequence
        cxn.pulse_streamer.stream_immediate(readout_file_name, 1, seq_args_string)
    
        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
        red_counts.append(sample_counts[0])

    
    return green_counts, red_counts

# %%

def sweep_test_pulse_length(nv_sig, test_color, test_power, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 100
    if not test_pulse_dur_list:
        test_pulse_dur_list = [0, 25, 50, 75, 100,  150 , 200, 250, 300, 400, 
                                   500, 750, 1000, 1500,
                                   2000]
#        test_pulse_dur_list = [0]
    # measure laser powers:
#    green_optical_power_pd, green_optical_power_mW, \
#            red_optical_power_pd, red_optical_power_mW, \
#            yellow_optical_power_pd, yellow_optical_power_mW = \
#            tool_belt.measure_g_r_y_power( 
#                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    # create some lists for data
    green_count_raw = []
    red_count_raw = []
    
    # Step through the pulse lengths for the test laser
    for test_time in test_pulse_dur_list:
        print('Testing {} us'.format(test_time/10**3))
        green_count, red_count = main(nv_sig, apd_indices, num_reps, test_color, test_time)
        
        green_count = [int(el) for el in green_count]
        red_count = [int(el) for el in red_count]
        
        green_count_raw.append(green_count)
        red_count_raw.append(red_count)
        
    green_counts = numpy.average(green_count_raw, axis = 1)
    red_counts = numpy.average(red_count_raw, axis = 1)
    fig, ax = plt.subplots() 
    ax.plot(test_pulse_dur_list, green_counts, 'go')
    ax.plot(test_pulse_dur_list, red_counts, 'ro')
    ax.set_xlabel('Test Pulse Illumination Time (ns)')
    ax.set_ylabel('Counts')
    
    # Save
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'test_color': test_color,
            'test_power': test_power,
            'test_power-units': 'mW',
            'test_pulse_dur_list': test_pulse_dur_list.tolist(),
            'test_pulse_dur_list-units': 'ns',
            'num_reps':num_reps,
            'nv_sig': nv_sig,
            'nv_sig-units': tool_belt.get_nv_sig_units(),
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
            'green_count_raw':green_count_raw,
            'green_count_raw-units': 'counts',
            'red_count_raw': red_count_raw,
            'red_count_raw-units': 'counts', 
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
    tool_belt.save_figure(fig, file_path)
    
    print(' \nRoutine complete!')
    return

# %% Run the files
    
if __name__ == '__main__':
#    sample_name = 'goepert-mayer'
    
        
    expected_count_list = [40, 45, 65, 64, 55, 35,  40, 45 ] #
    nv_coords_list = [
[-0.037, 0.119, 5.14],
[-0.090, 0.066, 5.04],
[-0.110, 0.042, 5.13],
[0.051, -0.115, 5.08],
[-0.110, 0.042, 5.06],

[0.063, 0.269, 5.09], 
[0.243, 0.184, 5.12],
[0.086, 0.220, 5.03],
]
    
    nv_2021_03_30 = { 'coords':[], 
            'name': '',
            'expected_count_rate': None, 'nd_filter': 'nd_0.5',
            'color_filter': '635-715 bp', 
#            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 10*10**7, 'am_589_power': 0.15, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 130, 
            'ao_515_pwr':0.65,
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':10, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}

    test_pulses = [0, 50, 100, 500]
    for i in [5]:#range(len(nv_coords_list)):
        nv_sig = copy.deepcopy(nv_2021_03_30)
        nv_sig['coords'] = nv_coords_list[i]
        nv_sig['expected_count_rate'] = expected_count_list[i]
        nv_sig['name'] = 'goeppert-mayer-nv{}_2021_04_15'.format(i)
#        for p in [0.606, 0.608, 0.611, 0.622, 0.63, 0.64, 0.658 ]:
        sweep_test_pulse_length(nv_sig, 532, 0.59, test_pulse_dur_list = test_pulses)
    
