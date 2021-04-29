# -*- coding: utf-8 -*-
"""
Created on mon Apr 8 10:45:09 2020

This file ru ns a sequence that pulses a green pulse either on of off the 
readout spot. It reads out in the SiVs band, to create SiV2- when on the spot
and SiV when off . Then a green pulse of variable power is pulsed on the
readout spot, followed by a yellow readout.

The point of this measurement is to determine how fast the SiV charge states
change under illumination. 

USE WITH 515 AM MOD

@author: agardill
"""
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
#import time
import matplotlib.pyplot as plt
import labrad
import copy

# %%
def build_voltage_list(start_coords_drift, signal_coords_drift, num_reps):

    # calculate the x values we want to step thru
    start_x_value = start_coords_drift[0]
    start_y_value = start_coords_drift[1]

    
    # we want this list to have the pattern [[readout], [readout], [readout], [target], 
    #                                                   [readout], [readout], [readout],...]
    # The glavo needs a 0th coord, so we'll pass the readout NV as the "starting" point
    x_points = [start_x_value]
    y_points = [start_y_value]
    
    # now append the coordinates in the following pattern:
    for i in range(num_reps):
        x_points.append(start_x_value)
        x_points.append(start_x_value) 
        x_points.append(signal_coords_drift[0])
        x_points.append(start_x_value)
        x_points.append(start_x_value)
        x_points.append(start_x_value)
        
        y_points.append(start_y_value)
        y_points.append(start_y_value) 
        y_points.append(signal_coords_drift[1])
        y_points.append(start_y_value)
        y_points.append(start_y_value)
        y_points.append(start_y_value) 
        
    return x_points, y_points

#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, num_reps, test_color, test_time, test_power):

    with labrad.connect() as cxn:
        on_counts, off_counts = main_with_cxn(cxn, nv_sig, apd_indices,
                                 num_reps, test_color, test_time, test_power)
        
    return on_counts, off_counts
def main_with_cxn(cxn, nv_sig, apd_indices, num_reps, test_color, test_time, test_power):

    tool_belt.reset_cfm_wout_uwaves(cxn)

# Initial Calculation and setup
    readout_time = nv_sig['pulsed_SCC_readout_dur']
    am_589_power = nv_sig['am_589_power']
    am_515_power = nv_sig['ao_515_pwr']
    nd_filter = nv_sig['nd_filter']
    siv_pulse_time = 10**6 # ns
    siv_pulse_distance = 0.056 # V
        
    prep_power_515 = am_515_power
    readout_power_589 = am_589_power
        
    
    # set the nd_filter for yellow
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)

    #delay of aoms and laser
    laser_515_delay = shared_params['515_AM_laser_delay'] ###
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']
    
    # if using AM for green, add an additional 300 ns to the pulse time. 
    # the AM laser has a 300 ns rise time
    if test_color == '515a':
        test_time = test_time + 300
    

    # Optimize
    opti_coords_list = []  
    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, '515a', disable=False)
    opti_coords_list.append(opti_coords)
    cxn.filter_slider_ell9k_color.set_filter('715 lp') 
    
     # Estimate the lenth of the sequance , load the sequence          
    file_name = 'isolate_nv_charge_dynamics_moving_target.py'
    seq_args = [siv_pulse_time, test_time, readout_time, 
            laser_515_delay, aom_589_delay, laser_638_delay, galvo_delay,
            readout_power_589, 
            prep_power_515, test_power, prep_power_515,             
            apd_indices[0], '515a', test_color, 589]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = cxn.pulse_streamer.stream_load(file_name, seq_args_string)
    seq_dur = ret_vals[0]
    period = seq_dur
    
    

    # Set up the voltages to step thru  
    # get the drift and add it to the start coordinates      
    drift = numpy.array(tool_belt.get_drift())
    start_coords = numpy.array(nv_sig['coords'])
    start_coords_drift = start_coords + drift
    # define the signal coords as start + dx.
    signal_coords_drift = start_coords_drift + [siv_pulse_distance, 0, 0]
    
    x_voltages, y_voltages = build_voltage_list(start_coords_drift, signal_coords_drift, num_reps)
    
    # Collect data
    # start on the readout NV
    tool_belt.set_xyz(cxn, start_coords_drift)
    
    # Load the galvo
    cxn.galvo.load_arb_points_scan(x_voltages, y_voltages, int(period))
    
    # Set up the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    
    #Run the sequence double the amount of time, one for the sig and one for the ref
    cxn.pulse_streamer.stream_start(num_reps*2)
    
    # We'll be lookign for three samples each repetition, and double that for 
    # the ref and sig
    total_num_reps = 3*2*num_reps
    
    # Read the counts
    new_samples = cxn.apd_tagger.read_counter_simple(total_num_reps)
    # The last of the triplet of readout windows is the counts we are interested in
    on_counts = new_samples[2::6]
    on_counts = [int(el) for el in on_counts]
    off_counts = new_samples[5::6]
    off_counts = [int(el) for el in off_counts]
    
    cxn.apd_tagger.stop_tag_stream()

    
    return on_counts, off_counts

# %%

def sweep_test_pulse_length(nv_sig, test_color, test_power, test_pulse_dur_list = None):
    apd_indices = [0]
    num_reps = 200
    if not test_pulse_dur_list:
        test_pulse_dur_list = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 
                                   500, 600,  700,  800,  900, 1000, 1500,
                                   2000]
#        test_pulse_dur_list = [0]
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    # create some lists for data
    on_count_raw = []
    off_count_raw = []
    
    # Step through the pulse lengths for the test laser
    for test_time in test_pulse_dur_list:
        print('Testing {} us'.format(test_time/10**3))
        on_count, off_count = main(nv_sig, apd_indices, num_reps, test_color, test_time, test_power)
        
#        on_count = [int(el) for el in on_count]
#        off_count = [int(el) for el in off_count]
        
        on_count_raw.append(on_count)
        off_count_raw.append(off_count)
        
    on_counts = numpy.average(on_count_raw, axis = 1)
    off_counts = numpy.average(off_count_raw, axis = 1)
    fig, ax = plt.subplots() 
    ax.plot(test_pulse_dur_list, on_counts, 'bo')
    ax.plot(test_pulse_dur_list, off_counts, 'go')
    ax.set_xlabel('Test Pulse Illumination Time (ns)')
    ax.set_ylabel('Counts')
    
    # Save
    test_pulse_dur_list = numpy.array(test_pulse_dur_list)
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'test_color': test_color,
            'test_power': test_power,
            'test_power-units': 'V',
            'test_pulse_dur_list': test_pulse_dur_list.tolist(),
            'test_pulse_dur_list-units': 'ns',
            'num_reps':num_reps,
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
            'on_count_raw':on_count_raw,
            'on_count_raw-units': 'counts',
            'off_count_raw': off_count_raw,
            'off_count_raw-units': 'counts', 
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
            'expected_count_rate': None, 'nd_filter': 'nd_0',
#            'color_filter': '635-715 bp', 
            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 4*10**7, 'am_589_power': 0.3, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 130, 
            'ao_515_pwr':0.65,
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':10, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}

    test_pulses = [50, 100, 150, 200, 250, 300, 310, 320, 330, 340, 350,
                   360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 500]
    for i in [5]:#range(len(nv_coords_list)):
        nv_sig = copy.deepcopy(nv_2021_03_30)
        nv_sig['coords'] = nv_coords_list[i]
        nv_sig['expected_count_rate'] = expected_count_list[i]
        nv_sig['name'] = 'goeppert-mayer-nv{}_2021_04_15'.format(i)
#        for p in (0.658, 0.64, 0.622, 0.611, 0.606):
        sweep_test_pulse_length(nv_sig, '515a' ,0.64)
    
