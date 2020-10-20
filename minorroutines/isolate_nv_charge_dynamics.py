# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:12:42 2020

@author: gardill
"""

import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
import matplotlib.pyplot as plt
import labrad
import majorroutines.image_sample as image_sample
# %%
# Connect to labrad in this file, as opposed to control panel
def main(target_sig, readout_sig, target_color, readout_color, apd_indices):

    with labrad.connect() as cxn:
        counts = main_with_cxn(cxn, target_sig, readout_sig, target_color, readout_color, apd_indices)
        
    return counts
def main_with_cxn(cxn, target_sig, readout_sig, target_color, readout_color, apd_indices):
    apd_index= apd_indices[0]
    tool_belt.reset_cfm(cxn)
    readout_file_name = 'simple_readout.py'
    target_file_name = 'simple_pulse.py'
    #delay of aoms and laser
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    laser_515_delay = shared_params['515_laser_delay']
    aom_589_delay = shared_params['589_aom_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    
    # Target 
    
    #If we are pulsing an initial laser:
    if target_color:
        # move the galvo to the target
        target_coords = target_sig['coords'] ### Drift??
        tool_belt.set_xyz(cxn,target_coords)
        time.sleep(0.1)
        # Pusle the laser
        if target_color == 532:
            target_pulse_time = target_sig['pulsed_reionization_dur']
            laser_delay = laser_515_delay
        elif target_color == 638:
            target_pulse_time = target_sig['pulsed_ionization_dur']
            laser_delay = laser_638_delay
        seq_args = [laser_delay, int(target_pulse_time), 0.0, target_color]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_load(target_file_name, seq_args_string) 
        

    
    # then readout with yellow
    aom_ao_589_pwr = readout_sig['am_589_power']
    nd_filter = readout_sig['nd_filter']  
    readout_pulse_time = readout_sig['pulsed_SCC_readout_dur']
    laser_delay = aom_589_delay
    
    cxn.filter_slider_ell9k.set_filter(nd_filter)  
   # move the galvo to the readout
    readout_coords = readout_sig['coords'] ### Drift??
    tool_belt.set_xyz(cxn, readout_coords)
    time.sleep(0.1)
    
    seq_args = [laser_delay, int(readout_pulse_time), aom_ao_589_pwr, apd_index, readout_color]           
    seq_args_string = tool_belt.encode_seq_args(seq_args)            
    cxn.pulse_streamer.stream_load(readout_file_name, seq_args_string)         
    # collect the counts
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(readout_file_name, 1, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
    sample_counts = new_counts[0]
#    print(sample_counts)

    # signal counts are even - get every second element starting from 0
    sig_counts = sample_counts[0] 
    
    
    cxn.apd_tagger.stop_tag_stream()
    
    return sig_counts
#%%
def charge_spot(target_nv_sig, readout_nv_sig, dark_spot_sig):
    apd_indices = [0]
    num_runs = 3
       
    # create some lists for dataopti_coords_list
    opti_coords_list = []
    control = []
    green_readout = []
    red_readout = []
    green_target = []
    red_target = []
    green_dark = []
    red_dark = []
    
    for run in range(num_runs):
        #optimize
        with labrad.connect() as cxn:
            opti_coords = optimize.main_with_cxn(cxn, readout_nv_sig, apd_indices, 532, disable=False)
            opti_coords_list.append(opti_coords)
        
        
        # Step through the experiments
        #Sweep red laser
        image_sample.main(target_nv_sig,  0.1, 0.1, 60, apd_indices, 638, save_data=False, plot_data=False, readout =10**3)
    
        # readout NV_readout
        with labrad.connect() as cxn:
            cxn.pulse_streamer.constant([7], 0.0, 0.0)
            time.sleep(2) 
        sig_count =  main(target_nv_sig, readout_nv_sig, None, 589, apd_indices)
        control.append(sig_count)
        readout_sec = readout_nv_sig['pulsed_SCC_readout_dur'] / 10**9
        print(str(sig_count  / readout_sec / 10**3) + 'kcps' )
        
    print('control measurement:')
    print(str(numpy.average(control)/ readout_sec / 10**3) + 'kcps')
# 
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                              readout_nv_sig['am_589_power'], readout_nv_sig['nd_filter'])
#    # Plot
#    title = 'Sweep pulse power for 589 nm'
#    text = 'Yellow pulse length set to ' + str(nv_sig['pulsed_SCC_readout_dur']/10**6) + ' ms'
#    fig = plot_power_sweep(optical_power_list, sig_counts_avg, ref_counts_avg, 
#                          snr_list, title, text = text)
#    # Save
#    timestamp, raw_data = compile_raw_data_power_sweep(nv_sig, 
#                     green_optical_power_pd, green_optical_power_mW, 
#                     red_optical_power_pd, red_optical_power_mW, 
#                     yellow_optical_power_pd, yellow_optical_power_mW, 
#                     power_list, optical_power_list, num_reps, 
#                     sig_count_raw, ref_count_raw, sig_counts_avg, ref_counts_avg, snr_list)
#
#    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
#    tool_belt.save_raw_data(raw_data, file_path + '-readout_pulse_pwr')
#
#    tool_belt.save_figure(fig, file_path + '-readout_pulse_pwr')
    
    print(' \nRoutine complete!')
    return

# %% Run the files
    
if __name__ == '__main__':
    sample_name = 'goeppert-mayer'
    NVA = { 'coords':[0.422, -0.080,  5.1],
            'name': '{}-NVA'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 2*10**6, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':18, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}   
    NVB = { 'coords':[0.441, -0.103,  5.1],
            'name': '{}-NVB'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 2*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 120, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':18, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}     
    dark = { 'coords':[0.434, -0.051,  5.1],
            'name': '{}-dark_region'.format(sample_name),
            'expected_count_rate': None, 'nd_filter': 'nd_0',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 2*10**7, 'am_589_power': 0.25, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 500, 'cobalt_638_power': 120, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':18, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0} 
    charge_spot(NVB, NVA, dark)