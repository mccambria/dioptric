# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:34:26 2020

Repeatedly measure the coutns of an NV after red or green light to determine
the NV0 or NV- average counts.

Can either do a single NV or a list of NVs

@author: agardill
"""
# %%
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import time
#import matplotlib.pyplot as plt
import labrad
#import majorroutines.image_sample as image_sample
import copy
import scipy.stats as stats
import minorroutines.SCC_optimize_pulses_wout_uwaves as SCC_optimize_pulses_wout_uwaves
# %%
# Apply a gren or red pulse, then measure the counts under yellow illumination. 
# Repeat num_reps number of times and returns the list of counts after red illumination, then green illumination
def main(nv_sig, apd_indices, num_reps):

    with labrad.connect() as cxn:
        sig_counts, ref_counts = main_with_cxn(cxn, nv_sig, apd_indices, num_reps)
        
    return sig_counts, ref_counts
def main_with_cxn(cxn, nv_sig, apd_indices, num_reps):

    tool_belt.reset_cfm(cxn)

# Initial Calculation and setup
    
    aom_ao_589_pwr = nv_sig['am_589_power']
    
    nd_filter = nv_sig['nd_filter']
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    reionization_time = nv_sig['pulsed_reionization_dur']
    ionization_time = nv_sig['pulsed_ionization_dur']
        

    
    
    #delay of aoms and laser
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    aom_589_delay = shared_params['589_aom_delay']
    laser_515_delay = shared_params['515_laser_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']
    

    # Set up our data lists
    opti_coords_list = []
    
    # Optimize
    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_indices, 532, disable=False)
    opti_coords_list.append(opti_coords)

    # Estimate the lenth of the sequance            
    seq_file = 'simple_readout_two_pulse.py'

        
    #### Load the measuremnt with green laser
    seq_args = [galvo_delay, laser_515_delay, aom_589_delay, reionization_time,
                readout_pulse_time, aom_ao_589_pwr, apd_indices[0], 532, 589]
#    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(seq_file, seq_args_string)

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(seq_file, num_reps, seq_args_string)

    nvm = cxn.apd_tagger.read_counter_simple(num_reps)
    
    # Load the measuremnt with red laser first
    seq_args = [galvo_delay, laser_638_delay, aom_589_delay, ionization_time,
                readout_pulse_time, aom_ao_589_pwr, apd_indices[0], 638, 589]
#    print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(seq_file, seq_args_string)

    # Load the APD
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(seq_file, num_reps, seq_args_string)

    nv0 = cxn.apd_tagger.read_counter_simple(num_reps)

    
    return nv0, nvm
# %%
def collect_charge_counts(nv_sig, num_reps, save_data = True):
    with labrad.connect() as cxn:
         ret_vals=collect_charge_counts_with_cxn(cxn, nv_sig, num_reps, save_data)
         nv0_avg, nv0_ste, nvm_avg, nvm_ste = ret_vals
         
    return nv0_avg, nv0_ste, nvm_avg, nvm_ste
def collect_charge_counts_with_cxn(cxn, nv_sig, num_reps, save_data = True):
    num_reps = 1000
    num_runs = 10
    
    apd_indices = [0]
    seq_file = 'simple_readout_two_pulse.py'
    
    #delay of aoms and laser, parameters, etc
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    aom_589_delay = shared_params['589_aom_delay']
    laser_515_delay = shared_params['515_laser_delay']
    laser_638_delay = shared_params['638_DM_laser_delay']
    galvo_delay = shared_params['large_angle_galvo_delay']

    aom_ao_589_pwr = nv_sig['am_589_power']
    
    nd_filter = nv_sig['nd_filter']
    readout_pulse_time = nv_sig['pulsed_SCC_readout_dur']
    cxn.filter_slider_ell9k.set_filter(nd_filter)
    
    reionization_time = nv_sig['pulsed_reionization_dur']
    ionization_time = nv_sig['pulsed_ionization_dur']
    
    # some lists to measure the 
    nv0 = []
    
    nvm = []
    
#    opti_coords_list=[]
    
    # Move the galvo
    drift = tool_belt.get_drift()
    coords = nv_sig['coords']
    coords_drift = numpy.array(coords) + numpy.array(drift)
    # move the galvo to the NV
    tool_belt.set_xyz(cxn, coords_drift)


    # Measure the NV0 counts
    seq_args = [galvo_delay, laser_638_delay, aom_589_delay, ionization_time,
                readout_pulse_time, aom_ao_589_pwr, apd_indices[0], 638, 589]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    
    # Optimize
#    opti_coords = optimize.main_with_cxn(cxn, nv_sig, apd_index, 532, disable=True)
#    opti_coords_list.append(opti_coords)
    for i in range(num_runs):
        print(i)
        
        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        # Clear the buffer
        cxn.apd_tagger.clear_buffer()
        # Run the sequence
        cxn.pulse_streamer.stream_immediate(seq_file, num_reps, seq_args_string)
    
        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        print(new_counts)
        sample_counts = new_counts[0]
    
        # signal counts are even - get every second element starting from 0
        counts = sample_counts[0::1]
        cxn.apd_tagger.stop_tag_stream()
            
        nv0.append(counts)
        
        # Measure the NV- counts
        seq_args = [galvo_delay, laser_515_delay, aom_589_delay, reionization_time,
                    readout_pulse_time, aom_ao_589_pwr, apd_indices[0], 532, 589]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
    
        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
        # Clear the buffer
        cxn.apd_tagger.clear_buffer()
        # Run the sequence
        cxn.pulse_streamer.stream_immediate(seq_file, num_reps, seq_args_string)
    
        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)
        sample_counts = new_counts[0]
    
        # signal counts are even - get every second element starting from 0
        counts = sample_counts[0::1]
        
        nvm.append(counts)
        
        cxn.apd_tagger.stop_tag_stream()
        
    print(nv0)
    print(nvm)
    nv0_avg = numpy.average(nv0)
    nv0_ste = stats.sem(nv0) 
    nvm_avg = numpy.average(nvm)
    nvm_ste = stats.sem(nvm) 
    
    
    if save_data:
        # measure laser powers:
        green_optical_power_pd, green_optical_power_mW, \
                red_optical_power_pd, red_optical_power_mW, \
                yellow_optical_power_pd, yellow_optical_power_mW = \
                tool_belt.measure_g_r_y_power(
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    

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
                'num_runs':num_reps,
#                'opti_coords_list': opti_coords_list,
                'nv0': nv0,
                'nv0-units': 'counts',
                'nvm': nvm,
                'nvm-units': 'counts',
                'nv0_avg': nv0_avg,
                'nv0_avg-units': 'counts',
                'nv0_ste': nv0_ste,
                'nv0_ste-units': 'counts',
                'nvm_avg': nvm_avg,
                'nvm_avg-units': 'counts',
                'nvm_ste': nvm_ste,
                'nvm_ste-units': 'counts'
                }
        
    
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-single_nv')
    
    print(str(nv0_avg) + ' +/-' + str(nv0_ste))
    print(str(nvm_avg) + ' +/-' + str(nvm_ste))
    
    return nv0_avg, nv0_ste, nvm_avg, nvm_ste

# %%
def collect_charge_counts_yellow_pwr(coords, parameters_sig, nd_filter, aom_power_list, num_reps, apd_indices ):
    with labrad.connect() as cxn:
        tool_belt.reset_cfm(cxn)
    
    nv0_list = []
    nvm_list = []
    nv0_avg_list = []
    nv0_ste_list = []
    nvm_avg_list = []
    nvm_ste_list = []
    yellow_power_list = []
    print(nd_filter)
    
    for power in aom_power_list:
        print(power)
        nv_sig = copy.deepcopy(parameters_sig)
        nv_sig['coords'] = coords
        nv_sig['am_589_power'] = power
        nv_sig['nd_filter'] = nd_filter
#        time.sleep(0.002)
        
        ret_vals = main(nv_sig,  apd_indices,num_reps)
        nv0_counts, nvm_counts = ret_vals
        nv0_counts = [int(el) for el in nv0_counts]
        nvm_counts = [int(el) for el in nvm_counts]
        
        nv0_list.append(nv0_counts)
        nvm_list.append(nvm_counts)
        
        nv0_avg = numpy.average(nv0_counts)
        nv0_ste = stats.sem(nv0_counts)
        nv0_avg_list.append(nv0_avg)
        nv0_ste_list.append(nv0_ste)
        
        nvm_avg = numpy.average(nvm_counts)
        nvm_ste = stats.sem(nvm_counts)
        nvm_avg_list.append(nvm_avg)
        nvm_ste_list.append(nvm_ste)    
    
        # measure laser powers:
        green_optical_power_pd, green_optical_power_mW, \
                red_optical_power_pd, red_optical_power_mW, \
                yellow_optical_power_pd, yellow_optical_power_mW = \
                tool_belt.measure_g_r_y_power(
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        yellow_power_list.append(yellow_optical_power_mW)
        
    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'parameters_sig': parameters_sig,
            'parameters_sig-units': tool_belt.get_nv_sig_units(),
            'coords': coords,
            'nd_filter': nd_filter,
            'aom_power_list': aom_power_list,
            'aom_power_list-units': 'V',
            'yellow_power_list': yellow_power_list,
            'yellow_power_list-units': 'mW',
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
            'num_runs':num_reps,
            'nv0_list': nv0_list,
            'nv0_list-units': 'counts',
            'nvm_list': nvm_list,
            'nvm_list-units': 'counts',                
            'nv0_avg_list': nv0_avg_list,
            'nv0_avg_list-units': 'counts',
            'nv0_ste_list': nv0_ste_list,
            'nv0_ste_list-units': 'counts',
            'nvm_avg_list': nvm_avg_list,
            'nvm_avg_list-units': 'counts',
            'nvm_ste_list': nvm_ste_list,
            'nvm_ste_list-units': 'counts'
            }
            
#    print(nv0_avg_list)
#    print(nvm_avg_list)
    file_path = tool_belt.get_file_path(__file__, timestamp, parameters_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
    
    return


# %%
def collect_charge_counts_yellow_time(coords, parameters_sig, readout_time_list, num_reps, apd_indices ):
    with labrad.connect() as cxn:
        tool_belt.reset_cfm(cxn)
    
    nv0_list = []
    nvm_list = []
    nv0_avg_list = []
    nv0_ste_list = []
    nvm_avg_list = []
    nvm_ste_list = []
#    yellow_power_list = []
    
    for readout_time in readout_time_list:
        print(str(readout_time/10**6) + ' ms')
        nv_sig = copy.deepcopy(parameters_sig)
        nv_sig['coords'] = coords
        nv_sig['pulsed_SCC_readout_dur'] = readout_time
#        time.sleep(0.002)
        
        ret_vals = main(nv_sig,  apd_indices,num_reps)
        nv0_counts, nvm_counts = ret_vals
        nv0_counts = [int(el) for el in nv0_counts]
        nvm_counts = [int(el) for el in nvm_counts]
        
        nv0_list.append(nv0_counts)
        nvm_list.append(nvm_counts)
        
        nv0_avg = numpy.average(nv0_counts)
        nv0_ste = stats.sem(nv0_counts)
        nv0_avg_list.append(nv0_avg)
        nv0_ste_list.append(nv0_ste)
        
        nvm_avg = numpy.average(nvm_counts)
        nvm_ste = stats.sem(nvm_counts)
        nvm_avg_list.append(nvm_avg)
        nvm_ste_list.append(nvm_ste)    
    
        # measure laser powers:
        green_optical_power_pd, green_optical_power_mW, \
                red_optical_power_pd, red_optical_power_mW, \
                yellow_optical_power_pd, yellow_optical_power_mW = \
                tool_belt.measure_g_r_y_power(
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
        
    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'parameters_sig': parameters_sig,
            'parameters_sig-units': tool_belt.get_nv_sig_units(),
            'coords': coords,
            'readout_time_list': readout_time_list,
            'readout_time_list-units': 'ns',
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
            'num_runs':num_reps,
            'nv0_list': nv0_list,
            'nv0_list-units': 'counts',
            'nvm_list': nvm_list,
            'nvm_list-units': 'counts',                
            'nv0_avg_list': nv0_avg_list,
            'nv0_avg_list-units': 'counts',
            'nv0_ste_list': nv0_ste_list,
            'nv0_ste_list-units': 'counts',
            'nvm_avg_list': nvm_avg_list,
            'nvm_avg_list-units': 'counts',
            'nvm_ste_list': nvm_ste_list,
            'nvm_ste_list-units': 'counts'
            }
            
#    print(nv0_avg_list)
#    print(nvm_avg_list)
    file_path = tool_belt.get_file_path(__file__, timestamp, parameters_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)
    
    return

# %%
def collect_charge_counts_list(coords_list, parameters_sig, num_reps, apd_indices):
    with labrad.connect() as cxn:
        tool_belt.reset_cfm(cxn)
    
    nv0_list = []
    nvm_list = []
    nv0_avg_list = []
    nv0_ste_list = []
    nvm_avg_list = []
    nvm_ste_list = []
    
    
    for coords in coords_list:
        print(coords)
        nv_sig = copy.deepcopy(parameters_sig)
        nv_sig['coords'] = coords
#        time.sleep(0.002)
        
        ret_vals = main(nv_sig,  apd_indices,num_reps)
        nv0_counts, nvm_counts = ret_vals
        nv0_counts = [int(el) for el in nv0_counts]
        nvm_counts = [int(el) for el in nvm_counts]
        
        nv0_list.append(nv0_counts)
        nvm_list.append(nvm_counts)
        
        nv0_avg = numpy.average(nv0_counts)
        nv0_ste = stats.sem(nv0_counts)
        nv0_avg_list.append(nv0_avg)
        nv0_ste_list.append(nv0_ste)
        
        nvm_avg = numpy.average(nvm_counts)
        nvm_ste = stats.sem(nvm_counts)
        nvm_avg_list.append(nvm_avg)
        nvm_ste_list.append(nvm_ste)    
    
        # measure laser powers:
        green_optical_power_pd, green_optical_power_mW, \
                red_optical_power_pd, red_optical_power_mW, \
                yellow_optical_power_pd, yellow_optical_power_mW = \
                tool_belt.measure_g_r_y_power(
                                  nv_sig['am_589_power'], nv_sig['nd_filter'])
    

        timestamp = tool_belt.get_time_stamp()
        raw_data = {'timestamp': timestamp,
                'parameters_sig': parameters_sig,
                'parameters_sig-units': tool_belt.get_nv_sig_units(),
                'coords_list': coords_list,
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
                'num_runs':num_reps,
                'nv0_list': nv0_list,
                'nv0_list-units': 'counts',
                'nvm_list': nvm_list,
                'nvm_list-units': 'counts',                
                'nv0_avg_list': nv0_avg_list,
                'nv0_avg_list-units': 'counts',
                'nv0_ste_list': nv0_ste_list,
                'nv0_ste_list-units': 'counts',
                'nvm_avg_list': nvm_avg_list,
                'nvm_avg_list-units': 'counts',
                'nvm_ste_list': nvm_ste_list,
                'nvm_ste_list-units': 'counts'
                }
        
#    print(nv0_avg_list)
#    print(nvm_avg_list)
    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path + '-nv_list')
    
    return

# %% Run the files

if __name__ == '__main__':
    apd_indicies = [0]
    
    start_coords_list =[    
    [0.317, 0.338, 5.09],
[0.190, 0.345, 5.00],
[-0.031, 0.299, 4.94],
[-0.042, 0.265, 4.98],
[-0.082, 0.266, 4.96], #
[0.333, 0.277, 4.93],
[0.32, 0.223, 5.01],
[-0.060, 0.179, 4.93], #
[0.187, 0.127, 4.97],
[0.172, 0.140, 4.91],
[0.033, 0.085, 4.93],
[0.125, 0.049, 4.97],
[-0.010, 0.052, 4.97],
[0.057, -0.106, 4.95],
[0.385, -0.174, 4.99],
[0.134, -0.192, 4.93],
[0.400, -0.299, 4.97],
[0.374, -0.296, 4.96],
[-0.194, -0.326, 4.97],
[0.260, -0.382, 4.98],
]
    expected_count_list = [75, 52, 40, 48, 46, 45, 50, 56, 45, 53, 44, 36, 
                           57, 63, 44, 72, 60, 40, 40, 45]

    
    base_nv_sig  = { 'coords':None,
            'name': 'goeppert-mayer-nv_2021_01_26',
            'expected_count_rate': None,'nd_filter': 'nd_1.0',
            'color_filter': '635-715 bp', 
#            'color_filter': '715 lp',
            'pulsed_readout_dur': 300,
            'pulsed_SCC_readout_dur': 3*10**7, 'am_589_power': 0.3, 
            'pulsed_initial_ion_dur': 25*10**3,
            'pulsed_shelf_dur': 200, 
            'am_589_shelf_power': 0.35,
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 130, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':10, 
            'magnet_angle': 0,
            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}
    
#    list_ = [ nv_coords_list[0], nv_coords_list[3],nv_coords_list[8], nv_coords_list[12],nv_coords_list[15]]
#    collect_charge_counts_list(nv_coords_list, base_nv_sig, 60, apd_indicies)
    collect_charge_counts_list(start_coords_list, base_nv_sig, 500, apd_indicies)

#    nd_filter = 'nd_0'
#    aom_power_list = [0.2,0.3,0.4,0.5,0.6,0.7]
#    for i in [0]:
#        coords = start_coords_list[i]
#        base_nv_sig['expected_count_rate'] = expected_count_list[i]
#        base_nv_sig['name'] = 'goeppert-mayer-nv{}_2021_01_11'.format(0)
#        collect_charge_counts_yellow_pwr(coords, base_nv_sig, nd_filter, aom_power_list, 1000, apd_indicies )        
#    nd_filter = 'nd_0.5'
#    aom_power_list = [0.2,0.3,0.4,0.5,0.6,0.7]
#    for i in [0,1,2,3]:
#        coords = nv_coords_list[i]
#        base_nv_sig['expected_count_rate'] = expected_count_list[i]
#        base_nv_sig['name'] = 'goeppert-mayer-nv{}_2021_01_11'.format(i)
#        collect_charge_counts_yellow_pwr(coords, base_nv_sig, nd_filter, aom_power_list, 1000, apd_indicies )        
#    nd_filter = 'nd_1.0'
#    for i in [0,1,2,3]:
#        coords = nv_coords_list[i]
#        base_nv_sig['expected_count_rate'] = expected_count_list[i]
#        base_nv_sig['name'] = 'goeppert-mayer-nv{}_2021_01_11'.format(i)
#        collect_charge_counts_yellow_pwr(coords, base_nv_sig, nd_filter, aom_power_list, 1000, apd_indicies )
#        
#    nd_filter = 'nd_1.5'
#    for i in [0,1,2,3]:
#        coords = nv_coords_list[i]
#        base_nv_sig['expected_count_rate'] = expected_count_list[i]
#        base_nv_sig['name'] = 'goeppert-mayer-nv{}_2021_01_11'.format(i)
#        collect_charge_counts_yellow_pwr(coords, base_nv_sig, nd_filter, aom_power_list, 1000, apd_indicies )
#    
#    base_nv_sig  = { 'coords':None,
#            'name': 'goeppert-mayer-nv_2021_01_11',
#            'expected_count_rate': None,'nd_filter': 'nd_1.0',
#            'color_filter': '635-715 bp',
##            'color_filter': '715 lp',
#            'pulsed_readout_dur': 300,
#            'pulsed_SCC_readout_dur': 30000000, 'am_589_power': 0.7, 
#            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120, 
#            'pulsed_initial_ion_dur': 25*10**3,
#            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':20, 
#            'magnet_angle': 0,
#            "resonance_LOW": 2.7,"rabi_LOW": 146.2, "uwave_power_LOW": 9.0,
#            "resonance_HIGH": 2.9774,"rabi_HIGH": 95.2,"uwave_power_HIGH": 10.0}          
#    readout_time_list = [10*10**3, 
#                               50*10**3, 100*10**3,500*10**3, 
#                               1*10**6,  5*10**6, 
#                                1*10**7,
#                               5*10**7]
#    for i in [0,1,2,3]:
#        coords = nv_coords_list[i]
#        base_nv_sig['expected_count_rate'] = expected_count_list[i]
#        base_nv_sig['name'] = 'goeppert-mayer-nv{}_2021_01_11'.format(i)
#        collect_charge_counts_yellow_time(coords, base_nv_sig, readout_time_list, 1000, apd_indicies )