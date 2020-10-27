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
import copy
import scipy.stats as stats
# %%

def red_scan(nv_sig, apd_indices):
#    image_sample.main(nv_sig,  0.1, 0.1, 60, apd_indices, 638, save_data=False, plot_data=False, readout =10**3)
#    time.sleep(1)
    return


def green_scan(nv_sig, apd_indices):
    image_sample.main(nv_sig,  0.1, 0.1, 60, apd_indices, 532, save_data=False, plot_data=False, readout =10**5)
    return

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
    
    drift = tool_belt.get_drift()
    
    # Target 
    
    #If we are pulsing an initial laser:
    if target_color:
        # move the galvo to the target # MIGHT WANT TO MOVE thiS OUTSIDE IF
        target_coords = target_sig['coords'] 
        target_coords_drift = numpy.array(target_coords) + numpy.array(drift)
#        print(target_coords)
#        print(target_coords_drift)
        tool_belt.set_xyz(cxn,target_coords_drift)
#        time.sleep(0.1)
        # Pusle the laser
        if target_color == 532:
            target_pulse_time = target_sig['pulsed_reionization_dur']
            laser_delay = laser_515_delay
        elif target_color == 638:
            target_pulse_time = target_sig['pulsed_ionization_dur']
            laser_delay = laser_638_delay
        seq_args = [laser_delay, int(target_pulse_time), 0.0, target_color]           
        seq_args_string = tool_belt.encode_seq_args(seq_args)            
        cxn.pulse_streamer.stream_immediate(target_file_name, 1, seq_args_string) 
        

    
    # then readout with yellow
    aom_ao_589_pwr = readout_sig['am_589_power']
    nd_filter = readout_sig['nd_filter']  
    readout_pulse_time = readout_sig['pulsed_SCC_readout_dur']
    laser_delay = aom_589_delay
    
    cxn.filter_slider_ell9k.set_filter(nd_filter)  
   # move the galvo to the readout
    readout_coords = readout_sig['coords']
    readout_coords_drift = numpy.array(readout_coords) + numpy.array(drift)
#    print(readout_coords)
#    print(readout_coords_drift)
    tool_belt.set_xyz(cxn, readout_coords_drift)
#    time.sleep(0.1)
    
    seq_args = [laser_delay, int(readout_pulse_time), aom_ao_589_pwr, apd_index, readout_color]           
    seq_args_string = tool_belt.encode_seq_args(seq_args)            
    cxn.pulse_streamer.stream_load(readout_file_name, seq_args_string)         
    # collect the counts
    cxn.apd_tagger.start_tag_stream(apd_indices)
    # Clear the buffer
    cxn.apd_tagger.clear_buffer()
    # Run the sequence
    cxn.pulse_streamer.stream_immediate(readout_file_name, 1, seq_args_string)

    new_counts = cxn.apd_tagger.read_counter_simple(1)
#    sample_counts = new_counts[0]
#    print(sample_counts)

    # signal counts are even - get every second element starting from 0
    sig_counts = new_counts[0] 
    
    
    cxn.apd_tagger.stop_tag_stream()
    
    return sig_counts
#%%
def charge_spot(readout_coords,target_A_coords, target_B_coords, parameters_sig, num_runs, init_scan):
    apd_indices = [0]
#    num_runs = 25
    # add the coords to the dictionry of measurement paramters
    readout_sig = copy.deepcopy(parameters_sig)
    readout_sig['coords'] = readout_coords
     
    target_A_sig = copy.deepcopy(parameters_sig)
    target_A_sig['coords'] = target_A_coords
    
    target_B_sig = copy.deepcopy(parameters_sig)
    target_B_sig['coords'] = target_B_coords
    
    readout_sec = readout_sig['pulsed_SCC_readout_dur'] / 10**9
       
    # create some lists for dataopti_coords_list
    opti_coords_list = []
    control = []
    green_readout = []
    red_readout = []
    green_target_A = []
    red_target_A = []
    green_target_B = []
    red_target_B = []

    start_timestamp = tool_belt.get_time_stamp()    
    green_scan(readout_sig, apd_indices)
#    with labrad.connect() as cxn:
#        opti_coords = optimize.main_with_cxn(cxn, readout_sig, apd_indices, 532, disable=False)
#        opti_coords_list.append(opti_coords)    
        
    for run in range(num_runs):
        print('run {}'.format(run))
        #optimize
#        if run % 10 == 0:
#            with labrad.connect() as cxn:
#                opti_coords = optimize.main_with_cxn(cxn, readout_nv_sig, apd_indices, 532, disable=False)
#                opti_coords_list.append(opti_coords)
            
        # Step through the experiments
    
        # control: readout NV_readout
#        if init_scan == 532:
#            green_scan(readout_sig, apd_indices)
#        elif init_scan == 638:
#            red_scan(readout_sig, apd_indices)        
#        sig_count =  main(parameters_sig, readout_sig, None, 589, apd_indices)
#        control_kcps = (sig_count  / 10**3) / readout_sec
#        control.append(control_kcps)
#        print('control: {} counts'.format(sig_count) )
##        print('control: {} kcps'.format(control_kcps) )
        
        # green_readout: measure NV after green pulse on readout NV
        if init_scan == 532:
            green_scan(readout_sig, apd_indices)
        elif init_scan == 638:
            red_scan(readout_sig, apd_indices)   
        sig_count =  main(readout_sig, readout_sig, 532, 589, apd_indices)
        green_readout_kcps = (sig_count  / 10**3) / readout_sec
        green_readout.append(green_readout_kcps)
        print('green_readout: {} counts'.format(sig_count) )
#        print('green readout: {} kcps'.format(green_readout_kcps) )
 
        # red_readout: measure NV after red pulse on readout NV
#        if init_scan == 532:
#            green_scan(readout_sig, apd_indices)
#        elif init_scan == 638:
#            red_scan(readout_sig, apd_indices)   
#        sig_count =  main(readout_sig, readout_sig, 638, 589, apd_indices)
#        red_readout_kcps = (sig_count  / 10**3) / readout_sec
#        red_readout.append(red_readout_kcps)
#        print('red_readout: {} counts'.format(sig_count) )
##        print('red readout: {} kcps'.format(red_readout_kcps) )
#        
#        # green_target: measure NV after green pulse on target NV
#        if init_scan == 532:
#            green_scan(readout_sig, apd_indices)
#        elif init_scan == 638:
#            red_scan(readout_sig, apd_indices)   
#        sig_count =  main(target_A_sig, readout_sig, 532, 589, apd_indices)
#        green_target_A_kcps = (sig_count  / 10**3) / readout_sec
#        green_target_A.append(green_target_A_kcps)
#        print('green_target_A: {} counts'.format(sig_count) )
##        print('green target: {} kcps'.format(green_target_kcps) )
# 
#        # red_target: measure NV after red pulse on target NV
#        if init_scan == 532:
#            green_scan(readout_sig, apd_indices)
#        elif init_scan == 638:
#            red_scan(readout_sig, apd_indices)   
#        sig_count =  main(target_A_sig, readout_sig, 638, 589, apd_indices)
#        red_target_A_kcps = (sig_count  / 10**3) / readout_sec
#        red_target_A.append(red_target_A_kcps)
#        print('red_target_A: {} counts'.format(sig_count) )
##        print('red target: {} kcps'.format(red_target_kcps) )
#        
#        # green_B: measure NV after green pulse on dark spot
#        if init_scan == 532:
#            green_scan(readout_sig, apd_indices)
#        elif init_scan == 638:
#            red_scan(readout_sig, apd_indices)  
#        sig_count =  main(target_B_sig, readout_sig, 532, 589, apd_indices)
#        green_target_B_kcps = (sig_count  / 10**3) / readout_sec
#        green_target_B.append(green_target_B_kcps)
#        print('green_target_B: {} counts'.format(sig_count) )
##        print('green dark: {} kcps'.format(green_dark_kcps) )
# 
#        # red_B: measure NV after red pulse on dark spot
#        if init_scan == 532:
#            green_scan(readout_sig, apd_indices)
#        elif init_scan == 638:
#            red_scan(readout_sig, apd_indices)  
#        sig_count =  main(target_B_sig, readout_sig, 638, 589, apd_indices)
#        red_target_B_kcps = (sig_count  / 10**3) / readout_sec
#        red_target_B.append(red_target_B_kcps)
#        print('red_target_B: {} counts'.format(sig_count) )
##        print('red dark: {} kcps'.format(red_dark_kcps) )
#        
        raw_data = {'start_time': start_timestamp,
                'readout_coords': readout_coords,
                'target_A_coords': target_A_coords,
                'target_B_coords': target_B_coords,
                'parameters_sig': parameters_sig,
                'parameters_sig-units': tool_belt.get_nv_sig_units(),
                'num_runs':num_runs,
                'opti_coords_list': opti_coords_list,
                'control': control,
                'control-units': 'kcps',
                'green_readout': green_readout,
                'green_readout-units': 'kcps',
                'red_readout': red_readout,
                'red_readout-units': 'kcps',
                'green_target_A': green_target_A,
                'green_target_A-units': 'kcps',
                'red_target_A': red_target_A,
                'red_target_A-units': 'kcps',
                'green_target_B': green_target_B,
                'green_target_B-units': 'kcps',
                'red_target_B': red_target_B,
                'red_target_B-units': 'kcps',
                }
    
        file_path = tool_belt.get_file_path(__file__, start_timestamp, parameters_sig['name'], 'incremental')
        if init_scan == 532:
            tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge-green_init')   
        elif init_scan == 638:
            tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge-red_init')       
        
    control_avg = numpy.average(control)
    control_ste = stats.sem(control)
    print('control measurement avg: {} +/- {} kcps'.format(control_avg, control_ste))
    green_readout_avg = numpy.average(green_readout)
    green_readout_ste = stats.sem(green_readout)
    print('green readout measurement avg: {} +/- {} kcps'.format(green_readout_avg, green_readout_ste))
    red_readout_avg = numpy.average(red_readout)
    red_readout_ste = stats.sem(red_readout)
    print('red readout measurement avg: {} +/- {} kcps'.format(red_readout_avg, red_readout_ste))
    green_target_A_avg = numpy.average(green_target_A)
    green_target_A_ste = stats.sem(green_target_A)
    print('green target measurement avg: {} +/- {} kcps'.format(green_target_A_avg, green_target_A_ste))
    red_target_A_avg = numpy.average(red_target_A)
    red_target_A_ste = stats.sem(red_target_A)
    print('red target measurement avg: {} +/- {} kcps'.format(red_target_A_avg, red_target_A_ste))
    green_target_B_avg = numpy.average(green_target_B)
    green_target_B_ste =  stats.sem(green_target_B)
    print('green dark measurement avg: {} +/- {} kcps'.format(green_target_B_avg, green_target_B_ste))
    red_target_B_avg = numpy.average(red_target_B)
    red_target_B_ste = stats.sem(red_target_B)
    print('red dark measurement avg: {} +/- {} kcps'.format(red_target_B_avg, red_target_B_ste))
    
    # measure laser powers:
    green_optical_power_pd, green_optical_power_mW, \
            red_optical_power_pd, red_optical_power_mW, \
            yellow_optical_power_pd, yellow_optical_power_mW = \
            tool_belt.measure_g_r_y_power( 
                              readout_sig['am_589_power'], parameters_sig['nd_filter'])
            
    # Save

    
    timestamp = tool_belt.get_time_stamp()
    raw_data = {'timestamp': timestamp,
            'readout_coords': readout_coords,
            'target_A_coords': target_A_coords,
            'target_B_coords': target_B_coords,
            'parameters_sig': parameters_sig,
            'parameters_sig-units': tool_belt.get_nv_sig_units(),
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
            'num_runs':num_runs,
            'opti_coords_list': opti_coords_list,
            'control': control,
            'control-units': 'kcps',
            'green_readout': green_readout,
            'green_readout-units': 'kcps',
            'red_readout': red_readout,
            'red_readout-units': 'kcps',
            'green_target_A': green_target_A,
            'green_target_A-units': 'kcps',
            'red_target_A': red_target_A,
            'red_target_A-units': 'kcps',
            'green_target_B': green_target_B,
            'green_target_B-units': 'kcps',
            'red_target_B': red_target_B,
            'red_target_B-units': 'kcps',
            
            'control_avg': control_avg,
            'control_avg-units': 'kcps',
            'green_readout_avg': green_readout_avg,
            'green_readout_avg-units': 'kcps',
            'red_readout_avg': red_readout_avg,
            'red_readout_avg-units': 'kcps',
            'green_target_A_avg': green_target_A_avg,
            'green_target_A_avg-units': 'kcps',
            'red_target_A_avg': red_target_A_avg,
            'red_target_A_avg-units': 'kcps',
            'green_target_B_avg': green_target_B_avg,
            'green_target_B_avg-units': 'kcps',
            'red_target_B_avg': red_target_B_avg,
            'red_target_B_avg-units': 'kcps',
            
            'control_ste': control_ste,
            'control_ste-units': 'kcps',
            'green_readout_ste': green_readout_ste,
            'green_readout_ste-units': 'kcps',
            'red_readout_ste': red_readout_ste,
            'red_readout_ste-units': 'kcps',
            'green_target_A_ste': green_target_A_ste,
            'green_target_A_ste-units': 'kcps',
            'red_target_A_ste': red_target_A_ste,
            'red_target_A_ste-units': 'kcps',
            'green_target_B_ste': green_target_B_ste,
            'green_target_B_ste-units': 'kcps',
            'red_target_B_ste': red_target_B_ste,
            'red_target_B_ste-units': 'kcps',
            }

    file_path = tool_belt.get_file_path(__file__, timestamp, parameters_sig['name'])
    if init_scan == 532:
        tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge-green_init')   
    elif init_scan == 638:
        tool_belt.save_raw_data(raw_data, file_path + '-isoalted_nv_charge-red_init')    

    
    print(' \nRoutine complete!')
    return


# %% Run the files
    
if __name__ == '__main__':
    sample_name = 'goeppert-mayer'
    NVA_coords = [0.432, -0.091,  5.1]
    NVB_coords = [0.464, -0.138,  5.1]
    dark_spot_1_coords = [0.436, -0.069,  5.1]
    dark_spot_2_coords = [0.480, -0.138, 5.1]
    
    # The parameters that we want to run these measurements with
    base_nv_sig  = { 'coords':None,
            'name': '{}-NVA'.format(sample_name),
            'expected_count_rate': 100, 'nd_filter': 'nd_0',
            'pulsed_SCC_readout_dur': 2*10**6, 'am_589_power': 0.25, 
            'pulsed_ionization_dur': 10**3, 'cobalt_638_power': 120, 
            'pulsed_reionization_dur': 100*10**3, 'cobalt_532_power':19,
            'magnet_angle': 0}
    

    # run the measurements!
    charge_spot(NVA_coords, NVB_coords, dark_spot_1_coords, base_nv_sig, 2, 638)
    charge_spot(dark_spot_1_coords, NVA_coords, dark_spot_2_coords,base_nv_sig, 2, 638)
    
