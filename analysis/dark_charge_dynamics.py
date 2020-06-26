# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:38:50 2020
Plotting dark time charge dynamics
@author: agardill
"""

# %% Imports

import numpy
from scipy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import utils.tool_belt as tool_belt

def main(file_1, file_2, file_1_label, fil_2_label):
    # File 1
    data = tool_belt.get_raw_data('SCC_dark_time_dynamics/branch_Spin_to_charge/2020_06/', 
                                  file_1)
    nv_sig = data['nv_sig']
    test_pulse_dur_list_1 = numpy.array(data['test_pulse_dur_list'])/10**6
    sig_count_list_1 = data['sig_count_list']
    init_pulse_time_1 = data["initial_pulse_time"]
    readout_1 = nv_sig["pulsed_SCC_readout_dur"]
    init_color_1 = data["init_color_ind"]
    
    # manipulate the data to normalize 
    first_point = sig_count_list_1[0]
    last_point = sig_count_list_1[-1]
    
#    if init_color_1 == '532 nm':
    sig_count_list_1 = (numpy.array(sig_count_list_1) - first_point)/ (first_point)
#    elif init_color_1 == '638 nm':
#        sig_count_list_1 = (numpy.array(sig_count_list_1) - last_point)/ (last_point)
    # Area A1
    data = tool_belt.get_raw_data('SCC_dark_time_dynamics/branch_Spin_to_charge/2020_06/', 
                                  file_2)
    nv_sig = data['nv_sig']
    test_pulse_dur_list_2 = numpy.array(data['test_pulse_dur_list'])/10**6
    sig_count_list_2 = data['sig_count_list']
    init_pulse_time_2 = data["initial_pulse_time"]
    readout_2 = nv_sig["pulsed_SCC_readout_dur"]
    init_color_2 = data["init_color_ind"]
    
    # manipulate the data to normalize 
    first_point = sig_count_list_2[0]
    last_point = sig_count_list_2[-1]
    
    
#    if init_color_2 == '532 nm':
    sig_count_list_2 = (numpy.array(sig_count_list_2)  - first_point)/ (first_point)
#    elif init_color_2 == '638 nm':
#        sig_count_list_2 = (numpy.array(sig_count_list_2) - last_point)/ (last_point)
    
    #if init_pulse_time_A5 != init_pulse_time_A1:
    #    print('Initial pulse times are not equal, do you with to continue?')
        
    fig, ax = plt.subplots(1,1, figsize=(10, 8))
    ax.plot(test_pulse_dur_list_1, sig_count_list_1,'ko',
                        label = file_1_label)
    ax.plot(test_pulse_dur_list_2, sig_count_list_2,'bo',
                        label = file_2_label)
    ax.set_xlabel('Dark time (ms)')
    ax.set_ylabel('% diff in counts')
    ax.set_title('Readout with {} ms 589 nm after {} ms {} pulse'.format(\
                       readout_2/10**6,  init_pulse_time_2/10**6, init_color_2))
    ax.set_xscale('log')
    ax.legend()
# %% Run the files
    
if __name__ == '__main__':

    file_1_label = 'hopper'
    file_2_label = 'bachman'
    # red 1000 ms, readout 1 ms
#    hopper_file = '2020_06_25-15_34_15-Hopper-ensemble-dark_time_w_638'
#    bachman_file = '2020_06_19-18_34_13-bachman-A1-B1-dark_time_w_638'
    
    # red 1 ms, readout 1 ms
    hopper_file = '2020_06_25-17_18_13-Hopper-ensemble-dark_time_w_638'
    bachman_file = '2020_06_19-16_06_17-bachman-A1-B1-dark_time_w_638'
    
    
    # green 1 ms, readout 1 ms
#    hopper_file = '2020_06_25-14_36_04-Hopper-ensemble-dark_time_w_532'
#    bachman_file = '2020_06_19-15_21_04-bachman-A1-B1-dark_time_w_532'
    
    main(hopper_file, bachman_file, file_1_label, file_2_label)
