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

def main(A5_data_file, A1_data_file):
    # Area A5
    data = tool_belt.get_raw_data('SCC_dark_time_dynamics/branch_Spin_to_charge/2020_06/', 
                                  A5_data_file)
    nv_sig = data['nv_sig']
    test_pulse_dur_list_5 = numpy.array(data['test_pulse_dur_list'])/10**6
    sig_count_list_A5 = data['sig_count_list']
    init_pulse_time_A5 = data["initial_pulse_time"]
    readout_A5 = nv_sig["pulsed_SCC_readout_dur"]
    init_color_A5 = data["init_color_ind"]
    
    # manipulate the data to normalize 
    first_point = sig_count_list_A5[0]
    last_point = sig_count_list_A5[-1]
    
    if init_color_A5 == '532 nm':
        sig_count_list_A5 = (numpy.array(sig_count_list_A5) - last_point)/ (first_point - last_point)
    elif init_color_A5 == '638 nm':
        sig_count_list_A5 = (numpy.array(sig_count_list_A5) - first_point)/ (last_point - first_point)
    # Area A1
    data = tool_belt.get_raw_data('SCC_dark_time_dynamics/branch_Spin_to_charge/2020_06/', 
                                  A1_data_file)
    nv_sig = data['nv_sig']
    test_pulse_dur_list_1 = numpy.array(data['test_pulse_dur_list'])/10**6
    sig_count_list_A1 = data['sig_count_list']
    init_pulse_time_A1 = data["initial_pulse_time"]
    readout_A1 = nv_sig["pulsed_SCC_readout_dur"]
    init_color_A1 = data["init_color_ind"]
    
    # manipulate the data to normalize 
    first_point = sig_count_list_A1[0]
    last_point = sig_count_list_A1[-1]
    
    
    if init_color_A1 == '532 nm':
        sig_count_list_A1 = (numpy.array(sig_count_list_A1) - last_point)/ (first_point - last_point)
    elif init_color_A1 == '638 nm':
        sig_count_list_A1 = (numpy.array(sig_count_list_A1) - first_point)/ (last_point - first_point)
    
    #if init_pulse_time_A5 != init_pulse_time_A1:
    #    print('Initial pulse times are not equal, do you with to continue?')
        
    fig, ax = plt.subplots(1,1, figsize=(10, 8))
    ax.plot(test_pulse_dur_list_5, sig_count_list_A5,'ko',
                        label = 'Area A5')
    ax.plot(test_pulse_dur_list_1, sig_count_list_A1,'bo',
                        label = 'Area A1')
    ax.set_xlabel('Dark time (ms)')
    ax.set_ylabel('Signal Counts')
    ax.set_title('Readout with {} ms 589 nm after {} ms {} pulse'.format(\
                       readout_A1/10**6,  init_pulse_time_A1/10**6, init_color_A1))
    ax.legend()
# %% Run the files
    
if __name__ == '__main__':

    # red 0.1 ms, readout 1 ms
#    A1_file = '2020_06_12-16_03_53-bachman-A1-A6-dark_time_w_638'
#    A5_file = '2020_06_03-15_10_19-bachman-A1-dark_time_w_638'
    
    # red 1 ms, readout 1 ms
    A1_file = '2020_06_12-16_12_54-bachman-A1-A6-dark_time_w_638'
    A5_file = '2020_06_04-12_44_00-bachman-A1-dark_time_w_638'
    
    # green 1 ms, readout 10 ms
#    A1_file = '2020_06_12-15_35_26-bachman-A1-A6-dark_time_w_532'
#    A5_file = '2020_06_03-16_50_12-bachman-A1-dark_time_w_532'
    
    # green 1 ms, readout 1 ms
#    A1_file = '2020_06_12-16_00_59-bachman-A1-A6-dark_time_w_532'
#    A5_file = '2020_06_03-16_43_38-bachman-A1-dark_time_w_532'
    
    main(A5_file, A1_file)
