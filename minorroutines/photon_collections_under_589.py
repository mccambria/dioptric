#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:39:55 2019

@author: yanfeili
"""

# %% import
import utils.tool_belt as tool_belt
import majorroutines.optimize as optimize
import numpy
import os
import matplotlib.pyplot as plt
import labrad

#%% Main
# Connect to labrad in this file, as opposed to control panel
def main(nv_sig, apd_indices, readout_power,readout_time,ionization_power, 
                  ionization_time,state,num_runs):
    with labrad.connect() as cxn:
        main_with_cxn(cxn, nv_sig, apd_indices, readout_power,readout_time,ionization_power, 
                  ionization_time,state,num_runs)

def main_with_cxn(cxn, nv_sig, apd_indices, readout_power,readout_time,ionization_power, 
                  ionization_time,state,num_runs):

    tool_belt.reset_cfm(cxn)

# %% Initial Calculation and setup
    apd_indices = [0]
    
    shared_params = tool_belt.get_shared_parameters_dict(cxn)
    
    #Define some parameters
    
    #delay of aoms and laser
    aom_delay589 = shared_params['589_aom_delay'] 
    #gate_time in this sequence is the readout time ~8 ms 
    gate_time = readout_time
    # Analyze the sequence
    file_name = os.path.basename(__file__)
    seq_args = [gate_time, aom_delay589,readout_time,apd_indices]
    seq_args = [int(el) for el in seq_args]
#    print(seq_args)
#    return
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    cxn.pulse_streamer.stream_load(file_name, seq_args_string)

    # Set up our data structure, an array of NaNs that we'll fill
    # we repeatively collect photons for tR 
    sig_counts = numpy.empty(num_runs, dtype=numpy.uint32)
    sig_counts[:] = numpy.nan
    # norm_avg_sig = numpy.empty([num_runs, num_steps])
    
    # create a list to store the optimized coordinates
    opti_coords_list = []
    
#%% Collect data
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):

        print('Run index: {}'. format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Optimize
        opti_coords = optimize.main_with_cxn(cxn, nv_sig, readout_power, apd_indices, 532)
        opti_coords_list.append(opti_coords)
        
        # Load the APD
        cxn.apd_tagger.start_tag_stream(apd_indices)
    
        # Get the counts
        new_counts = cxn.apd_tagger.read_counter_separate_gates(1)

        sample_counts = new_counts[0]

        # there is only one readout period 
        sig_gate_counts = sample_counts[0]
        sig_counts[run_ind] = sum(sig_gate_counts)


    cxn.apd_tagger.stop_tag_stream()
#%% Save data 
    timestamp = tool_belt.get_time_stamp()
     
    raw_data = {'timestamp': timestamp,
            'nv_sig': nv_sig,
            'readout_power':readout_power,
            'readout_power_unit':'nW',
            'readout_time':readout_time,
            'readout_time_unit':'ns',
            'nv_sig-units': tool_belt.get_nv_sig_units(),
            'num_runs': num_runs,
            'sig_counts': sig_counts.astype(int).tolist(),
            'sig_counts-units': 'counts'}

    file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig['name'])
    tool_belt.save_raw_data(raw_data, file_path)    
#%% plot the data
    def plot_Probability_distribution(aList,colorlabel):
        aList = aList.tolist()
        def get_unique_value(aList):
            unique_value_list = []
            for i in range(0,len(aList)):
                if aList[i] not in unique_value_list:
                    unique_value_list.append(aList[i])
            return unique_value_list
        unique_value = get_unique_value(aList)
        relative_frequency = []
        for i in range(0,len(unique_value)):
            relative_frequency.append(aList.count(unique_value[i])/ (len(aList)))
        fig = plt.plot(unique_value,relative_frequency,'or', color = colorlabel)
        plt.xlabel('number of photons (n)')
        plt.ylabel('P(n)')
        return fig
    plot_Probability_distribution(list(sig_counts),'b')
