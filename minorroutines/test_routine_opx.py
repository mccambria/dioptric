# -*- coding: utf-8 -*-
"""Template for minor routines. Minor routines are routines for which we will
probably not want to save the data or routines that are used infrequently.

Created on Sun Jun 16 11:38:17 2019

@author: mccambria
"""


# %% Imports


import utils.tool_belt as tool_belt
import labrad
import numpy as np
import time
# %% Functions


def clean_up(cxn):

    tool_belt.reset_cfm()


# %% Main


def main(nv_sig, delay, readout_time, apd_index, laser_name, laser_power, num_reps ):
    """When you run the file, we'll call into main, which should contain the
    body of the routine.
    """

    with labrad.connect() as cxn:
        
        new_counts, new_times, new_channels = main_with_cxn(cxn, delay, readout_time, apd_index, laser_name, laser_power, num_reps )
    
    return new_counts, new_times, new_channels
    
def main_with_cxn(cxn, delay, readout_time, apd_index, laser_name, laser_power, num_reps ):

    # %% Initial set up here
    
    # tool_belt.reset_cfm(cxn)

    # seq_args = [delay, readout_time, apd_index, laser_name, laser_power ]
    seq_args = [int(delay), int(readout_time), int(apd_index), laser_name, int(laser_power) ]
    # print(seq_args)
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    
    seq_file = 'simple_readout.py'
    
    counter_server = tool_belt.get_counter_server(cxn)
    tagger_server = tool_belt.get_tagger_server(cxn)
    pulsegen_server = tool_belt.get_pulsegen_server(cxn)
    
    tagger_server.start_tag_stream([0])
    # pulsegen_server.stream_load(seq_file, seq_args_string)
    # pulsegen_server.stream_start(num_reps)
    pulsegen_server.stream_immediate(seq_file, num_reps, seq_args_string)
    
    num_read_so_far = 0
    total_num_samples = 1
    total_counts = []
    # time.sleep(.12)
    a=counter_server.read_counter_complete() 
    print(a)
    # while num_read_so_far < total_num_samples:
    #     new_counts = counter_server.read_counter_complete() 
    #     num_new_samples = len(new_counts)
    #     # print(new_counts)
    #     if num_new_samples > 0:
    #         print(np.shape(new_counts))
            
    #         total_counts.extend(new_counts.tolist())
    #         num_read_so_far += num_new_samples
        
    new_times, new_channels = [], []# tagger_server.read_tag_stream()  
    # print(new_times)
    # cxn.qm_opx.stop_tag_stream(apd_index)
    # tool_belt.reset_cfm()
    return total_counts, new_times, new_channels

    # %% Collect the data

    # %% Wrap up

    # clean_up(cxn)


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':
    
    delay, readout_time, apd_index, laser_name, laser_power = 200, 3e7, 0, 'cobolt_515', 1
    num_reps=700
    nv_sig = []
    counts, times, new_channels = main( nv_sig, delay, readout_time, apd_index, laser_name, laser_power, num_reps )
    print('hi')
    print(counts)
    # print()
    # print(times)
    # print('')
    # print(np.asarray(times).astype(float).astype(int)/1000)
