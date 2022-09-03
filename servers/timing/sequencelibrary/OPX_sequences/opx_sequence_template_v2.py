#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:58:57 2022

Example sequence file for the OPX. This should serve as a template

@author: carterfox
"""

import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *


def qua_program(args, num_reps):
    
    readout_time = args[0]
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        
        counts_gate1 = declare(int)
        counts_gate2 = declare(int)
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_gate1 = [counts_gate1_apd_1,counts_gate1_apd_2]
        counts_gate2_apd_0 = declare(int)  # variable for number of counts
        counts_gate2_apd_1 = declare(int)
        counts_gate2 = [counts_gate2_apd_1,counts_gate2_apd_2]
        counts_st = declare_stream()  # stream for counts
        
        times_gate1_apd_0 = declare(int, size=100)  # why input a size??
        times_gate1_apd_1 = declare(int, size=100)
        times_gate2_apd_0 = declare(int, size=100)
        times_gate2_apd_1 = declare(int, size=100)
        times_gate1 = [times_gate1_apd_0,times_gate1_apd_1]
        times_gate2 = [times_gate2_apd_0,times_gate2_apd_1]
        times_st = declare_stream()
        
        
        n = declare(int)

        
        with for_(n, 0, n < num_reps, n+1):
            
            play('NV','pi',args[0]) #using args parameters...
            wait() #wait for pi pulse to end.
            
            play('laser_ON','AOM')
            for apd_ind in apd_indices:   #this is like the first gate of the apds
                measure("readout", "APD_".format(apd_ind), None, time_tagging.analog(times_gate1[apd_ind], readout_time, counts_gate1[apd_ind]))
            
            wait(25) 
            
            play('laser_ON','AOM') 
            for apd_ind in apd_indices:   #this is like the second gate of the apds
                measure("readout", "APD_".format(apd_ind), None, time_tagging.analog(times_gate2[apd_ind], readout_time, counts_gate2[apd_ind]))

            # save the sample to the count stream. sample is a list of gates, which is a list of counts from each apd
            # if there is only one gate, it will be in the same structure as read_counter_simple wants so we are good
           
            # in all sequences, these lists need to be populated with however many gates we have. In this case 2. 
            counts_apd_0 = [counts_gate1[0], counts_gate2[0]]
            counts_apd_1 = [counts_gate1[1], counts_gate2[1]]
            
            # the code below should apply to all sequences. It takes all the counts from both possible apds and saves it based on which apds are actually in use
            counts_all_apds = [counts_apd_0,counts_apd_1]
            counts_cur_apds = []
            for apd_ind in apd_indices:
                counts_cur_apds.append(counts_all_apds[apd_ind])
                save(counts_cur_apds, counts_st) 
            # save time tags too?
            
        with stream_processing():
            counts_st.save("counts")
            times_st.save("times")
        
    return seq


def get_seq(opx, config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    
    return seq, final, [period]

def get_full_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(args, num_reps)

    return seq, final, [period]
    
    