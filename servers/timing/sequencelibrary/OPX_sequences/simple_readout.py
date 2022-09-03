#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

simple readout sequence for the opx in qua

"""


import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *

def qua_program(args, num_reps):
    
    delay, readout_time, apd_index, laser_name, laser_power = args

    delay = numpy.int64(delay)
    readout_time = numpy.int64(readout_time)

    period = numpy.int64(delay + readout_time + 300)
  
    
    """ from pulse streamer version. just to look at for now
    train = [(period-200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(pulser_do_daq_clock, train)

    train = [(delay, LOW), (readout_time, HIGH), (300, LOW)]
    seq.setDigital(pulser_do_daq_gate, train)

    train = [(period, HIGH)]
    #tool_belt.process_laser_seq(pulse_streamer, seq, config, laser_name, laser_power, train)  
       ### only applies to analog because we won't use feedthrough.
       ### need to figure out what we need to do with this tool_belt function for analog controls of lasers on the opx

    """
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        
        counts_gate1 = declare(int)
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_gate1 = [counts_gate1_apd_0,counts_gate1_apd_1]
        
        counts_st = declare_stream()  # stream for counts
        
        times_gate1_apd_0 = declare(int, size=100)  # why input a size??
        times_gate1_apd_1 = declare(int, size=100)
        times = [times_gate1_apd_0,times_gate1_apd_1]
        
        n = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
        
            # there must be an element for the laser_name 
            play(laser_ON,laser_name,duration=int(readout_time))  
            #Either we make this statement turn on the laser with an indefinite pulse, 
            #or we need the sequence to populate a pulse in the configuration for the play statement to grab
            
            wait(delay) # wait for the delay before starting apds
    
            for apd_ind in apd_indices:   #readout during the readout time on all apds
                measure("readout", "APD_".format(apd_ind), None, time_tagging.analog(times[apd_ind], readout_time, counts_gate1[apd_ind]))
                    
            # save the sample to the count stream. sample is a list of gates, which is a list of counts from each apd
            # if there is only one gate, it will be in the same structure as read_counter_simple wants so we are good
           
            # in all sequences, these lists need to be populated with however many gates we have. In this case 2. 
            counts_apd_0 = [counts_gate1[0]]
            counts_apd_1 = [counts_gate1[1]]
            
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
    