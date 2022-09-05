#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:41:23 2022

@author: carterfox
"""


import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import utils.tool_belt as tool_belt
from utils.tool_belt import States


def qua_program(args, num_reps):
    
    readout, state, laser_name, laser_power, apd_index = args
    
    state = States(state)
    sig_gen_name = config['Microwaves']['sig_gen_{}'.format(state.name)]
    uwave_delay = config['Microwaves'][sig_gen_name]['delay']
    laser_delay = config['Optics'][laser_name]['delay']
    meas_buffer = config['CommonDurations']['cw_meas_buffer']
    transient = 0

    readout = numpy.int64(readout)
    front_buffer = max(uwave_delay, laser_delay)
    period = front_buffer + 2 * (transient + readout + meas_buffer)

    # Get what we need out of the wiring dictionary
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)

    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        
        apd_indices_st = declare_stream()
        save(apd_indices,apd_indices_st)
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_gate2_apd_0 = declare(int)  # variable for number of counts
        counts_gate2_apd_1 = declare(int)
        counts_gate1 = [counts_gate1_apd_0,counts_gate1_apd_1]
        counts_gate2 = [counts_gate2_apd_0,counts_gate2_apd_1]
        
        counts_st = declare_stream()  # stream for counts
        
        times_gate1_apd_0 = declare(int, size=100)  # why input a size??
        times_gate1_apd_1 = declare(int, size=100)
        times = [times_gate1_apd_0,times_gate1_apd_1]
        times_gate2_apd_0 = declare(int, size=100)  # why input a size??
        times_gate2_apd_1 = declare(int, size=100)
        times_gate1 = [times_gate1_apd_0,times_gate1_apd_1]
        times_gate2 = [times_gate1_apd_0,times_gate1_apd_1]
        
        times_st = declare_stream()
        
        n = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
        
            # there must be an element for the laser_name 
            play(laser_ON,laser_name,duration=int(period))  
            #Either we make this statement turn on the laser with an indefinite pulse, 
            #or we need the sequence to populate a pulse in the configuration for the play statement to grab
            
            wait(front_buffer + transient) # wait for the delay before starting apds
    
            for apd_ind in apd_indices:   #readout during the readout time on all apds
                measure("readout", "APD_".format(apd_ind), None, time_tagging.analog(times_gate1[apd_ind], readout_time, counts_gate1[apd_ind]))
            
            wait(front_buffer - uwave_delay + transient + readout + meas_buffer + transient)
            
            play("ON", "sig_gen_gate_chan_name", duration=int(readout))  # play microwave pulse. the freq is set from outside the opx. in the opx it should just have the digital switch for the sig gen
            
            wait(uwave_delay)
            
            for apd_ind in apd_indices:   #readout during the readout time on all apds
                measure("readout", "APD_".format(apd_ind), None, time_tagging.analog(times_gate2[apd_ind], readout_time, counts_gate2[apd_ind]))
            
            wait(meas_buffer+uwave_delay)
            
            # save the sample to the count stream. sample is a list of gates, which is a list of counts from each apd
            # if there is only one gate, it will be in the same structure as read_counter_simple wants so we are good
           
            # in all sequences, these lists need to be populated with however many gates we have. In this case 2. 
            counts_apd_0 = [counts_gate1[0],counts_gate2[0]]
            counts_apd_1 = [counts_gate1[1],counts_gate2[1]]
            times_apd_0 = [times_gate1[0],times_gate2[0]]
            times_apd_1 = [times_gate2[1],times_gate2[1]]
            
            # the code below should apply to all sequences. It takes all the counts from both possible apds and saves it based on which apds are actually in use
            counts_all_apds = [counts_apd_0,counts_apd_1]
            counts_cur_apds = []
            times_all_apds = [times_apd_0,times_apd_1]
            times_cur_apds = []
            for apd_ind in apd_indices:
                counts_cur_apds.append(counts_all_apds[apd_ind])
                times_cur_apds.append(times_all_apds[apd_ind])
            save(counts_cur_apds, counts_st) 
            save(times_cur_apds, times_st) 

        with stream_processing():
            counts_st.save("counts")
            times_st.save("times")
            apd_indices_st.save("apd_indices")
        
    return seq


def get_seq(opx, config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    
    return seq, final, [period]

def get_full_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(args, num_reps)

    return seq, final, [period]
    