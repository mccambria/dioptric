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


def qua_program(args, num_reps, x_voltage_list, y_voltage_list, z_voltage_list):
    
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
        
        counts_apd_0 = declare(int)  # variable for number of counts
        counts_apd_1 = declare(int)
        counts_st = declare_stream()  # stream for counts
        
        times_apd_0 = declare(int, size=100)  # why input a size??
        times_apd_1 = declare(int, size=100)
        times_st = declare_stream()
                
        n = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
        
            # there must be an element for the laser_name 
            ###piezos
            with if_(len(x_voltage_list) > 0):
                play("cw"*x_voltage_list[n],"x_channel")
            with if_(len(y_voltage_list) > 0):
                play("cw"*y_voltage_list[n],"y_channel")
            with if_(len(z_voltage_list) > 0):
                play("cw"*z_voltage_list[n],"z_channel")
                
                
            ###green laser
            play(laser_ON,laser_name,duration=int(period))  
            #Either we make this statement turn on the laser with an indefinite pulse, 
            #or we need the sequence to populate a pulse in the configuration for the play statement to grab
            
            ###apds
            with for_each_(apd_ind, apd_indices): 
                
                with if_(apd_ind = 0):
                    wait(front_buffer + transient, "APD_0") # wait for the delay before starting apds
                    measure("readout", "APD_0", None, time_tagging.analog(times_apd_0, readout_time, counts_apd_0))
                    wait(uwave_delay, "APD_0") # wait for the delay before starting apds
                    measure("readout", "APD_0", None, time_tagging.analog(times_apd_0, readout_time, counts_apd_0))
                with if_(apd_ind = 1):
                    wait(front_buffer + transient, "APD_1") # wait for the delay before starting apds
                    measure("readout", "APD_1", None, time_tagging.analog(times_apd_1, readout_time, counts_apd_1))
                    wait(uwave_delay, "APD_1") # wait for the delay before starting apds
                    measure("readout", "APD_1", None, time_tagging.analog(times_apd_1, readout_time, counts_apd_1))
                    
                    
            ###microwaves
            wait(front_buffer - uwave_delay + transient + readout + meas_buffer + transient, "sig_gen_gate_chan_name")
            play("ON", "sig_gen_gate_chan_name", duration=int(readout))  # play microwave pulse. the freq is set from outside the opx. in the opx it should just have the digital switch for the sig gen
            
            
            # save the sample to the count stream. sample is a list of gates, which is a list of counts from each apd
            # if there is only one gate, it will be in the same structure as read_counter_simple wants so we are good
            
            # in all sequences, these lists need to be populated with however many gates we have. In this case 2. 
            with for_each_(apd_ind, apd_indices):
                
                with if_(apd_ind = 0):
                    save(counts_apd_0, counts_st)
                    save(times_apd_0, times_st)
                with if_(apd_ind = 1):
                    save(counts_apd_1, counts_st)
                    save(times_apd_1, times_st)
                
        with stream_processing():
            counts_st.buffer(num_gates).buffer(num_apds).save_all("counts")
            times_st.buffer(num_gates).save_all("times")
        
    return seq


def get_seq(opx, config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    
    return seq, final, [period]

def get_full_seq(opx, config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(args, num_reps)

    return seq, final, [period]
    