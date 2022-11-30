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


def qua_program(args, num_reps, x_voltage_list=[], y_voltage_list=[], z_voltage_list=[]):
    
    readout, state, laser_name, laser_power, apd_index = args
    
    state = States(state)
    sig_gen_name = config['Microwaves']['sig_gen_{}'.format(state.name)]
    uwave_delay = config['Microwaves'][sig_gen_name]['delay']
    uwave_delay_cc = uwave_delay // 4
    laser_delay = config['Optics'][laser_name]['delay']
    laser_delay_cc = laser_delay // 4
    meas_buffer = config['CommonDurations']['cw_meas_buffer']
    meas_buffer_cc = meas_buffer // 4
    transient = 0
    transient_cc = transient // 4
    readout = numpy.int64(readout)
    front_buffer = max(uwave_delay, laser_delay)
    front_buffer_cc = front_buffer // 4
    period = front_buffer + 2 * (transient + readout + meas_buffer)
    period_cc = period // 4
    # Get what we need out of the wiring dictionary
    sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
    
    num_apds = len(apd_indices)
    num_gates = 2
    timetag_list_size = int(15900 / num_gates / num_apds)
    
    desired_time_between_gates = meas_buffer + transient
    intrinsic_time_between_gates = 124 - 16  #124ns delay + 16ns because the first 16ns in the wait command here don't contribute
    time_between_gates = desired_time_between_gates - intrinsic_time_between_gates
    time_between_gates_cc = time_between_gates // 4
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_gate2_apd_0 = declare(int)  # variable for number of counts
        counts_gate2_apd_1 = declare(int)
        counts_st = declare_stream()  # stream for counts
        
        times_gate1_apd_0 = declare(int, size=timetag_list_size)  # why input a size??
        times_gate1_apd_1 = declare(int, size=timetag_list_size)
        times_gate2_apd_0 = declare(int, size=timetag_list_size)  # why input a size??
        times_gate12_apd_1 = declare(int, size=timetag_list_size)
        times_st = declare_stream()
                
        n = declare(int)
        i = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
        
            align()  
            
            ###green laser
            play(laser_ON,laser_name,duration=int(period))  
            
            ###apds
            if 0 in apd_indices:
                wait((front_buffer + transient)//4, "APD_0") # wait for the delay before starting apds
                measure("readout", "APD_0", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))
                
            if 1 in apd_indices:
                wait((front_buffer + transient)//4, "APD_1") # wait for the delay before starting apds
                measure("readout", "APD_1", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_gate1_apd_1))
                
            if num_apds == 2:  # wait for them both to finish if we are using two apds
                align("APD_0","APD_1")
            
            if 0 in apd_indices:
                wait(time_between_gates_cc, "APD_0") # wait for the delay before starting apds
                measure("readout", "APD_0", None, time_tagging.analog(times_gate2_apd_0, readout_time, counts_gate2_apd_0))
                
            if 1 in apd_indices:
                wait(time_between_gates_cc, "APD_1") # wait for the delay before starting apds
                measure("readout", "APD_1", None, time_tagging.analog(times_gate2_apd_1, readout_time, counts_gate2_apd_1))
                    
            
            ###microwaves
            wait((front_buffer - uwave_delay + transient + readout + meas_buffer + transient)//4, "sig_gen_gate_chan_name")
            play("ON", "sig_gen_gate_chan_name", duration=int(readout))  # play microwave pulse. the freq is set from outside the opx. in the opx it should just have the digital switch for the sig gen
            
            
            ###trigger piezos
            if (len(x_voltage_list) > 0):
                wait((period - 200) // 4, "x_channel")
                play("ON", "x_channel", duration=100)  
            if (len(y_voltage_list) > 0):
                wait((period - 200) // 4, "y_channel")
                play("ON", "y_channel", duration=100)  
            if (len(z_voltage_list) > 0):
                wait((period - 200) // 4, "z_channel")
                play("ON", "z_channel", duration=100)  
                
        
            ###saving
            if 0 in apd_indices:
                save(counts_gate1_apd_0, counts_st)
                save(counts_gate2_apd_0, counts_st)
                with for_(i, 0, i < counts_gate1_apd_0, i + 1):
                    save(times_gate1_apd_0[i], times_st)  # save time tags to stream
                with for_(i, 0, i < counts_gate2_apd_0, i + 1):
                    save(times_gate2_apd_0[i], times_st)
                    
            if 1 in apd_indices:
                save(counts_gate1_apd_1, counts_st)
                save(counts_gate2_apd_1, counts_st)
                with for_(i, 0, i < counts_gate1_apd_1, i + 1):
                    save(times_gate1_apd_1[i], times_st)  # save time tags to stream
                with for_(i, 0, i < counts_gate2_apd_1, i + 1):
                    save(times_gate2_apd_1[i], times_st)
                    
                
        with stream_processing():
            counts_st.buffer(num_gates).buffer(num_apds).buffer(num_reps).save_all("counts")
            times_st.save_all("times")
        
    return seq


def get_seq(config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    
    return seq, final, [period]

def get_full_seq(config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(args, num_reps)

    return seq, final, [period]
    