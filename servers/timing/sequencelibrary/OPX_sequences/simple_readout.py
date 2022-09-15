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

def qua_program(args, num_reps, x_voltage_list, y_voltage_list, z_voltage_list):
    
    delay, readout_time, apd_index, laser_name, laser_power = args

    delay = numpy.int64(delay)
    readout_time = numpy.int64(readout_time)

    period = numpy.int64(delay + readout_time + 300)
    
    num_gates = 1
    num_apds = len(apd_indices)
    first_apd = apd_indices[0]
    last_apd = apd_indices[-1]
  
    
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
    timetag_list_size = 10000000
    num_apds = len(apd_indices)
    num_gates = 1
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_st = declare_stream()  # stream for counts
        
        
        times_gate1_apd_0 = declare(int, size=timetag_list_size)  # why input a size??
        times_gate1_apd_1 = declare(int, size=timetag_list_size)
        times_st = declare_stream()
                
        n = declare(int)
        apd_ind = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
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
                    wait(delay, "APD_0") # wait for the delay before starting apds
                    measure("readout", "APD_0", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))
                with if_(apd_ind = 1):
                    wait(delay, "APD_1") # wait for the delay before starting apds
                    measure("readout", "APD_1", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_gate1_apd_1))
            
            
            # save the sample to the count stream. sample is a list of gates, which is a list of counts from each apd
            # if there is only one gate, it will be in the same structure as read_counter_simple wants so we are good
           
            # in all sequences, these lists need to be populated with however many gates we have. In this case 2. 
            with for_each_(apd_ind, apd_indices): 
                
                with if_(apd_ind = 0):
                    save(counts_gate1_apd_0, counts_st)
                    save(times_gate1_apd_0, times_st)
                with if_(apd_ind = 1):
                    save(counts_gate1_apd_1, counts_st)
                    save(times_gate1_apd_1, times_st)
                
        with stream_processing():
            counts_st.buffer(num_gates).buffer(num_apds).save_all("counts")
            times_st.buffer(timetag_list_size).buffer(num_gates).buffer(num_apds).save_all("times")
        
    return seq


def get_seq(opx, config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    
    return seq, final, [period]

def get_full_seq(opx, config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(args, num_reps, x_voltage_list,y_voltage_list,z_voltage_list)

    return seq, final, [period]
    

if __name__ == '__main__':
    
    print('hi')