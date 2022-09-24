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
from opx_configuration_file import *

def qua_program(config, args, num_reps, x_voltage_list=[], y_voltage_list=[], z_voltage_list=[]):
    
    delay, readout_time, apd_index, laser_name, laser_power = args
    
    # opx_wiring = config['Wiring']['QmOpx']
    
    # apd_indices = config['']
    apd_indices = [0,1]
    

    delay = numpy.int64(delay)
    readout_time = numpy.int64(readout_time)

   
    num_gates = 1
    num_apds = len(apd_indices)
    timetag_list_size = int(15900 / num_gates / num_apds)
    
    
    max_readout_time = 1000000
   
    
    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time
        laser_on_time = readout_time + 200*num_readouts
        
    elif readout_time<= max_readout_time:
        num_readouts=1
        apd_readout_time = readout_time
        laser_on_time = readout_time + delay
    
    period = numpy.int64(delay + laser_on_time + 300)
    period_cc = int(period//4)
    
    delay_cc = int(delay // 4)
    clock_delay_cc = int((period-200)//4)
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        if num_apds == 2:
            counts_gate1_apd_0 = declare(int)  # variable for number of counts
            counts_gate1_apd_1 = declare(int)
            times_gate1_apd_0 = declare(int,size=timetag_list_size)
            times_gate1_apd_1 = declare(int,size=timetag_list_size)
            counts_cur0 = declare(int)
            counts_cur1 = declare(int)
            
        if num_apds == 1:
            counts_gate1_apd = declare(int) 
            times_gate1_apd = declare(int,size=timetag_list_size)
            counts_cur = declare(int)
        
        counts_st = declare_stream()  # stream for counts
                
        n = declare(int)
        i = declare(int)
            
        
        with for_(n, 0, n < num_reps, n + 1):
            
            # align()  
            
            ###green laser
            # wait(laser_delay, laser_name)
            play("laser_ON",laser_name,duration=period_cc)  
            
            ###apds
            if num_apds == 2:
                wait(delay_cc, "APD_0","APD_1")
            if num_apds == 1:
                wait(delay_cc,"APD_{}".format(apd_indices[0]))
        
            with for_(i,0,i<num_readouts,i+1):  
                
                if num_apds == 2:
                    measure("readout", "APD_0", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_cur0))
                    assign(counts_gate1_apd_0,counts_cur0+counts_gate1_apd_0)
                    measure("readout", "APD_1", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_cur1))
                    assign(counts_gate1_apd_1,counts_cur1+counts_gate1_apd_1)
                    
                if num_apds == 1:
                    measure("readout", "APD_{}".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd, apd_readout_time, counts_cur))
                    assign(counts_gate1_apd,counts_cur+counts_gate1_apd)
            
                # if num_apds == 2:  # wait for them both to finish if we are using two apds
                #     align("APD_0","APD_1")
            # save the sample to the count stream. sample is a list of gates, which is a list of counts from each apd
            # if there is only one gate, it will be in the same structure as read_counter_simple wants so we are good
           
            ###trigger piezos
            wait(clock_delay_cc,"clock")
            play("clock_pulse","clock")
            align()
                
            
            ###saving
            if num_apds == 2:
                save(counts_gate1_apd_0, counts_st)
                save(counts_gate1_apd_1, counts_st)
                        
            if num_apds == 1:
                save(counts_gate1_apd, counts_st)
                        
            # wait(500)
            
        with stream_processing():
            counts_st.buffer(num_gates).buffer(num_apds).buffer(num_reps).save_all("counts")
        
    return seq


def get_seq(config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    final =''
    period = 0
    return seq, final, [period]

def get_full_seq(config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    final = ''
    period = 0
    return seq, final, [period]
    

if __name__ == '__main__':

        print('hi')
        qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
        readout_time = 1000
        qm = qmm.open_qm(config_opx)
        simulation_duration =  120000 // 4 # clock cycle units - 4ns
        x_voltage_list,y_voltage_list,z_voltage_list = [],[],[]
        num_repeat=5
        delay = 20000
        args = [delay, readout_time, 0,'green_laser_do',1]
        config = []
        seq , f, p = get_full_seq(config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
        
        job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
        job_sim.get_simulated_samples().con1.plot()
        # plt.xlim(100,12000)