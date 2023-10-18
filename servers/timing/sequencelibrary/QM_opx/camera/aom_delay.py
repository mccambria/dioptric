#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

Sequence for determining a laser delay. 
The name 'aom_delay' is a bit outdated. It is just used because that is what the file for the pulse streamer version is called, but this can be used for any laser. 

"""


import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from opx_configuration_file import *

def qua_program(opx, config, args, num_reps):
    
    ### get inputed parameters
    durations = args[0:3]
    durations = [numpy.int64(el) for el in durations]
    tau, max_tau, readout_time = durations
    laser_name, laser_power = args[3:5]
    
    ### get laser info
    laser_pulse, laser_delay_time, laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,laser_name,laser_power)
     
    ### get apd and gate information 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    
    ### specify number of gates and determine length of timetag streams to use 
    num_gates = 2
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / num_apds)    
    
    ### specify number of readouts to buffer the count stream by so the counter function on the server knows the combine iterative readouts
    ### it's just 1 here because the readout will always be much shorter than the max readout time the opx can do (~5ms)
    num_readouts = 1

    ### set up necessary times in terms of clock cycles
    illumination = 10*readout_time
    half_illumination = illumination // 2
    inter_time = max(10**3, max_tau) + 100
    inter_time_cc = int(inter_time // 4)
    illumination_cc = int(illumination // 4 )
    half_illumination_cc = int(half_illumination // 4 )
    readout_time_cc = int(readout_time // 4)
    tau_cc = int(tau // 4)
    back_buffer_cc = inter_time_cc

    period = illumination_cc + inter_time_cc + tau_cc + illumination + back_buffer_cc    
    
    with program() as seq:
        
        ### define qua variables and streams 
        counts_gate1_apd_0 = declare(int)  
        counts_gate1_apd_1 = declare(int)
        times_gate1_apd_0 = declare(int,size=timetag_list_size)
        times_gate1_apd_1 = declare(int,size=timetag_list_size)
        counts_gate2_apd_0 = declare(int)  
        counts_gate2_apd_1 = declare(int)
        times_gate2_apd_0 = declare(int,size=timetag_list_size)
        times_gate2_apd_1 = declare(int,size=timetag_list_size)
        
        counts_st_apd_0 = declare_stream()
        counts_st_apd_1 = declare_stream()     
        

        n = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()    
                           
            play(laser_pulse*amp(laser_amplitude),laser_name,duration=illumination_cc) 
            
            if num_apds == 2:
                wait(half_illumination_cc ,"do_apd_0_gate","do_apd_1_gate" )
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_gate1_apd_1))
                save(counts_gate1_apd_0, counts_st_apd_0)
                save(counts_gate1_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(half_illumination_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))
                save(counts_gate1_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
            align()
            wait(inter_time_cc)
            
            play(laser_pulse*amp(laser_amplitude),laser_name,duration=illumination_cc) 
                            
            if num_apds == 2:
                wait(tau_cc + illumination_cc - readout_time_cc ,"do_apd_0_gate","do_apd_1_gate")
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, readout_time, counts_gate2_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate2_apd_1, readout_time, counts_gate2_apd_1))
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(counts_gate2_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(tau_cc + illumination_cc - readout_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate2_apd_0, readout_time, counts_gate2_apd_0))
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
            align()
            wait(back_buffer_cc)
        
        play("clock_pulse","do_sample_clock") 
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            
    return seq, period, num_gates



def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = ''
    ### specify what one 'sample' means for the data processing. 
    sample_size = 'all_reps'
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    tool_belt.set_delays_to_sixteen(config)
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    qm = qmm.open_qm(config_opx)
    
    simulation_duration =  15000 // 4 # clock cycle units - 4ns
    
    num_repeat=1

    args = [100, 100.0, 500.0,'laserglow_589', 1]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)

    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # plt.show()
