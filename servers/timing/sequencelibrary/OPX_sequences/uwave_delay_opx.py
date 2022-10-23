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
from utils.tool_belt import States

def qua_program(opx, config, args, num_reps):
    
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 2
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / 2)    
    num_readouts = 1

    durations = args[0:5]
    durations = [numpy.int64(el) for el in durations]
    tau, max_tau, readout_time, pi_pulse, polarization = durations
    
    state, apd_index, laser_name, laser_power = args[5:9]
    state = States(state)
    sig_gen = config['Microwaves']['sig_gen_{}'.format(state.name)]
    laser_pulse = 'laser_ON_{}'.format(tool_belt.get_mod_type(laser_name))
    
    wait_time = config['CommonDurations']['uwave_buffer']
    laser_delay_time = config['Optics'][laser_name]['delay']
    
    common_delay = max(laser_delay_time, pi_pulse + abs(tau)) + 24
    common_delay_cc = int(common_delay // 4 )
    uwave_wait_cc = int( (common_delay_cc*4 - (pi_pulse - tau)) // 4 )
    laser_wait_cc = int( (common_delay_cc*4 - laser_delay_time) // 4 )
    # Include a buffer on the front end so that we can delay channels that
    # should start off the sequence HIGH
    front_buffer = max(max_tau+pi_pulse, laser_delay_time)
    
    front_buffer_cc = int(front_buffer // 4)
    polarization_cc = int(polarization // 4)
    laser_delay_time_cc = int(laser_delay_time // 4)
    wait_time_cc = int(wait_time //4)
    readout_time_cc = int(readout_time // 4)
    pi_pulse_cc = int(pi_pulse // 4)
    period = front_buffer + wait_time*2 + polarization*2 
    
    # print(pi_pulse)
    # print(common_delay - laser_delay_time)
    # print(common_delay - pi_pulse - tau)
    # print(common_delay)
    
    with program() as seq:
        
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
            
            wait(front_buffer_cc)
                           
            play(laser_pulse,laser_name,duration=polarization_cc) 
            
            if num_apds == 2:
                wait(laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate" )
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_gate1_apd_1))
                save(counts_gate1_apd_0, counts_st_apd_0)
                save(counts_gate1_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate1_apd_0, readout_time, counts_gate1_apd))
                save(counts_gate1_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
            align()
            wait(wait_time_cc)
            align()
            
            wait(laser_wait_cc, laser_name)
            play(laser_pulse,laser_name,duration=polarization_cc)
            
            wait(uwave_wait_cc,sig_gen)
            play("uwave_ON",sig_gen, duration=pi_pulse_cc)
            
            
            if num_apds == 2:
                wait(common_delay_cc,"do_apd_0_gate","do_apd_1_gate" )
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, readout_time, counts_gate2_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate2_apd_1, readout_time, counts_gate2_apd_1))
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(counts_gate2_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(common_delay_cc,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate2_apd_0, readout_time, counts_gate2_apd))
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
            align()
            wait(wait_time_cc)
        
        play("clock_pulse","do_sample_clock") # clock pulse after all the reps so the tagger sees all reps as one sample
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            # counts_st_apd_0.buffer(num_readouts).buffer(total_num_gates).save_all("counts_apd0") 
            # counts_st_apd_1.buffer(num_readouts).buffer(total_num_gates).save_all("counts_apd1")
            
    return seq, period, num_gates



def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = ''
    sample_size = 'all_reps'
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    qm = qmm.open_qm(config_opx)
    
    simulation_duration =  15000 // 4 # clock cycle units - 4ns
    
    num_repeat=1#4e4

    args = [-100, 100, 350, 80, 1000.0, 1, 0, 'cobolt_515', None]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)

    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # plt.show()
# 
    # job = qm.execute(seq)

    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")
    
    # counts_apd0, counts_apd1 = results.fetch_all() 
    
    # # print('')
    # print(counts_apd0.tolist())
    # # print('')
    # print(counts_apd1.tolist())