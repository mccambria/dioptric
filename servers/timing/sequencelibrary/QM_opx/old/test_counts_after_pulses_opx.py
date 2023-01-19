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

def qua_program(opx, config, args, num_reps):
    
    (extra_delay, init_laser_on_time, init_laser_name, init_laser_power,
     readout_laser_on_time, readout_laser_name, readout_laser_power) = args
    
    readout_laser_pulse, readout_laser_delay_time, readout_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,readout_laser_name,readout_laser_power)
    init_laser_pulse, init_laser_delay_time, init_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,init_laser_name,init_laser_power)
    
    delay1 = max((init_laser_delay_time - readout_laser_delay_time),16)
    delay1_cc = int(delay1/4)
    num_gates = 1
    period = 0
    timetag_list_size = int(15900 / num_gates / 2)    
    readout_laser_delay_time_cc = int(readout_laser_delay_time/4)
    readout_laser_on_time_cc = int(readout_laser_on_time/4)
    init_laser_on_time_cc = int(init_laser_on_time/4)
    extra_delay_cc = int(extra_delay/4)
            
    with program() as seq:
        
        counts_gate1_apd_0 = declare(int)  
        times_gate1_apd_0 = declare(int,size=timetag_list_size)
        counts_st_apd_0 = declare_stream()
        
        n = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()  
            
            
            play(init_laser_pulse*amp(init_laser_amplitude),init_laser_name,duration=init_laser_on_time_cc) 
            align()
            wait(delay1_cc)
            wait(extra_delay_cc)
            align()    
            play(readout_laser_pulse*amp(readout_laser_amplitude),readout_laser_name,duration=readout_laser_on_time_cc)
            wait(readout_laser_delay_time_cc,"do_apd_0_gate")
            measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, readout_laser_on_time, counts_gate1_apd_0))
            save(counts_gate1_apd_0, counts_st_apd_0)
            
            wait(5000)
            align()
                
        with stream_processing():
            counts_st_apd_0.save_all("counts_apd0") 
            # counts_st_apd_0.buffer(num_readouts).buffer(num_gates).save_all("counts_apd0") 
            # counts_st_apd_1.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1")


    return seq, period, num_gates


def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = '' 
    sample_size = 'all_reps' # 'all_reps
    
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    
    # (extra_delay, init_laser_on_time, init_laser_name, init_laser_power,
    #  readout_laser_on_time, readout_laser_name, readout_laser_power) = args
    
    qm = qmm.open_qm(config_opx)
    simulation_duration =  32000 // 4 # clock cycle units - 4ns
    
    num_repeat=1000
    extra_delay = 5000
    
    init_laser_name = 'laserglow_589'
    init_laser_on_time = 2000000
    init_laser_power = 0
    
    readout_laser_name = 'laserglow_589'
    readout_laser_on_time = 5e6
    readout_laser_power = 0.0
    args = [extra_delay, init_laser_on_time, init_laser_name, init_laser_power,
      readout_laser_on_time, readout_laser_name, readout_laser_power]
    seq , f, p, ng, ss = get_seq([],config, args, num_repeat)
    
    # job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    # job_sim.get_simulated_samples().con1.plot()
    # plt.show()
# 
    job = qm.execute(seq)
    
    results = fetching_tool(job, data_list = ["counts_apd0"], mode="wait_for_all")
    counts_apd0 = results.fetch_all() 
    print(np.average(counts_apd0))
    
    # print(time.time() - st)
    
    # print('')
    # print(np.shape(counts_apd0.tolist()))
    # # print('')
    # print(np.shape(counts_apd1.tolist()))
    # time.sleep(2)
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="live")
    # counts_apd0, counts_apd1 = results.fetch_all() 
    
    # # print('')
    # print(np.shape(counts_apd0.tolist()))
    # # print('')
    # print(np.shape(counts_apd1.tolist()))
    # print(counts_apd0.tolist())