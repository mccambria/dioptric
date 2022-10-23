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
    
    delay, readout_time, apd_index, laser_name, laser_power = args
    
    laser_pulse = 'laser_ON_{}'.format(tool_belt.get_mod_type(laser_name))
    # opx_wiring = config['Wiring']['QmOpx']
    # apd_indices = config['apd_indices']
    
    apd_indices =  config['apd_indices']
    # apd_indices = opx#.apd_indices
    # apd_indices = opx.apd_indices
    
    num_apds = len(apd_indices)
    num_gates = 1
    timetag_list_size = int(15900 / num_gates / 2)
    readout_time = numpy.int64(readout_time)
    
    delay_between_readouts_iterations = 244 
    
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
   
    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time
        
    elif readout_time<= max_readout_time:
        num_readouts=1
        apd_readout_time = readout_time
    
    apd_readout_time_cc = int(apd_readout_time // 4)
    meas_delay = 100
    meas_delay_cc = meas_delay // 4
    
    # laser_on_time = delay + (readout_time + delay_between_readouts_iterations*num_readouts ) * num_gates
    # laser_on_time = delay + apd_readout_time
    laser_on_time = meas_delay + apd_readout_time
    laser_on_time_cc = laser_on_time // 4
    
    period = numpy.int64(delay + laser_on_time + 300)
    period_cc = int(period//4)
    
    delay = numpy.int64(delay)
    delay_cc = int(delay // 4)
    clock_delay_cc = int((period-200)//4)
        
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_gate1_apd = declare(int)
        times_gate1_apd_0 = declare(int,size=timetag_list_size)
        times_gate1_apd_1 = declare(int,size=timetag_list_size)
        times_gate1_apd = declare(int,size=timetag_list_size)
        times_st_apd_0 = declare_stream()
        times_st_apd_1 = declare_stream()
        times_st_apd = declare_stream()
        empty_time_stream = declare_stream()
        counts_st_apd_0 = declare_stream()
        counts_st_apd_1 = declare_stream()
        counts_st_apd = declare_stream()
        empty_stream = declare_stream()
        rep_readout_delay_st = declare_stream()
        
        save(0,empty_time_stream)
        save(0,times_st_apd)
        save(0,times_st_apd_0)
        save(0,times_st_apd_1)
        
        n = declare(int)
        i = declare(int)
        j = declare(int)
        k = declare(int)
    
        with for_(n, 0, n < num_reps, n + 1):
            align()  
            wait(delay_cc)
            align()  
            
            ###green laser
            # play("laser_ON",laser_name,duration=period_cc)  
            
            
            play(laser_pulse,laser_name,duration=meas_delay_cc) 
            wait(meas_delay_cc, "do_apd_0_gate","do_apd_1_gate")
            
            with for_(i,0,i<num_readouts,i+1):  
                
                if num_apds == 2:
                # with if_(num_apds==2):
                    play(laser_pulse,laser_name,duration=laser_on_time_cc) 
                    
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    
                    # assign(counts_gate1_apd_0,2)
                    # assign(counts_gate1_apd_1,4)

                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(counts_gate1_apd_1, counts_st_apd_1)
                    
                    with for_(j, 0, j < counts_gate1_apd_0, j + 1):
                        save(times_gate1_apd_0[j], times_st_apd_0) 
                        
                    with for_(k, 0, k < counts_gate1_apd_1, k + 1):
                        save(times_gate1_apd_1[k], times_st_apd_1)
                    
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    play(laser_pulse,laser_name,duration=laser_on_time_cc)  
                    
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd, apd_readout_time, counts_gate1_apd))
                    save(counts_gate1_apd, counts_st_apd)
                    save(0,empty_stream)
                    
                    with for_(k, 0, k < counts_gate1_apd, k + 1):
                        save(times_gate1_apd[k], times_st_apd) 
                        save(0, empty_time_stream) 
                        
                    align("do_apd_{}_gate".format(apd_indices[0]))
                    
                    
            
            ##clock pulse
            align()
            wait(25,"do_sample_clock")
            play("clock_pulse","do_sample_clock") #clock pulse that triggers piezos and advances sample in the tagger after each rep
                        
                        
     
        if num_apds == 2:
            with stream_processing():
                counts_st_apd_0.buffer(num_readouts).buffer(num_gates).save_all("counts_apd0") 
                counts_st_apd_1.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1")
                times_st_apd_0.save_all("times_apd0")
                times_st_apd_1.save_all("times_apd1")
                
        if num_apds == 1:    
            with stream_processing():
                counts_st_apd.buffer(num_readouts).buffer(num_gates).save_all("counts_apd0") 
                empty_stream.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1") 
                times_st_apd.save_all("times_apd0")
                empty_time_stream.save_all("times_apd1")
            
    return seq, period



def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period = qua_program(opx,config, args, num_repeat)
    final = ''
    # period = 0
    return seq, final, [period]
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    readout_time = 1000000
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
    
    qm = qmm.open_qm(config_opx)
    simulation_duration =  80000 // 4 # clock cycle units - 4ns
    num_repeat=10
    delay = 1000
    args = [delay, readout_time,0, 'cobolt_515',1]
    seq , f, p = get_seq([],config, args, num_repeat)
    
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # # plt.xlim(100,12000)
# 
    # job = qm.execute(seq)
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1"], mode="wait_for_all")
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")
    # counts_apd0, counts_apd1 = results.fetch_all() 
    
    
    # while num_count_samples_read < num_repeat:
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1"], mode="wait_for_all")
    # counts_apd0, counts_apd1, times_apd0, times_apd1 = results.fetch_all() 
    # print((counts_apd0))
    # print((counts_apd1))
    # print(times_apd0)
    # print(times_apd1)
    # print(counts_apd0.tolist())
    
    
    
    
    

