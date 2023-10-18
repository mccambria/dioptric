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
    
    ### get inputted parameters
    delay, readout_time, laser_name, laser_power = args
    
    ### get laser information
    laser_pulse, laser_delay_time, laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,laser_name,laser_power)

    ### specify number of gates and determine length of timetag streams to use 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 1
    timetag_list_size = int(15900 / num_gates / num_apds)    
    
    ### determine if the readout time is longer than the max opx readout time and therefore we need to loop over smaller readouts. 
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
   
    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time
        
    elif readout_time<= max_readout_time:
        num_readouts=1
        apd_readout_time = readout_time
    
    
    ### determine necessary times and delays and put them in clock cycles
    meas_delay = 100
    meas_delay_cc = meas_delay // 4
    
    delay_between_readouts_iterations = 200 #simulated - conservative estimate
    
    # laser_on_time= delay + meas_delay + num_readouts*(apd_readout_time + delay_between_readouts_iterations) + 300
    laser_on_time=  meas_delay + apd_readout_time  
    laser_on_time_cc = laser_on_time // 4
    laser_delay_time_cc = int(laser_delay_time/4)
    delay1_cc = int(meas_delay_cc + laser_delay_time_cc)
    delay_cc = max(int(delay // 4),4)
    period_cc = delay_cc + num_readouts*(laser_on_time_cc) + 25
    period = int(period_cc*4)
                
    with program() as seq:
        
        ### define qua variables and streams
        counts_gate1_apd_0 = declare(int)  
        counts_gate1_apd_1 = declare(int)
        times_gate1_apd_0 = declare(int,size=timetag_list_size)
        times_gate1_apd_1 = declare(int,size=timetag_list_size)
        counts_st_apd_0 = declare_stream()
        counts_st_apd_1 = declare_stream()        
        
        n = declare(int)
        i = declare(int)
        j = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()  
            wait(delay_cc)            
            align()  
            
            with for_(i,0,i<num_readouts,i+1):  
                
                play(laser_pulse*amp(laser_amplitude),laser_name,duration=laser_on_time_cc)
                wait(delay1_cc,"do_apd_0_gate","do_apd_1_gate")
                
                if num_apds == 2:
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(counts_gate1_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
            align("do_apd_0_gate","do_apd_1_gate","do_sample_clock")
            wait(25,"do_sample_clock")
            play("clock_pulse","do_sample_clock")
            
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")


    return seq, period, num_gates


def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = '' 
    ### specify what one 'sample' means for the data processing. 
    sample_size = 'one_rep' # 'all_reps
    
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    
    qm = qmm.open_qm(config_opx)
    simulation_duration =  52000 // 4 # clock cycle units - 4ns
    num_repeat=2
    # delay = 3000
    args=[5e2, 10e3,  'cobolt_515', 1]
    seq , f, p, ng, ss = get_seq([],config, args, num_repeat)
    
    # start_t = time.time()
    # compilied_program_id = qm.compile(seq)
    # t1 = time.time()
    # print(t1 - start_t)

    # program_job = qm.queue.add_compiled(compilied_program_id)
    # job = program_job.wait_for_execution()
    # print(time.time()-t1)
    plt.figure(figsize=(10,6))
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
# 
    # print(time.time())
    # job = qm.execute(seq)
    # print(time.time())
    # job = qm.execute(seq)
    # st = time.time()
    
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")
    # counts_apd0, counts_apd1 = results.fetch_all() 
    # print(np.sum(counts_apd0,1))
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