#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

simple readout sequence for the opx in qua

"""


import numpy
import utils.tool_belt as tool_belt
from utils.tool_belt import Mod_types
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from opx_configuration_file import *

def qua_program(opx, config, args, num_reps):
    
    delay, readout_time, apd_index, laser_name, laser_power = args
    
    # laser_mod_type = config["Optics"][laser_name]["mod_type"]
    # laser_pulse = 'laser_ON_{}'.format(eval(laser_mod_type).name)
    laser_pulse, laser_delay_time, laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,laser_name,laser_power)
    # print(laser_pulse,laser_amplitude)
    apd_indices =  config['apd_indices']
    
    num_apds = len(apd_indices)
    num_gates = 1
    timetag_list_size = int(15900 / num_gates / 2)    
    
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
   
    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time
        
    elif readout_time<= max_readout_time:
        num_readouts=1
        apd_readout_time = readout_time
    
    meas_delay = 100
    meas_delay_cc = meas_delay // 4
    
    delay_between_readouts_iterations = 200 #simulated - conservative estimate
    
    laser_on_time= delay + meas_delay + num_readouts*(apd_readout_time + delay_between_readouts_iterations) + 300
    laser_on_time=  meas_delay + apd_readout_time  
    laser_on_time_cc = laser_on_time // 4
    # print(laser_on_time)
    
    delay_cc = max(int(delay // 4),4)
    period = int(laser_on_time)
            
    with program() as seq:
        
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
            # start the laser a little earlier than the apds
            # play(laser_pulse*amp(laser_amplitude),laser_name,duration=laser_on_time_cc)
            # wait(meas_delay_cc,"do_apd_0_gate","do_apd_1_gate")
            
            # play("laser_ON",laser_name,duration=laser_on_time_cc) 
            
            with for_(i,0,i<num_readouts,i+1):  
                
                play(laser_pulse*amp(laser_amplitude),laser_name,duration=laser_on_time_cc)
                wait(meas_delay_cc,"do_apd_0_gate","do_apd_1_gate")
                if num_apds == 2:
                    
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(counts_gate1_apd_1, counts_st_apd_1)
                    
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    # play("laser_ON",laser_name,duration=laser_on_time_cc)  
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate1_apd_0, apd_readout_time, counts_gate1_apd))
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    
                    align("do_apd_0_gate","do_apd_1_gate")
                    
            ##clock pulse that advances piezos and ends a sample in the tagger
            # align()
            align("do_apd_0_gate","do_apd_1_gate","do_sample_clock")
            wait(25,"do_sample_clock")
            play("clock_pulse","do_sample_clock")
            
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            # counts_st_apd_0.buffer(num_readouts).buffer(num_gates).save_all("counts_apd0") 
            # counts_st_apd_1.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1")


    return seq, period, num_gates


def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = '' 
    sample_size = 'one_rep' # 'all_reps
    
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    
    readout_time = 4000
    max_readout_time = 1000# config['PhotonCollection']['qm_opx_max_readout_time']
    
    qm = qmm.open_qm(config_opx)
    simulation_duration =  12000 // 4 # clock cycle units - 4ns
    num_repeat=3
    delay = 3000
    args = [200, readout_time, 0,'laserglow_589',0.55]
    args=[800, 1000.0, 0, 'cobolt_515', None]
    seq , f, p, ng, ss = get_seq([],config, args, num_repeat)
    
    # start_t = time.time()
    compilied_program_id = qm.compile(seq)
    # t1 = time.time()
    # print(t1 - start_t)

    # program_job = qm.queue.add_compiled(compilied_program_id)
    # job = program_job.wait_for_execution()
    # print(time.time()-t1)
    
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
# 
    # print(time.time())
    # job = qm.execute(seq)
    # print(time.time())
    # job = qm.execute(seq)
    # st = time.time()
    
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="live")
    # counts_apd0, counts_apd1 = results.fetch_all() 
    
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