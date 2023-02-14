#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

simple readout sequence with three pulses

"""


import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from opx_configuration_file import *

def qua_program(opx, config, args, num_reps):
    
    ### get inputted parameters
    first_init_pulse_time, init_pulse_time, readout_time, first_init_laser_key, init_laser_key, readout_laser_key,\
      first_init_laser_power,init_laser_power, read_laser_power, readout_on_pulse_ind  = args
    
    ### get laser information
    first_init_laser_pulse, first_init_laser_delay_time, first_init_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,first_init_laser_key,first_init_laser_power)
    init_laser_pulse, init_laser_delay_time, init_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,init_laser_key,init_laser_power)
    readout_laser_pulse, readout_laser_delay_time, readout_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,readout_laser_key,read_laser_power)
    
    ### get necessary delays
    delay1_cc = int( (max(first_init_laser_delay_time - init_laser_delay_time,20))//4 )
    delay2_cc = int( (max(init_laser_delay_time - readout_laser_delay_time,20))//4 )
    delay3_cc = int( (max(readout_laser_delay_time - first_init_laser_delay_time,20))//4 )
    intra_pulse_delay = config['CommonDurations']['scc_ion_readout_buffer']
    
    ### specify number of gates and determine length of timetag streams to use 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 1
    timetag_list_size = int(15900 / num_gates / 2) 
    
    positioning = config['Positioning']
    if 'xy_small_response_delay' in positioning:
        pos_move_time = positioning['xy_small_response_delay']
    else:
        pos_move_time = positioning['xy_delay']
    
    ### determine if the readout time is longer than the max opx readout time and therefore we need to loop over smaller readouts. 
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
    delay_between_readouts_iterations = 200 #simulated - conservative estimate
    
    # for now lets assume we use two different lasers.
    
    if readout_on_pulse_ind == 2:
        
        if readout_time > max_readout_time:
            num_readouts = int(readout_time / max_readout_time)
            apd_readout_time = max_readout_time
            
        elif readout_time <= max_readout_time:
            num_readouts=1
            apd_readout_time = readout_time

        init_laser_on_time = init_pulse_time
        readout_laser_on_time = num_readouts*(apd_readout_time) + (num_readouts-1)*(delay_between_readouts_iterations)
        # print(readout_laser_on_time,apd_readout_time)
        
    elif readout_on_pulse_ind == 1:
        
        if init_pulse_time > max_readout_time:
            num_readouts = int(init_pulse_time / max_readout_time)
            apd_readout_time = max_readout_time
            
        elif init_pulse_time <= max_readout_time:
            num_readouts=1
            apd_readout_time = init_pulse_time

        readout_laser_on_time = readout_time
        init_laser_on_time = num_readouts*(apd_readout_time) + (num_readouts-1)*(delay_between_readouts_iterations)
        
    first_init_laser_on_time = first_init_pulse_time
    period = pos_move_time + first_init_laser_on_time + init_laser_on_time + intra_pulse_delay + readout_laser_on_time + 300 + 50000
    
    with program() as seq:
        
        ### define qua variables and streams
        counts_gate1_apd_0 = declare(int)  
        counts_gate1_apd_1 = declare(int)
        times_gate1_apd_0 = declare(int,size=timetag_list_size)
        times_gate1_apd_1 = declare(int,size=timetag_list_size)
        counts_st_apd_0 = declare_stream()
        counts_st_apd_1 = declare_stream()  
        
        times_st_apd_0 = declare_stream()
        times_st_apd_1 = declare_stream()
        
        save(0,times_st_apd_0)
        save(0,times_st_apd_1)
        
        n = declare(int)
        i = declare(int)
        j = declare(int)
        k = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            wait(pos_move_time//4)   
            play(first_init_laser_pulse*amp(first_init_laser_amplitude),first_init_laser_key,duration=first_init_laser_on_time//4) 
        
            align()
            wait(delay1_cc)
            wait(intra_pulse_delay//4)
                
            if readout_on_pulse_ind == 2:
                
                play(init_laser_pulse*amp(init_laser_amplitude),init_laser_key,duration=init_laser_on_time//4) 
            
            elif readout_on_pulse_ind == 1:
                
                with for_(i,0,i<num_readouts,i+1):                 
                    
                    play(init_laser_pulse*amp(init_laser_amplitude),init_laser_key,duration=apd_readout_time//4) 
                    
                    wait(init_laser_delay_time//4,"do_apd_0_gate","do_apd_1_gate")
                                    
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
                        
                    with for_(j, 0, j < counts_gate1_apd_0, j + 1):
                        save(times_gate1_apd_0[j], times_st_apd_0) 
                        
                    with for_(k, 0, k < counts_gate1_apd_1, k + 1):
                        save(times_gate1_apd_1[k], times_st_apd_1)
                        
            
            align()
            # wait(readout_laser_delay_time//4)
            wait(delay2_cc)
            wait(10000)  ### added this to allow more buffer time. 
            wait(intra_pulse_delay//4)
            align()
            
                
            if readout_on_pulse_ind == 1:
                
                play(readout_laser_pulse*amp(readout_laser_amplitude),readout_laser_key,duration=readout_laser_on_time//4) 
            
            elif readout_on_pulse_ind == 2:
                
                with for_(i,0,i<num_readouts,i+1):  
                    
                    play(readout_laser_pulse*amp(readout_laser_amplitude),readout_laser_key,duration=apd_readout_time//4) 
                    
                    wait(readout_laser_delay_time//4,"do_apd_0_gate","do_apd_1_gate")
                                        
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
                        
                    with for_(j, 0, j < counts_gate1_apd_0, j + 1):
                        save(times_gate1_apd_0[j], times_st_apd_0) 
                        
                    with for_(k, 0, k < counts_gate1_apd_1, k + 1):
                        save(times_gate1_apd_1[k], times_st_apd_1)
                        
            
            ##clock pulse that advances piezos and ends a sample in the tagger
            align()
            wait(delay3_cc)
            wait(25)
            play("clock_pulse","do_sample_clock")
            wait(25)
            align()
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            times_st_apd_0.save_all("times_apd0") 
            times_st_apd_1.save_all("times_apd1")


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
    
    # readout_time = 3e3
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
    
    qm = qmm.open_qm(config_opx)
    simulation_duration =  34000 // 4 # clock cycle units - 4ns
    num_repeat=1
    # init_pulse_time, readout_time, init_laser_key, readout_laser_key,\
      # init_laser_power, read_laser_power, readout_on_pulse_ind, apd_index  = args
    
    
    # start_t = time.time()
    # compilied_program_id = qm.compile(seq)
    # t1 = time.time()
    # print(t1 - start_t)

    # program_job = qm.queue.add_compiled(compilied_program_id)
    # job = program_job.wait_for_execution()
    # print(time.time()-t1)
    
    # job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    # job_sim.get_simulated_samples().con1.plot()
    # plt.show()
# 
    # print(time.time())
    # print(time.time())
    # job = qm.execute(seq)
    # st = time.time()
    # args = [1000,300, 2000, 'cobolt_515','cobolt_638', 'laserglow_589',1,1,0.4,2,0]
    args = [5000.0, 100.0, 5000, "cobolt_638", "cobolt_515", "cobolt_515", 1, 1, 1, 2]
    seq , f, p, ng, ss = get_seq([],config, args, num_repeat)
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
    # job = qm.execute(seq)
    
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1"], mode="wait_for_all")
    # counts_apd0, counts_apd1, times_apd0, times_apd1 = results.fetch_all() 
    # print(np.average(counts_apd0))
    # print(np.sum(counts_apd0))
    
    # args = [10000, 200e6, 'cobolt_638', 'laserglow_589',1,1,2,0]
    # seq , f, p, ng, ss = get_seq([],config, args, num_repeat)
    # job = qm.execute(seq)
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1"], mode="wait_for_all")
    # counts_apd0, counts_apd1, times_apd0, times_apd1 = results.fetch_all() 
    # # print(counts_apd0)
    # print(np.sum(counts_apd0))
    # print(times_apd0)
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