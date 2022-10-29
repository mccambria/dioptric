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
from utils.tool_belt import Mod_types
from opx_configuration_file import *
from utils.tool_belt import States

def qua_program(opx, config, args, num_reps):
    
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 2
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / 2)    
    num_readouts = 1

    durations = []
    for ind in range(4):
        durations.append(numpy.int64(args[ind]))
        
    # Unpack the durations
    tau, polarization, readout_time, max_tau = durations

    # Get the APD indices
    apd_index = args[4]

    # Signify which signal generator to use
    state = args[5]
    state = States(state)
    sig_gen = config['Microwaves']['sig_gen_{}'.format(state.name)]
    
    # Laser specs
    laser_name = args[6]
    laser_power = args[7]
    
    laser_mod_type = config["Optics"][laser_name]["mod_type"]
    laser_pulse = 'laser_ON_{}'.format(eval(laser_mod_type).name)
    laser_delay_time = config['Optics'][laser_name]['delay']
    uwave_delay_time = config['Microwaves'][sig_gen]['delay']
    signal_wait_time = config['CommonDurations']['uwave_buffer']
    background_wait_time = 0*signal_wait_time
    reference_wait_time = 2 * signal_wait_time
    reference_time = readout_time#signal_wait_time

    prep_time = polarization + signal_wait_time + tau + signal_wait_time
    end_rest_time = max_tau - tau + 16 # 8/3/2022 issue with PESR and rabi, an not collecting counts if end time is 0 ns. Adding just a bit of buffer to this.
    mid_duration = polarization + reference_wait_time - readout_time
     
    # Include a buffer on the front end so that we can delay channels that
    # should start off the sequence HIGH
    reference_laser_on = reference_time + background_wait_time + end_rest_time + 2*laser_delay_time
    polarization_cc = int(polarization // 4)
    laser_delay_time_cc = int(laser_delay_time // 4)
    mid_duration_cc = int(mid_duration // 4)

    readout_time_cc = int(readout_time // 4)
    tau_cc = int(tau // 4)
    reference_laser_on_cc = int(reference_laser_on // 4)
    signal_wait_time_cc = int(signal_wait_time // 4)
    period = polarization + signal_wait_time_cc + tau + signal_wait_time_cc + polarization + mid_duration + reference_laser_on
    
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
        
        times_st_apd_0 = declare_stream()
        times_st_apd_1 = declare_stream()
        
        save(0,times_st_apd_0)
        save(0,times_st_apd_1)

        n = declare(int)
        j = declare(int)
        k = declare(int)
        
        jj = declare(int)
        kk = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()    
            
            play(laser_pulse,laser_name,duration=polarization_cc) 
            
            align()
            wait(signal_wait_time_cc)
            
            if tau_cc >= 4:
                play("uwave_ON",sig_gen, duration=tau_cc)
            
            align()
            wait(signal_wait_time_cc)
                           
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
                
            with for_(j, 0, j < counts_gate1_apd_0, j + 1):
                save(times_gate1_apd_0[j], times_st_apd_0) 
                
            with for_(k, 0, k < counts_gate1_apd_1, k + 1):
                save(times_gate1_apd_1[k], times_st_apd_1)
                
            align()
            wait(mid_duration_cc)
            align()
            
            play(laser_pulse,laser_name,duration=reference_laser_on_cc) 
                            
            if num_apds == 2:
                wait(laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate")
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, readout_time, counts_gate2_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate2_apd_1, readout_time, counts_gate2_apd_1))
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(counts_gate2_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate2_apd_0, readout_time, counts_gate2_apd))
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
            with for_(jj, 0, jj < counts_gate2_apd_0, jj + 1):
                save(times_gate2_apd_0[jj], times_st_apd_0) 
                
            with for_(kk, 0, kk < counts_gate2_apd_1, kk + 1):
                save(times_gate2_apd_1[kk], times_st_apd_1)
                
            align()
        
        play("clock_pulse","do_sample_clock") # clock pulse after all the reps so the tagger sees all reps as one sample
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            times_st_apd_0.save_all("times_apd0") 
            times_st_apd_1.save_all("times_apd1")
            
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
    
    simulation_duration =  35000 // 4 # clock cycle units - 4ns
    
    num_repeat=1

    args = [100, 100000.0, 95000, 100, 1, 3, 'cobolt_515', 1]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)

    # job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    # job_sim.get_simulated_samples().con1.plot()
    # plt.show()
# 
    job = qm.execute(seq)

    results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1"], mode="wait_for_all")
    
    # a = time.time()
    counts_apd0, counts_apd1, times0, times1  = results.fetch_all() 
    # counts_apd0 = np.sum(counts_apd0,1)
    # ref_counts = np.sum(counts_apd0[0::2])
    # sig_counts = np.sum(counts_apd0[1::2])
    # print(ref_counts/sig_counts)
    # print(np.sum(counts_apd0))
    # print(time.time()-a)
    # # print('')
    # print(counts_apd0.tolist())
    # # print('')
    # print(counts_apd1.tolist())