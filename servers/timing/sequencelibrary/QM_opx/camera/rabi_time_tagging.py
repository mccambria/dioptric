#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

rabi sequence with time tagging. 

"""


import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from opx_configuration_file import *
from utils.tool_belt import States

def qua_program(opx, config, args, num_reps):
    
    ### get inputted parameters
    durations = []
    for ind in range(4):
        durations.append(numpy.int64(args[ind]))
    tau, polarization, readout_time, max_tau = durations
    state = args[4]
    state = States(state)
    sig_gen = config['Microwaves']['sig_gen_{}'.format(state.name)]
    laser_name = args[5]
    laser_power = args[6]
    
    ### specify number of gates and determine length of timetag streams to use 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 2
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / num_apds)  
    
    ### specify number of readouts to buffer the count stream by so the counter function on the server knows the combine iterative readouts
    ### it's just 1 here because the readout will always be much shorter than the max readout time the opx can do (~5ms)
    num_readouts = 1
    
    ### get laser information
    laser_pulse, laser_delay_time, laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,laser_name,laser_power)
    
    ### get microwave information
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
    
    ### determine necessary delays
    laser_m_uwave_delay_cc = int( max( (laser_delay_time - uwave_delay_time)/4 , 4 ) )
    uwave_m_laser_delay_cc = int( max( (uwave_delay_time - laser_delay_time)/4 , 4 ) )
    
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
        
        times_st_apd_0 = declare_stream()
        times_st_apd_1 = declare_stream()
        
        ### save an initial 0 to each time stream so that if we don't get any time tags, there is still something there. 
        ### The server knows to remove this first entry and to use the counts stream to determine which time tags are from each rep
        save(0,times_st_apd_0)
        save(0,times_st_apd_1)

        n = declare(int)
        j = declare(int)
        k = declare(int)
        
        jj = declare(int)
        kk = declare(int)
        
        tau_cc_qua = declare(int)
        assign(tau_cc_qua,tau_cc)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()    
            
            play(laser_pulse,laser_name,duration=polarization_cc) 
            
            align()
            wait(laser_m_uwave_delay_cc)
            wait(signal_wait_time_cc)
            
            with if_(tau_cc_qua >= 4):
                play("uwave_ON",sig_gen, duration=tau_cc)
                align()
            with elif_(tau_cc_qua <= 3):
                align()
            
            wait(uwave_m_laser_delay_cc)
            wait(signal_wait_time_cc)
                           
            play(laser_pulse*amp(laser_amplitude),laser_name,duration=polarization_cc) 
            
            if num_apds == 2:
                wait(laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate" )
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_gate1_apd_1))
                save(counts_gate1_apd_0, counts_st_apd_0)
                save(counts_gate1_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))
                save(counts_gate1_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
                
            ### save time tags by looping through the time tag array and saving to the stream
            with for_(j, 0, j < counts_gate1_apd_0, j + 1):
                save(times_gate1_apd_0[j], times_st_apd_0) 
                
            with for_(k, 0, k < counts_gate1_apd_1, k + 1):
                save(times_gate1_apd_1[k], times_st_apd_1)
                
            align()
            wait(mid_duration_cc)
            align()
            
            play(laser_pulse*amp(laser_amplitude),laser_name,duration=reference_laser_on_cc) 
                            
            if num_apds == 2:
                wait(laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate")
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, readout_time, counts_gate2_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate2_apd_1, readout_time, counts_gate2_apd_1))
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(counts_gate2_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate2_apd_0, readout_time, counts_gate2_apd_0))
                save(counts_gate2_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
            ### save time tags by looping through the time tag array and saving to the stream
            with for_(jj, 0, jj < counts_gate2_apd_0, jj + 1):
                save(times_gate2_apd_0[jj], times_st_apd_0) 
                
            with for_(kk, 0, kk < counts_gate2_apd_1, kk + 1):
                save(times_gate2_apd_1[kk], times_st_apd_1)
                
            align()
        
        play("clock_pulse","do_sample_clock") 
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            times_st_apd_0.save_all("times_apd0") 
            times_st_apd_1.save_all("times_apd1")
            
    return seq, period, num_gates



def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = ''
    sample_size = 'all_reps' ### specify what one 'sample' means for the data processing. 
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

    args = [100, 1000.0, 500, 100, 3, 'cobolt_515', 1]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)

    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
    # compilied_program_id = qm.compile(seq)
    # program_job = qm.queue.add_compiled(compilied_program_id)
    
    # job = program_job.wait_for_execution()
# 
    # job = qm.execute(seq)

    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","times_apd0","times_apd1"], mode="wait_for_all")
    
    # a = time.time()
    # counts_apd0, counts_apd1, times0, times1  = results.fetch_all() 
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