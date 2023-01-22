#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

sequence for a T1 type measurement with scc readout and a yellow pulse during the wait time.

"""


import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from opx_configuration_file import *

def qua_program(opx, config, args, num_reps):
    
    ### get inputted parameters
    green_laser_key, second_init_laser_key, red_laser_key, readout_laser_key,\
                green_pulse_time, second_init_pulse_time, red_pulse_time, readout_pulse_time,\
                green_laser_power, second_init_laser_power, red_laser_power, readout_laser_power,\
                sig_gen, pi_pulse = args
    
    ### get laser information
    green_laser_pulse, green_laser_delay_time, green_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,green_laser_key,green_laser_power)
    second_init_laser_pulse, second_init_laser_delay_time, second_init_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,second_init_laser_key,second_init_laser_power)
    red_laser_pulse, red_laser_delay_time, red_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,red_laser_key,red_laser_power)
    readout_laser_pulse, readout_laser_delay_time, readout_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,readout_laser_key,readout_laser_power)

    ### specify number of gates and determine length of timetag streams to use 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 1
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / num_apds)    
    
    ### specify number of readouts to buffer the count stream by so the counter function on the server knows the combine iterative readouts
    ### here we assume its only one readout necessary (so at most ~5ms). 
    ### Need to loop over readouts if you want to read out longer. see other sequences for examples
    num_readouts = 1
    
    ### get microwave information
    uwave_delay = config['Microwaves'][sig_gen]['delay']
    uwave_buffer = config['CommonDurations']['uwave_buffer']
    scc_ion_buffer = config['CommonDurations']['scc_ion_readout_buffer']
    pi_pulse_cc = int(round(pi_pulse / 4))
    uwave_pulse, uwave_amp, uwave_time_cc = tool_belt.get_opx_uwave_pulse_info(config,pi_pulse)
    
    green_pulse_time_cc = int(round(green_pulse_time/4))
    red_pulse_time_cc = int(round(red_pulse_time/4))
    readout_pulse_time_cc = int(round(readout_pulse_time/4))
    second_init_pulse_time_cc = int(round(second_init_pulse_time/4))
    if second_init_pulse_time_cc <4:
        second_init_pulse_time_cc = 4
        second_init_laser_amplitude = 0.0
    
    ### compute necessary delays. 
    readout_pulse_time_cc = int(round(readout_pulse_time/4))
    readout_laser_delay_time_cc = int(readout_laser_delay_time/4)
    yellow_m_red_delay_cc = int(max(round((readout_laser_delay_time - red_laser_delay_time)/4),4))
    yellow_m_green_delay_cc = int(max(round((readout_laser_delay_time - green_laser_delay_time)/4),4))
    green_m_second_init_delay_cc = int(max(round((green_laser_delay_time - second_init_laser_delay_time)/4),4))
    green_m_readout_delay_cc = int(max(round((green_laser_delay_time - readout_laser_delay_time)/4),4))
    second_init_m_green_delay_cc = int(max(round((second_init_laser_delay_time - green_laser_delay_time)/4),4))
    red_m_uwave_delay_cc = int(max(round((red_laser_delay_time - uwave_delay)/4),4))
    red_m_readout_delay_cc = int(max(round((red_laser_delay_time - readout_laser_delay_time)/4),4))
    uwave_m_red_delay_cc = int(max(round((uwave_delay - red_laser_delay_time)/4),4))
    uwave_m_readout_delay_cc = int(max(round((uwave_delay - readout_laser_delay_time)/4),4))
    uwave_m_second_init_delay_cc = int(max(round((uwave_delay - second_init_laser_delay_time)/4),4))
    second_init_m_uwave_delay_cc = int(max(round((second_init_laser_delay_time-uwave_delay)/4),4))
    readout_m_red_delay_cc = int(max(round((readout_laser_delay_time - red_laser_delay_time)/4),4))
    readout_m_green_delay_cc = int(max(round((readout_laser_delay_time - green_laser_delay_time)/4),4))
    extra_5000_delay_cc = int(5000/4)
    extra_5000000_delay_cc = int(5000000/4)
    extra_10000_delay_cc = int(10000/4)
    
    period_cc = green_pulse_time_cc + second_init_pulse_time_cc + readout_pulse_time_cc
    period = int(period_cc/4)
    
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
            play(green_laser_pulse*amp(green_laser_amplitude),green_laser_key,duration=green_pulse_time_cc)
            
            align()
            wait(second_init_m_green_delay_cc)
            align()
            
            play(second_init_laser_pulse*amp(second_init_laser_amplitude),second_init_laser_key,duration=second_init_pulse_time_cc)
            
            align()
            wait(second_init_m_uwave_delay_cc)
            wait(extra_5000_delay_cc)
            align()
            
            play(uwave_pulse,sig_gen,duration=uwave_time_cc)
            
            align()
            wait(uwave_m_red_delay_cc)
            wait(uwave_buffer)
            align()
            
            play(red_laser_pulse*amp(red_laser_amplitude),red_laser_key,duration=red_pulse_time_cc)
            
            align()
            wait(red_m_readout_delay_cc)
            wait(extra_10000_delay_cc)
            align()
                
            play(readout_laser_pulse*amp(readout_laser_amplitude),readout_laser_key,duration=readout_pulse_time_cc) 
            wait(readout_laser_delay_time_cc,"do_apd_0_gate","do_apd_1_gate")
            measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, readout_pulse_time, counts_gate1_apd_0))
            measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, readout_pulse_time, counts_gate1_apd_1))
            save(counts_gate1_apd_0, counts_st_apd_0)
            save(counts_gate1_apd_1, counts_st_apd_1)
            align("do_apd_0_gate","do_apd_1_gate")
            
            align()
            wait(readout_m_green_delay_cc)
            wait(extra_5000_delay_cc)
            wait(extra_5000000_delay_cc)
            align()
            
            
        play("clock_pulse","do_sample_clock")
        wait(25)
        align()
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")


    return seq, period, num_gates


def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = '' 
    ### specify what one 'sample' means for the data processing. 
    sample_size = 'all_reps' # 'one_rep'
    
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    config = tool_belt.get_config_dict()
    # tool_belt.set_delays_to_sixteen(config)

    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    qmm.close_all_quantum_machines()
    # readout_time = 3e3
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
    
    qm = qmm.open_qm(config_opx)
    
    simulation_duration =  108400 // 4 # clock cycle units - 4ns
    num_repeat=2
    # plt.figure()
    
    args = ['cobolt_515', 'laserglow_589', 'cobolt_638', 'laserglow_589', 
            1000, 2000.0, 116, 2000.0, 
            1, 1, 1, .5, 'sig_gen_TEKT_tsg4104a', 54]
    seq , f, p, ng, ss = get_seq([],config, args, num_repeat)
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.legend(loc='upper right',fontsize=7)
    plt.show()
    # job = qm.execute(seq)
    
    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")
    # counts_apd0, counts_apd1 = results.fetch_all() 
    # print(counts_apd0)
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