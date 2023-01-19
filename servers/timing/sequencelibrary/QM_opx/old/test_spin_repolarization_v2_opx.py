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
    
    first_init_laser_key, init_laser_key, ion_laser_key, readout_laser_key,\
                first_init_pulse_time, second_init_time, ion_time, readout_time,\
                first_init_laser_power, init_laser_power, ion_laser_power, read_laser_power,\
                sig_gen, pi_pulse = args
    
    
    first_init_laser_pulse, first_init_laser_delay_time, first_init_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,first_init_laser_key,first_init_laser_power)
    init_laser_pulse, init_laser_delay_time, init_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,init_laser_key,init_laser_power)
    ion_laser_pulse, ion_laser_delay_time, ion_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,ion_laser_key,ion_laser_power)
    readout_laser_pulse, readout_laser_delay_time, readout_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,readout_laser_key,read_laser_power)
    
    pi_pulse_cc = int(round(pi_pulse / 4))
    
    delay1_cc = int( (max(first_init_laser_delay_time - init_laser_delay_time + 1000,1000))//4 )
    # print(delay1_cc)
    uwave_delay_time = config['Microwaves'][sig_gen]['delay']
    delay2_cc = int( (max(uwave_delay_time - ion_laser_delay_time + 10000,1000))//4 )
    delay7_cc = int( (max(ion_laser_delay_time - readout_laser_delay_time + 10000,1000))//4 )
    delay5_cc = int( (max(init_laser_delay_time-uwave_delay_time + 10000,1000))//4 )
    delay6_cc = int( (max(first_init_laser_delay_time-uwave_delay_time + 1000,1000))//4 )
    intra_pulse_delay = config['CommonDurations']['uwave_buffer']

    apd_indices =  config['apd_indices']
    
    num_apds = len(apd_indices)
    num_gates = 4
    timetag_list_size = int(15900 / num_gates / 2)    
    
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
    delay_between_readouts_iterations = 200 #simulated - conservative estimate
        
        
    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time
        
    elif readout_time <= max_readout_time:
        num_readouts=1
        apd_readout_time = readout_time

    init_laser_on_time = second_init_time
    readout_laser_on_time = num_readouts*(apd_readout_time) + (num_readouts-1)*(delay_between_readouts_iterations)
        
    first_init_laser_on_time = first_init_pulse_time
    period = init_laser_on_time + intra_pulse_delay + readout_laser_on_time + 300 
    ion_time_cc = int(round(ion_time/4))
    apd_readout_time_cc = int(round(apd_readout_time/4))
    init_laser_on_time_cc = int(round(init_laser_on_time/4))
    if init_laser_on_time_cc > 10e6:
        init_laser_on_time_cc = int(init_laser_on_time_cc / 2)
        num_init_laser = 2
    else:
        num_init_laser = 1
    
    first_init_laser_on_time_cc = int(round(first_init_laser_on_time/4))
    readout_laser_delay_time_cc = int(max(round(readout_laser_delay_time/4),4))
    
    with program() as seq:
        
        counts_gate1_apd_0 = declare(int)  
        counts_gate1_apd_1 = declare(int)
        times_gate1_apd_0 = declare(int,size=timetag_list_size)
        times_gate1_apd_1 = declare(int,size=timetag_list_size)
        counts_gate2_apd_0 = declare(int)  
        counts_gate2_apd_1 = declare(int)
        times_gate2_apd_0 = declare(int,size=timetag_list_size)
        times_gate2_apd_1 = declare(int,size=timetag_list_size)
        counts_gate3_apd_0 = declare(int)  
        counts_gate3_apd_1 = declare(int)
        times_gate3_apd_0 = declare(int,size=timetag_list_size)
        times_gate3_apd_1 = declare(int,size=timetag_list_size)
        counts_gate4_apd_0 = declare(int)  
        counts_gate4_apd_1 = declare(int)
        times_gate4_apd_0 = declare(int,size=timetag_list_size)
        times_gate4_apd_1 = declare(int,size=timetag_list_size)
        counts_st_apd_0 = declare_stream()
        counts_st_apd_1 = declare_stream()  
        
        n = declare(int)
        i = declare(int)
        k = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()
            play(first_init_laser_pulse*amp(first_init_laser_amplitude),first_init_laser_key,duration=first_init_laser_on_time_cc)
            align()
            
            wait(delay1_cc)
            
            if init_laser_on_time >=16:
                with for_(k,0,k<num_init_laser,k+1): 
                    play(init_laser_pulse*amp(init_laser_amplitude),init_laser_key,duration=init_laser_on_time_cc) 
                align()
            else:
                align()
                
            wait(delay5_cc)
            
            play("uwave_ON",sig_gen, duration=pi_pulse_cc)
            align()      
                
            wait(delay2_cc)
            align()
            play(ion_laser_pulse*amp(ion_laser_amplitude),ion_laser_key,duration=ion_time_cc)
            align()
            wait(delay7_cc)
            
            align()
                
            with for_(i,0,i<num_readouts,i+1):  
                
                play(readout_laser_pulse*amp(readout_laser_amplitude),readout_laser_key,duration=apd_readout_time_cc) 
                
                wait(readout_laser_delay_time//4,"do_apd_0_gate","do_apd_1_gate")
                                    
                if num_apds == 2:
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(counts_gate1_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                    
                if num_apds == 1:
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate1_apd_0, apd_readout_time, counts_gate1_apd))
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
            align()
            wait(500)
            align()
            play(first_init_laser_pulse*amp(first_init_laser_amplitude),first_init_laser_key,duration=first_init_laser_on_time_cc) 
        
            align()
            
            wait(delay1_cc)
            
            if init_laser_on_time >=16:
                with for_(k,0,k<num_init_laser,k+1): 
                    play(init_laser_pulse*amp(init_laser_amplitude),init_laser_key,duration=init_laser_on_time_cc) 
                align()
            else:
                align()
                
            wait(delay5_cc)
            
            align()
            play("uwave_ON",sig_gen, duration=pi_pulse_cc)
            align()  
                
            wait(delay2_cc)
            align()
            wait(ion_time)
            align()
            wait(delay7_cc)
            
            align()
                
            with for_(i,0,i<num_readouts,i+1):  
                
                play(readout_laser_pulse*amp(readout_laser_amplitude),readout_laser_key,duration=apd_readout_time_cc) 
                
                wait(readout_laser_delay_time//4,"do_apd_0_gate","do_apd_1_gate")
                                    
                if num_apds == 2:
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, apd_readout_time, counts_gate2_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate2_apd_1, apd_readout_time, counts_gate2_apd_1))
                    save(counts_gate2_apd_0, counts_st_apd_0)
                    save(counts_gate2_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                    
                if num_apds == 1:
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate2_apd_0, apd_readout_time, counts_gate2_apd))
                    save(counts_gate2_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                        
            
            align()
            wait(500)
            align()
            
            align()
            play(first_init_laser_pulse*amp(first_init_laser_amplitude),first_init_laser_key,duration=first_init_laser_on_time_cc) 
        
            align()
            
            wait(delay1_cc)
            
            if init_laser_on_time >=16:
                with for_(k,0,k<num_init_laser,k+1): 
                    play(init_laser_pulse*amp(init_laser_amplitude),init_laser_key,duration=init_laser_on_time_cc) 
                align()
            else:
                align()
            
            wait(delay5_cc)
            
            wait(pi_pulse_cc)
            align()     
                
            wait(delay2_cc)
            align()
            play(ion_laser_pulse*amp(ion_laser_amplitude),ion_laser_key,duration=ion_time_cc)
            align()
            wait(delay7_cc)
            
            align()
                
            with for_(i,0,i<num_readouts,i+1):  
                
                play(readout_laser_pulse*amp(readout_laser_amplitude),readout_laser_key,duration=apd_readout_time_cc) 
                
                wait(readout_laser_delay_time//4,"do_apd_0_gate","do_apd_1_gate")
                                    
                if num_apds == 2:
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate3_apd_0, apd_readout_time, counts_gate3_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate3_apd_1, apd_readout_time, counts_gate3_apd_1))
                    save(counts_gate3_apd_0, counts_st_apd_0)
                    save(counts_gate3_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                    
                if num_apds == 1:
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate3_apd_0, apd_readout_time, counts_gate3_apd))
                    save(counts_gate3_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
            align()
            wait(500)
            align()
            play(first_init_laser_pulse*amp(first_init_laser_amplitude),first_init_laser_key,duration=first_init_laser_on_time_cc) 
        
            align()
            
            wait(delay1_cc)
            
            if init_laser_on_time >=16:
                with for_(k,0,k<num_init_laser,k+1): 
                    play(init_laser_pulse*amp(init_laser_amplitude),init_laser_key,duration=init_laser_on_time_cc) 
                align()
            else:
                align()
            
            wait(delay5_cc)
            
            wait(pi_pulse_cc)
            align()  
                
            wait(delay2_cc)
            align()
            wait(ion_time_cc)
            align()
            wait(delay7_cc)
            
            
            align()
                
                
            with for_(i,0,i<num_readouts,i+1):  
                
                play(readout_laser_pulse*amp(readout_laser_amplitude),readout_laser_key,duration=apd_readout_time_cc) 
                
                wait(readout_laser_delay_time//4,"do_apd_0_gate","do_apd_1_gate")
                                    
                if num_apds == 2:
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate4_apd_0, apd_readout_time, counts_gate4_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate4_apd_1, apd_readout_time, counts_gate4_apd_1))
                    save(counts_gate4_apd_0, counts_st_apd_0)
                    save(counts_gate4_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                    
                if num_apds == 1:
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate4_apd_0, apd_readout_time, counts_gate4_apd))
                    save(counts_gate4_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                        
            
            ##clock pulse that advances piezos and ends a sample in the tagger
            align()
            wait(500)
            align()
            wait(100)
            
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
    sample_size = 'all_reps' # 'one_rep'
    
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    config = tool_belt.get_config_dict()
    # tool_belt.set_delays_to_sixteen(config)

    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    
    # readout_time = 3e3
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
    
    qm = qmm.open_qm(config_opx)
    simulation_duration =  284000 // 4 # clock cycle units - 4ns
    num_repeat=2
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
    plt.figure()
    
    args = ['cobolt_515', 'laserglow_589', 'cobolt_638', 'laserglow_589', 
            5000.0, 5000.0, 116, 5000.0, None, 0.4, None, 0.4, 'sig_gen_TEKT_tsg4104a', 54]
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