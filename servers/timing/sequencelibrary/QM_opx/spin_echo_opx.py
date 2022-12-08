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
    num_gates = 4
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / 2)    
    num_readouts = 1

    durations = []
    for ind in range(6):
        durations.append(numpy.int64(args[ind]))
        
    # Unpack the durations
    tau_shrt, polarization, readout_time, pi_pulse, pi_on_2_pulse, tau_long = durations

    # Get the APD indices
    state, laser_name, laser_power = args[6:9]
    
    state = States(state)
    sig_gen = config['Microwaves']['sig_gen_{}'.format(state.name)]
    signal_time = polarization
    
    reference_time = polarization
    pre_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    post_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    # time between signal and reference without illumination
    sig_to_ref_wait_time_base = pre_uwave_exp_wait_time + post_uwave_exp_wait_time
    sig_to_ref_wait_time_shrt = sig_to_ref_wait_time_base 
    sig_to_ref_wait_time_long = sig_to_ref_wait_time_base 
    rf_delay_time = config['Microwaves'][sig_gen]['delay']
    back_buffer = 200
    back_buffer_cc = int(round(back_buffer/4))    
    
    laser_pulse, laser_delay_time, laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,laser_name,laser_power)
    
    uwave_experiment_shrt = pi_on_2_pulse + tau_shrt + pi_pulse + tau_shrt + pi_on_2_pulse
    uwave_experiment_long = pi_on_2_pulse + tau_long + pi_pulse + tau_long + pi_on_2_pulse

    # Include a buffer on the front end so that we can delay channels that
    # should start off the sequence HIGH
    polarization_cc = int(round(polarization / 4))
    laser_delay_time_cc = int(round(laser_delay_time / 4))

    readout_time_cc = int(round(readout_time / 4))
    delay1_cc = int( round( (laser_delay_time - rf_delay_time + polarization + pre_uwave_exp_wait_time) /4) )
    delay2_cc = int( round( (laser_delay_time+sig_to_ref_wait_time_long) /4 ))
    delay21_cc = int( round( (post_uwave_exp_wait_time + laser_delay_time - rf_delay_time)/4))
    delay3_cc = int(round( (laser_delay_time - rf_delay_time + pre_uwave_exp_wait_time) /4 ))
    delay4_cc = int( round( (laser_delay_time+sig_to_ref_wait_time_shrt) /4) )
    tau_shrt_cc = int(round(tau_shrt/4))
    tau_long_cc = int(round(tau_long/4))
    post_uwave_exp_wait_time_cc = int(round(post_uwave_exp_wait_time/4))
    pi_on_2_pulse_cc = int(round(pi_on_2_pulse/4))
    pi_pulse_cc = int(round(pi_pulse/4))
    
    period_cc= delay1_cc + pi_on_2_pulse_cc + tau_shrt_cc + pi_pulse_cc + tau_shrt_cc + pi_on_2_pulse_cc + delay21_cc \
        + polarization_cc + delay2_cc + polarization_cc + delay3_cc + pi_on_2_pulse_cc + tau_long_cc + pi_pulse_cc + \
            tau_long_cc + pi_on_2_pulse_cc + delay21_cc + polarization_cc + delay4_cc + polarization_cc \
                + laser_delay_time_cc + back_buffer_cc
    period = int(period_cc*4)
    
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
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()    
            
            play(laser_pulse,laser_name,duration=polarization_cc) 
                        
            wait(delay1_cc, sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            if tau_shrt >= 16:
                wait(tau_shrt_cc ,sig_gen)
            if pi_pulse >= 16:
                play("uwave_ON",sig_gen, duration=pi_pulse_cc)
            if tau_shrt >= 16:
                wait(tau_shrt_cc ,sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            align()
            wait(delay21_cc)
            
            
            align()
                           
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
            wait(delay2_cc)
            align()
            
            play(laser_pulse,laser_name,duration=polarization_cc) 
                            
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
                
            align()
            # play(laser_pulse,laser_name,duration=int((rf_delay_time + polarization) //4) ) 
            
            # wait(int( (pre_uwave_exp_wait_time + uwave_experiment_long + post_uwave_exp_wait_time) //4 ), laser_name)
            
            wait(delay3_cc, sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            if tau_long >= 16:
                wait(tau_long_cc ,sig_gen)
            if pi_pulse >= 16:
                play("uwave_ON",sig_gen, duration=pi_pulse_cc)
            if tau_long >= 16:
                wait(tau_long_cc ,sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            align()
            wait(delay21_cc)
            # wait(int( (laser_delay_time + polarization + pre_uwave_exp_wait_time) //4) ,sig_gen)
            
            
            align()
                           
            play(laser_pulse,laser_name,duration=polarization_cc) 
            
            if num_apds == 2:
                wait(laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate" )
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate3_apd_0, readout_time, counts_gate3_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate3_apd_1, readout_time, counts_gate3_apd_1))
                save(counts_gate3_apd_0, counts_st_apd_0)
                save(counts_gate3_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate3_apd_0, readout_time, counts_gate3_apd))
                save(counts_gate3_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
            align()
            wait(delay4_cc)
            align()
            
            play(laser_pulse,laser_name,duration=polarization_cc) 
                            
            if num_apds == 2:
                wait(laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate")
                measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate4_apd_0, readout_time, counts_gate4_apd_0))
                measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate4_apd_1, readout_time, counts_gate4_apd_1))
                save(counts_gate4_apd_0, counts_st_apd_0)
                save(counts_gate4_apd_1, counts_st_apd_1)
                
            if num_apds == 1:
                wait(laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate4_apd_0, readout_time, counts_gate4_apd))
                save(counts_gate4_apd_0, counts_st_apd_0)
                save(0, counts_st_apd_1)
                
            align()
            wait(laser_delay_time_cc)
            align()
            wait(back_buffer_cc)
        
        play("clock_pulse","do_sample_clock") # clock pulse after all the reps so the tagger sees all reps as one sample
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            
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
    
    simulation_duration =  80550 // 4 # clock cycle units - 4ns
    
    num_repeat=2
# tau_shrt, polarization, readout_time, pi_pulse, pi_on_2_pulse, tau_long = durations
    args = [100, 1000.0, 350, 50, 52, 32, 1, 'cobolt_515', None]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)

    # job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    # job_sim.get_simulated_samples().con1.plot()
    # plt.show()
# 
    job = qm.execute(seq)

    results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")
    
    # a = time.time()
    counts_apd0, counts_apd1 = results.fetch_all() 
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