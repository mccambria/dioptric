#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

ramsey sequence with scc readout

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
    for ind in range(7):
        durations.append(numpy.int64(args[ind]))
    tau_shrt, reion_time, ion_time, readout_time, pi_pulse, pi_on_2_pulse, tau_long = durations
    state = args[7]
    green_laser_name, red_laser_name, yellow_laser_name = args[8:11]
    green_laser_power, red_laser_power, yellow_laser_power = args[11:14]

    ### specify number of gates and determine length of timetag streams to use 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 4
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / num_apds)    
    
    ### get laser info
    green_laser_pulse, green_laser_delay_time, green_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,green_laser_name,green_laser_power)
    red_laser_pulse, red_laser_delay_time, red_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,red_laser_name,red_laser_power)
    yellow_laser_pulse, yellow_laser_delay_time, yellow_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,yellow_laser_name,yellow_laser_power)
    
    ### get microwave information
    state = States(state)
    sig_gen = config['Microwaves']['sig_gen_{}'.format(state.name)]
    rf_delay_time = config['Microwaves'][sig_gen]['delay']
    signal_wait_time = config['CommonDurations']['uwave_buffer']
    signal_wait_time_cc = int(signal_wait_time//4)
    pre_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    post_uwave_exp_wait_time = pre_uwave_exp_wait_time
    scc_ion_readout_buffer = config['CommonDurations']['scc_ion_readout_buffer']
    scc_ion_readout_buffer_cc = int(scc_ion_readout_buffer//4)

    reion_time_cc = int(reion_time//4)
    ion_time_cc = int(ion_time//4)
    readout_time_cc = int(readout_time//4)
    green_laser_delay_time_cc = int(green_laser_delay_time//4)
    red_laser_delay_time_cc = int(red_laser_delay_time//4)
    yellow_laser_delay_time_cc = int(yellow_laser_delay_time//4)

    sig_to_ref_wait_time_base = pre_uwave_exp_wait_time + post_uwave_exp_wait_time
    sig_to_ref_wait_time_shrt = sig_to_ref_wait_time_base 
    sig_to_ref_wait_time_long = sig_to_ref_wait_time_base 
    back_buffer = 200
    back_buffer_cc = int(back_buffer//4)    

    readout_time_cc = int(readout_time // 4)
    
    ### compute necessary delays
    red_m_yellow_delay_cc = max(int((red_laser_delay_time - yellow_laser_delay_time)//4),4)
    yellow_m_green_delay_cc = max(int((yellow_laser_delay_time - green_laser_delay_time)//4),4)
    green_m_red_delay_cc = max(int((green_laser_delay_time - red_laser_delay_time)//4),4)
    green_m_rf_delay_cc = max(int((green_laser_delay_time - rf_delay_time)//4),4)
    yellow_m_rf_delay_cc = max(int((yellow_laser_delay_time - rf_delay_time)//4),4)
    rf_m_red_delay_cc = max(int((rf_delay_time - red_laser_delay_time)//4),4)
    delay21_cc = int( (post_uwave_exp_wait_time + rf_m_red_delay_cc*4)//4)
    delay1_cc = int( (green_m_rf_delay_cc*4 + reion_time + pre_uwave_exp_wait_time) //4 )
    delay2_cc = int((yellow_m_green_delay_cc*4 + sig_to_ref_wait_time_long) //4)
    delay3_cc = int( (yellow_m_rf_delay_cc*4 + pre_uwave_exp_wait_time) //4 )
    tau_shrt_cc = int(tau_shrt//4)
    double_tau_shrt_cc = int(2*tau_shrt_cc)
    tau_long_cc = int(tau_long//4)
    double_tau_long_cc = int(2*tau_long_cc)
    post_uwave_exp_wait_time_cc = int(post_uwave_exp_wait_time//4)
    pi_on_2_pulse_cc = int(pi_on_2_pulse//4)
    
    
    ### determine if the readout time is longer than the max opx readout time and therefore we need to loop over smaller readouts. 
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
    
    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time
        
    elif readout_time <= max_readout_time:
        num_readouts=1
        apd_readout_time = readout_time
    
    apd_readout_time_cc = int(apd_readout_time//4)
    
    period = 2* (reion_time + delay1_cc*4 + pi_on_2_pulse + tau_shrt + tau_long + pi_on_2_pulse + delay21_cc*4 + ion_time + \
        red_m_yellow_delay_cc*4 + scc_ion_readout_buffer + readout_time + delay2_cc*4 + \
            reion_time + green_m_red_delay_cc*4 + signal_wait_time + signal_wait_time + ion_time + red_m_yellow_delay_cc*4 + \
                scc_ion_readout_buffer + readout_time + delay2_cc*4) + back_buffer
    
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
        m = declare(int)
        i = declare(int)
        k = declare(int)
        j = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()    
                        
            play(green_laser_pulse*amp(green_laser_amplitude),green_laser_name,duration=reion_time_cc)
                        
            wait(delay1_cc, sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            wait(double_tau_shrt_cc ,sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            align()
            wait(delay21_cc)        
            align()
            
            play(red_laser_pulse*amp(red_laser_amplitude),red_laser_name,duration=ion_time_cc)
            
            align()
            wait(red_m_yellow_delay_cc)
            wait(scc_ion_readout_buffer_cc)
            align()
            
            with for_(i,0,i<num_readouts,i+1):
                
                play(yellow_laser_pulse*amp(yellow_laser_amplitude),yellow_laser_name,duration=apd_readout_time_cc) 
                
                if num_apds == 2:
                    wait(yellow_laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate" )
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(counts_gate1_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    wait(yellow_laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    save(counts_gate1_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
            align()
            wait(delay2_cc) 
            align()
                        
            play(green_laser_pulse*amp(green_laser_amplitude),green_laser_name,duration=reion_time_cc)
            align()
            wait(green_m_red_delay_cc) 
            wait(signal_wait_time_cc)
            wait(signal_wait_time_cc)
            align()
            play(red_laser_pulse*amp(red_laser_amplitude),red_laser_name,duration=ion_time_cc)
            align()
            wait(red_m_yellow_delay_cc)   
            wait(scc_ion_readout_buffer_cc)
            align()
            
            with for_(k,0,k<num_readouts,k+1):
            
                play(yellow_laser_pulse*amp(yellow_laser_amplitude),yellow_laser_name,duration=apd_readout_time_cc) 
                                
                if num_apds == 2:
                    wait(yellow_laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate")
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, apd_readout_time, counts_gate2_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate2_apd_1, apd_readout_time, counts_gate2_apd_1))
                    save(counts_gate2_apd_0, counts_st_apd_0)
                    save(counts_gate2_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    wait(yellow_laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate2_apd_0, apd_readout_time, counts_gate2_apd_0))
                    save(counts_gate2_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                
            
            align()
            wait(delay2_cc) 
            align()
            
            #"now second measurement"
            
            play(green_laser_pulse*amp(green_laser_amplitude),green_laser_name,duration=reion_time_cc)
                        
            wait(delay1_cc, sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            wait(double_tau_long_cc ,sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            align()
            wait(delay21_cc)         
            align()
            
            play(red_laser_pulse*amp(red_laser_amplitude),red_laser_name,duration=ion_time_cc)
            
            align() 
            wait(red_m_yellow_delay_cc) 
            wait(scc_ion_readout_buffer_cc)
            align()
            
            with for_(j,0,j<num_readouts,j+1):
                
                play(yellow_laser_pulse*amp(yellow_laser_amplitude),yellow_laser_name,duration=apd_readout_time_cc) 
                
                if num_apds == 2:
                    wait(yellow_laser_delay_time_cc,"do_apd_0_gate","do_apd_1_gate" )
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate3_apd_0, apd_readout_time, counts_gate3_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate3_apd_1, apd_readout_time, counts_gate3_apd_1))
                    save(counts_gate3_apd_0, counts_st_apd_0)
                    save(counts_gate3_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    wait(yellow_laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate3_apd_0, apd_readout_time, counts_gate3_apd_0))
                    save(counts_gate3_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
            align()
            wait(delay2_cc) 
            align()
            
            #"now second reference measurement"
            
            play(green_laser_pulse*amp(green_laser_amplitude),green_laser_name,duration=reion_time_cc)
            align()
            wait(green_m_red_delay_cc) 
            wait(signal_wait_time_cc)
            wait(signal_wait_time_cc)
            align()
            play(red_laser_pulse*amp(red_laser_amplitude),red_laser_name,duration=ion_time_cc)
            align()
            wait(red_m_yellow_delay_cc)   
            wait(scc_ion_readout_buffer_cc)
            align()
            
            with for_(m,0,m<num_readouts,m+1):
            
                play(yellow_laser_pulse*amp(yellow_laser_amplitude),yellow_laser_name,duration=apd_readout_time_cc) 
                                
                if num_apds == 2:
                    wait(yellow_laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate")
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate4_apd_0, apd_readout_time, counts_gate4_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate4_apd_1, apd_readout_time, counts_gate4_apd_1))
                    save(counts_gate4_apd_0, counts_st_apd_0)
                    save(counts_gate4_apd_1, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    wait(yellow_laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate4_apd_0, apd_readout_time, counts_gate4_apd_0))
                    save(counts_gate4_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    align("do_apd_0_gate","do_apd_1_gate")
                
            
            align()
            wait(delay2_cc) 
            wait(back_buffer_cc)
        
        play("clock_pulse","do_sample_clock") 
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
            
    return seq, period, num_gates



def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = ''
    ### specify what one 'sample' means for the data processing. 
    sample_size = 'all_reps'
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    qm = qmm.open_qm(config_opx)
    
    simulation_duration =  85000 // 4 # clock cycle units - 4ns
    
    num_repeat=1
    args = [80.0, 
            200.0, 200, 400, 
            80, 40.0, 
            100, 
            1, 
            'cobolt_515', 'cobolt_638', 'laserglow_589',
            1, 1, 0.45]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)
    plt.figure()

    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
# 
    # job = qm.execute(seq)

    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")
    
    # a = time.time()
    # counts_apd0, counts_apd1 = results.fetch_all() 
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