#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

rabi sequence with scc readout

"""

import labrad
import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from opx_configuration_file import *
from utils.tool_belt import States

def qua_program(opx, config, args, num_reps):
    
    ### get inputted parameters
    (
        readout_time, reion_time, ion_time, tau, shelf_time, uwave_tau_max,
        green_laser_name, yellow_laser_name, red_laser_name,
        sig_gen, reion_power, ion_power, shelf_power, readout_power,
    ) = args 
    
    ### specify number of gates and determine length of timetag streams to use 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 2
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / num_apds)    

    ### get laser information    
    green_laser_pulse, green_laser_delay_time, green_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,green_laser_name,reion_power)
    red_laser_pulse, red_laser_delay_time, red_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,red_laser_name,ion_power)
    yellow_laser_pulse, yellow_laser_delay_time, yellow_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,yellow_laser_name,readout_power)
        
    ### get microwave information
    uwave_delay_time = config['Microwaves'][sig_gen]['delay']
    signal_wait_time = config['CommonDurations']['uwave_buffer']
    scc_ion_readout_buffer = config['CommonDurations']['scc_ion_readout_buffer']
    
    post_wait_time = uwave_tau_max - tau
    background_wait_time = 0*signal_wait_time
    reference_wait_time = 2 * signal_wait_time
    reference_time = readout_time#signal_wait_time

    ### determine if the readout time is longer than the max opx readout time and therefore we need to loop over smaller readouts. 
    max_readout_time = config['PhotonCollection']['qm_opx_max_readout_time']
    
    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time
        
    elif readout_time <= max_readout_time:
        num_readouts=1
        apd_readout_time = readout_time
    
    tau_cc = int(round(tau / 4))
    ion_time_cc = int(round(ion_time / 4))
    post_wait_time_cc = int(round(post_wait_time/4))
    apd_readout_time_cc = int(round(apd_readout_time/4))
    reion_time_cc = int(round(reion_time/4))
    scc_ion_readout_buffer_cc = int(round(scc_ion_readout_buffer/4))
    signal_wait_time_cc = int(signal_wait_time // 4)
    period = 2 * (reion_time + signal_wait_time + tau + signal_wait_time + ion_time + scc_ion_readout_buffer + readout_time ) - tau
    
    ### compute necessary delays. 
    yellow_laser_delay_time_cc = int(round(yellow_laser_delay_time/4))
    green_m_uwave_delay_cc = int( max( (green_laser_delay_time - uwave_delay_time)/4 , 4 ) )
    green_m_red_delay_cc = int( max( (green_laser_delay_time - red_laser_delay_time)/4 , 4 ) )
    uwave_m_red_delay_cc = int( max( (uwave_delay_time - red_laser_delay_time)/4 , 4 ) )
    red_m_yellow_delay_cc = int( max( (red_laser_delay_time - yellow_laser_delay_time)/4 , 4 ) )
    yellow_m_green_delay_cc = int( max( (yellow_laser_delay_time - green_laser_delay_time)/4 , 4 ) )

    
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
        i = declare(int)
        k = declare(int)
        
        tau_cc_qua = declare(int)
        assign(tau_cc_qua,tau_cc)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()
            play(green_laser_pulse*amp(green_laser_amplitude),green_laser_name,duration=reion_time_cc)
            align()
            # wait(green_laser_delay_time//4)
            wait(green_m_uwave_delay_cc)
            wait(signal_wait_time_cc)
            
            with if_(tau_cc_qua >= 4):
                play("uwave_ON",sig_gen, duration=tau_cc)
                align()
            with elif_(tau_cc_qua <= 3):
                align()
            
            align()
            wait(uwave_m_red_delay_cc)
            wait(signal_wait_time_cc)
            align()
            if ion_time >= 16:
                play(red_laser_pulse*amp(red_laser_amplitude),red_laser_name,duration=ion_time_cc)
            
            align()
            wait(red_m_yellow_delay_cc)
            # wait(red_laser_delay_time//4)
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
                        
            if post_wait_time>=16:
                wait(int(post_wait_time_cc))
            wait(signal_wait_time_cc)
            wait(yellow_m_green_delay_cc)
            
            align()
            play(green_laser_pulse*amp(green_laser_amplitude),green_laser_name,duration=reion_time_cc)
            align()
            
            wait(green_m_red_delay_cc)
            wait(signal_wait_time_cc)
            wait(signal_wait_time_cc)
            
            align()
            if ion_time >= 16:
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
            wait(yellow_m_green_delay_cc)
            align()
        
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
    cxn = labrad.connect()
    cxn.qm_opx.start_tag_stream([0,1])
    simulation_duration =  60000 // 4 # clock cycle units - 4ns
    
    num_repeat=1
    
    # (
    #     readout_time,
    #     reion_time,
    #     ion_time,
    #     pi_pulse,
    #     shelf_time,
    #     uwave_tau_max,
    #     green_laser_name,
    #     yellow_laser_name,
    #     red_laser_name,
    #     sig_gen_name,
    #     apd_index,
    #     reion_power,
    #     ion_power,
    #     shelf_power,
    #     readout_power,
    # ) = args 
    # tool_belt.set_delays_to_sixteen(config)
    # config['PhotonCollection']['qm_opx_max_readout_time'] = 1000
    # args = [5e3,1e3,30,90,0,90,'cobolt_515','laserglow_589','cobolt_638',
    #         'signal_generator_tsg4104a',0,None,None,0,.26]
    args = [5000.0, 1000.0, 140, 180.56, 0, 500, 
            'cobolt_515', 'laserglow_589', 'cobolt_638', 
            'sig_gen_TEKT_tsg4104a',  1, 1, None, 0.45]
    # args = [400,1000,16,90,0,90,'cobolt_515','cobolt_515','cobolt_638','signal_generator_tsg4104a',0,None,None,0,0.26]
    # args = [0, 1000.0, 350, 0, 1, 3, 'cobolt_515', 1]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)

    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
# 
    # job = qm.execute(seq)
    # seq_args_str = tool_belt.encode_seq_args(args)
    # cxn.qm_opx.stream_immediate('rabi_scc.py',3,seq_args_str)

    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")
    
    # a = time.time()
    # counts_apd0, counts_apd1 = results.fetch_all() 
    # print(counts_apd0)
    # print(counts_apd1)
    # print(cxn.qm_opx.read_counter_separate_gates(1))
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