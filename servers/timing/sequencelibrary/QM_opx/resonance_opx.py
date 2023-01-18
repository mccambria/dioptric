#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

esr sequence for the opx in qua

"""


import numpy
import utils.tool_belt as tool_belt
from utils.tool_belt import States
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from opx_configuration_file import *

def qua_program(opx, config, args, num_reps):
    
    ### get inputted parameters
    readout, state, laser_name, laser_power, apd_index = args
    
    ### get laser info
    laser_pulse, laser_delay, laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,laser_name,laser_power)
    
    ### get microwave info
    state = States(state)
    sig_gen_name = config['Microwaves']['sig_gen_{}'.format(state.name)]
    uwave_delay = config['Microwaves'][sig_gen_name]['delay']
    meas_buffer = config['CommonDurations']['cw_meas_buffer']
    transient = 0
    
    readout_time = numpy.int64(readout)
    front_buffer = max(uwave_delay, laser_delay)
    
    
    ### specify number of gates and determine length of timetag streams to use 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 2
    total_num_gates = int(num_gates*num_reps)
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
    laser_on_time =  apd_readout_time + delay_between_readouts_iterations
    laser_on_time_cc = int(laser_on_time // 4)
    meas_buffer_cc = int(meas_buffer//4)
    front_buffer_cc = int(front_buffer // 4)
    front_buffer_m_uwave_delay_cc = int( (front_buffer - uwave_delay) //4)
    laser_m_uwave_delay_cc = int( max(laser_delay - uwave_delay , 4 ) )
    delay1_cc = int(front_buffer_cc + laser_m_uwave_delay_cc)
    delay2_cc = int(front_buffer_cc + laser_delay/4 )
    
    meas_buffer_p_transient_cc = int( (meas_buffer + transient) //4 )
    uwave_on_time = num_readouts*(apd_readout_time + 200)
    uwave_on_time_cc = int(uwave_on_time//4)
    period = front_buffer + 200 + 2 * (transient + num_readouts*(apd_readout_time + 200) + meas_buffer)
    period_cc = int(period // 4)
    
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
        
        n = declare(int)
        i = declare(int)
        j = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()              
            
            # start the laser a little earlier than the apds
            play(laser_pulse*amp(laser_amplitude),laser_name,duration=period_cc)
            
            wait(delay1_cc, sig_gen_name)
            wait(delay2_cc,"do_apd_0_gate","do_apd_1_gate")
            
            
            with for_(i,0,i<num_readouts,i+1):  
                
                # play("laser_ON",laser_name,duration=laser_on_time_cc) 
                
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
            align(sig_gen_name,"do_apd_0_gate","do_apd_1_gate",sig_gen_name)
            
            
            wait(meas_buffer_p_transient_cc,"do_apd_0_gate","do_apd_1_gate",sig_gen_name)
                        
            play("uwave_ON",sig_gen_name,duration=uwave_on_time_cc)            

            with for_(i,0,i<num_readouts,i+1):  
                                
                if num_apds == 2:
                    
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, apd_readout_time, counts_gate2_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate2_apd_1, apd_readout_time, counts_gate2_apd_1))
                    save(counts_gate2_apd_0, counts_st_apd_0)
                    save(counts_gate2_apd_1, counts_st_apd_1)
                    
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    # play("laser_ON",laser_name,duration=laser_on_time_cc)  
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate2_apd_0, apd_readout_time, counts_gate2_apd))
                    save(counts_gate2_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    
                    align("do_apd_0_gate","do_apd_1_gate")
                    
            ##clock pulse that ends a sample in the tagger
            align("do_apd_0_gate","do_apd_1_gate",sig_gen_name,"do_sample_clock")
            wait(meas_buffer_cc, sig_gen_name,"do_apd_0_gate","do_apd_1_gate","do_sample_clock")
            
            play("clock_pulse","do_sample_clock")
            wait(25,sig_gen_name,"do_apd_0_gate","do_apd_1_gate")
            
        
        with stream_processing():
            counts_st_apd_0.buffer(num_readouts).save_all("counts_apd0") 
            counts_st_apd_1.buffer(num_readouts).save_all("counts_apd1")
    return seq, period, num_gates



def get_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq, period, num_gates = qua_program(opx,config, args, num_repeat)
    final = ''
    ### specify what one 'sample' means for the data processing.
    sample_size = 'one_rep'
    return seq, final, [period], num_gates, sample_size
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    qm = qmm.open_qm(config_opx)
    
    simulation_duration =  25000 // 4 # clock cycle units - 4ns
    apd_index = 0
    laser_power = 1
    laser_name = 'cobolt_515'
    state = 1
    readout=1e3
    num_repeat = 1
    args = [readout, state, laser_name, laser_power, apd_index]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)

    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # plt.show()
# 
    # job = qm.execute(seq)

    # results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")
    
    # counts_apd0, counts_apd1 = results.fetch_all() 
    
    # # print('')
    # print(counts_apd0.tolist())
    # # print('')
    # print(counts_apd1.tolist())