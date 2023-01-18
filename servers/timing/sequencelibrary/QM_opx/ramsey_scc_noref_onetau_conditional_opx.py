#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:16:25 2022

@author: carterfox

Sequence for a ramsey type experiment with one tau and no reference measurement, using scc readout.
Conditional logic is applied with a charge state check after green initialization and by separating the readout
into chunks so it can be stopped if a photon is detected quickly, helping to preserve the charge state. 

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
    for ind in range(6):
        durations.append(numpy.int64(args[ind]))
    tau, reion_time, ion_time, readout_time, pi_pulse, pi_on_2_pulse = durations
    state = args[6]
    green_laser_name, red_laser_name, yellow_laser_name = args[7:10]
    green_laser_power, red_laser_power, yellow_laser_power = args[10:13]
    photon_threshold = args[13]
    chop_factor = args[14]
    
    ### specify number of gates and determine length of timetag streams to use 
    apd_indices =  config['apd_indices']
    num_apds = len(apd_indices)
    num_gates = 1
    total_num_gates = int(num_gates*num_reps)
    timetag_list_size = int(15900 / num_gates / num_apds)    
    
    ### get laser info
    green_laser_pulse, green_laser_delay_time, green_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,green_laser_name,green_laser_power)
    red_laser_pulse, red_laser_delay_time, red_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,red_laser_name,red_laser_power)
    yellow_laser_pulse, yellow_laser_delay_time, yellow_laser_amplitude = tool_belt.get_opx_laser_pulse_info(config,yellow_laser_name,yellow_laser_power)
    
    ### get microwave information
    state = States(state)
    sig_gen = config['Microwaves']['sig_gen_{}'.format(state.name)]
    pre_uwave_exp_wait_time = config['CommonDurations']['uwave_buffer']
    rf_delay_time = config['Microwaves'][sig_gen]['delay']
    post_uwave_exp_wait_time = pre_uwave_exp_wait_time
    scc_ion_readout_buffer = config['CommonDurations']['scc_ion_readout_buffer']
    scc_ion_readout_buffer_cc = int(scc_ion_readout_buffer//4)
    
    ### get necessary times and delays and put them in clock cycles
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
    
    red_m_yellow_delay_cc = max(int((red_laser_delay_time - yellow_laser_delay_time)//4),4)
    yellow_m_green_delay_cc = max(int((yellow_laser_delay_time - green_laser_delay_time)//4),4)
    green_m_rf_delay = max(green_laser_delay_time-rf_delay_time , 16)
    rf_m_red_delay_cc = max(int((rf_delay_time - red_laser_delay_time)//4),4)
    delay21_cc = int( (post_uwave_exp_wait_time + rf_m_red_delay_cc*4)//4)
    
    wait_after_init_pulse = 2000
    delay1_cc = int( (green_m_rf_delay + reion_time + wait_after_init_pulse) //4 )
    delay2_cc = int((yellow_m_green_delay_cc*4 + sig_to_ref_wait_time_long) //4)
    tau_cc = int(tau//4)
    double_tau_cc = int(2*tau_cc)
    pi_on_2_pulse_cc = int(pi_on_2_pulse//4)
    
    ### the readout is split up according to the chop factor which determines how long each smaller readout window is. 
    ### it is assumed the readout chunks are shorter than the opx max readout time (~5ms)
    apd_readout_time = int(readout_time / chop_factor)
    num_readouts = int(readout_time / apd_readout_time)
    apd_readout_time_cc = int(apd_readout_time//4)
    
    max_num_readouts = num_readouts

    period = 2* (reion_time + delay1_cc*4 + pi_on_2_pulse + double_tau_cc*4 + pi_on_2_pulse + delay21_cc*4 + \
                 ion_time + red_m_yellow_delay_cc*4 + scc_ion_readout_buffer + readout_time + delay2_cc*4 + back_buffer)
    
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
        counts_total_st = declare_stream()
        counts_all_st = declare_stream()
        num_ops_st = declare_stream() 
        reinit_state_st = declare_stream()
        j_st = declare_stream()
        i_st = declare_stream()

        n = declare(int)
        m = declare(int)
        i = declare(int)
        k = declare(int)
        j = declare(int)
        
        condition_met = declare(bool)
        reinit_state = declare(bool)
        counts_total = declare(int)
        assign(reinit_state,True)
        
        with for_(n, 0, n < num_reps, n + 1):

            assign(j,0)
            assign(condition_met,False)
            assign(counts_total,0)
            save(reinit_state,reinit_state_st)
            
            align()    
            
            #'now first only measurement
            with if_(reinit_state==True):
                play(green_laser_pulse*amp(green_laser_amplitude),green_laser_name,duration=reion_time_cc)
                wait(delay1_cc, sig_gen)
            with elif_(reinit_state==False):
                wait(delay1_cc, sig_gen)
                
            
            # wait(delay1_cc, sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            wait(double_tau_cc ,sig_gen)
            play("uwave_ON",sig_gen, duration=pi_on_2_pulse_cc)
            align()
            wait(delay21_cc)        
            align()
            
            play(red_laser_pulse*amp(red_laser_amplitude),red_laser_name,duration=ion_time_cc)
            
            align()
            wait(red_m_yellow_delay_cc)
            wait(scc_ion_readout_buffer_cc)
            align()
            
            with while_(condition_met==False):
                
                play(yellow_laser_pulse*amp(yellow_laser_amplitude),yellow_laser_name,duration=apd_readout_time_cc) 
                
                if num_apds == 2:
                    wait(yellow_laser_delay_time_cc ,"do_apd_0_gate","do_apd_1_gate" )
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    assign(counts_total,counts_total+counts_gate1_apd_0+counts_gate1_apd_1)
                    save(counts_total,counts_all_st)
                    align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                    wait(yellow_laser_delay_time_cc ,"do_apd_{}_gate".format(apd_indices[0]))
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(counts_gate1_apd_0, apd_readout_time, counts_gate1_apd))
                    # save(counts_gate1_apd_0, counts_st_apd_0)
                    save(0, counts_st_apd_1)
                    assign(counts_total,counts_total+counts_gate1_apd_0)
                    align("do_apd_0_gate","do_apd_1_gate")
                
                assign(j,j+1)
                
                with if_(counts_total>=photon_threshold):
                    assign(condition_met,True)
                    assign(reinit_state,False)
                
                with else_():
                    
                    with if_(j>=max_num_readouts):
                        assign(condition_met,True)
                        assign(reinit_state,True)
                    
                    with else_():
                        assign(condition_met,False)
                        
                        
                save(condition_met,j_st)
            save(reinit_state,i_st)
                
            save(counts_total,counts_total_st)
            save(0, counts_st_apd_1)
            save(j,num_ops_st)
            
            align()
            wait(delay2_cc) 
            wait(back_buffer_cc)
        
        play("clock_pulse","do_sample_clock") 
        
        with stream_processing():
            counts_total_st.buffer(1).save_all("counts_apd0") 
            counts_st_apd_1.buffer(1).save_all("counts_apd1")
            num_ops_st.save_all('num_ops_1')
            reinit_state_st.boolean_to_int().save_all('num_ops_2')
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
    
    simulation_duration =  95000 // 4 # clock cycle units - 4ns
    
    num_repeat=5
    args = [8.0, 500, 340, 5e6, 0, 46, 1, 
            'cobolt_515', 'cobolt_638', 'laserglow_589', None, None, 0.55,
            2,10]
    seq , f, p, ns, ss = get_seq([],config, args, num_repeat)

    # job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    # job_sim.get_simulated_samples().con1.plot()
    # plt.show()
# 
    job = qm.execute(seq)

    results = fetching_tool(job, data_list = ["counts_apd0","counts_apd1","counts_all","num_ops_1","num_ops_2",
                                              "j_st","i_st"], mode="wait_for_all")
    
    # # # a = time.time()
    counts_apd0, counts_apd1, counts_total, num_ops, reinit_state_st, j_st, i_st = results.fetch_all() 
    print(counts_apd0.tolist())
    print(counts_apd1.tolist())
    print(counts_total.tolist())
    print(num_ops)
    print(reinit_state_st)
    print(j_st)
    print(i_st)
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