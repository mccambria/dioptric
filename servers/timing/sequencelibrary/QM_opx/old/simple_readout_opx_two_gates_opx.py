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
    
    delay, readout_time, apd_index, laser_name, laser_power = args
    
    # opx_wiring = config['Wiring']['QmOpx']
    # apd_indices = config['apd_indices']
    
    # apd_indices = config['']
    # apd_indices = opx#.apd_indices
    apd_indices = opx.apd_indices
    
    num_apds = len(apd_indices)
    num_gates = 2
    timetag_list_size = int(15900 / num_gates / 2)
    readout_time = numpy.int64(readout_time)
    
    
    max_readout_time = 1000#000
   
    if readout_time > max_readout_time:
        num_readouts = int(readout_time / max_readout_time)
        apd_readout_time = max_readout_time
        
    elif readout_time<= max_readout_time:
        num_readouts=1
        apd_readout_time = readout_time
    
    laser_on_time = delay + (readout_time + 200*num_readouts ) * num_gates
    
    period = numpy.int64(delay + laser_on_time + 300)
    period_cc = int(period//4)
    
    delay = numpy.int64(delay)
    delay_cc = int(delay // 4)
    clock_delay_cc = int((period-200)//4)
    
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_gate1_apd = declare(int)
        counts_gate2_apd_0 = declare(int)  # variable for number of counts
        counts_gate2_apd_1 = declare(int)
        counts_gate2_apd = declare(int)
        times_gate1_apd_0 = declare(int,size=timetag_list_size)
        times_gate1_apd_1 = declare(int,size=timetag_list_size)
        times_gate1_apd = declare(int,size=timetag_list_size)
        times_gate2_apd_0 = declare(int,size=timetag_list_size)
        times_gate2_apd_1 = declare(int,size=timetag_list_size)
        times_gate2_apd = declare(int,size=timetag_list_size)
        times_st_apd_0 = declare_stream()
        times_st_apd_1 = declare_stream()
        times_st_apd = declare_stream()
        empty_time_stream = declare_stream()
        counts_st_apd_0 = declare_stream()
        counts_st_apd_1 = declare_stream()
        counts_st_apd = declare_stream()
        empty_stream = declare_stream()
        empty_var = declare(int)
        assign(empty_var,0)
        n = declare(int)
        i = declare(int)
        j = declare(int)
        k = declare(int)
        num_apd = declare(int)
            
    
        with for_(n, 0, n < num_reps, n + 1):
            
            align()  
            
            ###green laser
            play("laser_ON",laser_name,duration=period_cc)  
            
            
            ##trigger piezos
            # wait(clock_delay_cc,"do_sample_clock")
            # play("clock_pulse","do_sample_clock")
            
            
            ###apds
            wait(delay_cc, "do_apd_0_gate","do_apd_1_gate")
            
            with for_(i,0,i<num_readouts,i+1):  
                
                if num_apds == 2:
                # with if_(num_apds==2):
                
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    save(2, counts_st_apd_0)
                    save(8, counts_st_apd_1)
                    
                    with for_(j, 0, j < counts_gate1_apd_0, j + 1):
                        save(times_gate1_apd_0[j], times_st_apd_0) 
                    with for_(k, 0, k < counts_gate1_apd_1, k + 1):
                        save(times_gate1_apd_1[k], times_st_apd_1)
                    
                    # align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd, apd_readout_time, counts_gate1_apd))
                    save(counts_gate1_apd, counts_st_apd)
                    save(empty_var,empty_stream)
                    
                    with for_(k, 0, k < counts_gate1_apd, k + 1):
                        save(times_gate1_apd[k], times_st_apd) 
                        save(0, empty_time_stream) 
                        
                        
            align()
            play("laser_ON",laser_name,duration=period_cc)              
            with for_(i,0,i<num_readouts,i+1):  
                
                if num_apds == 2:
                # with if_(num_apds==2):
                
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, apd_readout_time, counts_gate2_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate2_apd_1, apd_readout_time, counts_gate2_apd_1))
                    save(1, counts_st_apd_0)
                    save(5, counts_st_apd_1)
                    
                    with for_(j, 0, j < counts_gate2_apd_0, j + 1):
                        save(times_gate2_apd_0[j], times_st_apd_0) 
                    with for_(k, 0, k < counts_gate2_apd_1, k + 1):
                        save(times_gate2_apd_1[k], times_st_apd_1)
                    
                    # align("do_apd_0_gate","do_apd_1_gate")
                    
                if num_apds == 1:
                
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate2_apd, apd_readout_time, counts_gate2_apd))
                    save(counts_gate2_apd, counts_st_apd)
                    save(empty_var,empty_stream)
                    
                    with for_(k, 0, k < counts_gate1_apd, k + 1):
                        save(times_gate2_apd[k], times_st_apd) 
                        save(0, empty_time_stream) 
                        
                        
     
        if num_apds == 2:
            with stream_processing():
                counts_st_apd_0.buffer(num_readouts).buffer(num_gates).save_all("counts_apd0") 
                counts_st_apd_1.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1")
                times_st_apd_0.save_all("times_apd0")
                times_st_apd_1.save_all("times_apd1")
                
        if num_apds == 1:    
            with stream_processing():
                counts_st_apd.buffer(num_readouts).buffer(num_gates).save_all("counts_apd0") 
                empty_stream.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1") 
                times_st_apd.save_all("times_apd0")
                empty_time_stream.save_all("times_apd1")
            
    return seq


def get_seq(opx, config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(opx, config, args, num_reps=1)
    final =''
    period = 0
    return seq, final, [period]

def get_full_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(opx,config, args, num_repeat)
    final = ''
    period = 0
    return seq, final, [period]
    

if __name__ == '__main__':
    
    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    
    print('hi')
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    
    readout_time = 3000
    qm = qmm.open_qm(config_opx)
    simulation_duration =  10000 // 4 # clock cycle units - 4ns
    
    num_repeat=5
    delay = 200
    args = [delay, readout_time, 0,'do_laserglow_532_dm',1]
    config = []
    apd_indices = [0,1]
    seq , f, p = get_full_seq(apd_indices,config, args, num_repeat)
    
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # plt.xlim(100,12000)

    # job = qm.execute(seq)
    
    # results_end = fetching_tool(job, data_list = ["counts_apd0","counts_apd1"], mode="live")
    
    # while results_end.is_processing():
        
    # #     counts1, counts2 = results_end.fetch_all()
    # #     # print(np.shape(counts1),np.shape(counts2))
    #     counts_apd0, counts_apd1 = results_end.fetch_all() #just not sure if its gonna put it into the list structure we want
    #     counts_apd0 = np.sum(counts_apd0,2).tolist()
    #     counts_apd1 = np.sum(counts_apd1,2).tolist()
        
    #     max_ind = max(len(counts_apd0),len(counts_apd1))
    #     counts_apd0 = counts_apd0[0:max_ind]
    #     counts_apd1 = counts_apd1[0:max_ind]
        
    #     return_counts = []
        
    #     if len(apd_indices)==2:
    #         for i in range(len(counts_apd0)):
    #             return_counts.append([counts_apd0[i],counts_apd1[i]])
                
    #     elif len(apd_indices)==1:
    #         for i in range(len(counts_apd0)):
    #             return_counts.append([counts_apd0[i]])
                
    #     print(return_counts)