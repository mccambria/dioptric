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

def qua_program(opx, config, args, num_reps, x_voltage_list=[], y_voltage_list=[], z_voltage_list=[]):
    
    delay, readout_time, apd_index, laser_name, laser_power = args
    
    # opx_wiring = config['Wiring']['QmOpx']
    # apd_indices = config['apd_indices']
    
    # apd_indices = config['']
    apd_indices = opx#.apd_indices
    
    num_apds = len(apd_indices)
    num_gates = 1
    timetag_list_size = int(15900 / num_gates / num_apds)
    readout_time = numpy.int64(readout_time)
    
    
    max_readout_time = 1000
   
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
        times_gate1_apd_0 = declare(int,size=timetag_list_size)
        times_gate1_apd_1 = declare(int,size=timetag_list_size)
        times_gate1_apd = declare(int,size=timetag_list_size)
        
    
        counts_st_apd_0 = declare_stream()
        counts_st_apd_1 = declare_stream()
        counts_st_apd = declare_stream()
        empty_stream = declare_stream()
        empty_var = declare(int)
        assign(empty_var,0)
        n = declare(int)
        i = declare(int)
        
        counts_cur0 = declare(int)
        counts_cur1 = declare(int)
        counts_cur02 = declare(int)
        counts_cur12 = declare(int)
            
    
        with for_(n, 0, n < num_reps, n + 1):
            
            align()  
            
            ###green laser
            play("laser_ON",laser_name,duration=period_cc)  
            
            
            ##trigger piezos
            wait(clock_delay_cc,"do_sample_clock")
            play("clock_pulse","do_sample_clock")
            
            
            ###apds
            wait(delay_cc, "do_apd_0_gate","do_apd_1_gate")
            
            with for_(i,0,i<num_readouts,i+1):  
            # for i in range(num_readouts):  
                
                if num_apds == 2:
                
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    assign(counts_cur0,1+counts_cur0)
                    assign(counts_cur1,5+counts_cur1)
            
            
            
            align()
            with for_(i,0,i<num_readouts,i+1):  
            # for i in range(num_readouts):  
                
                if num_apds == 2:
                
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, apd_readout_time, counts_gate1_apd_1))
                    assign(counts_cur02,10+counts_cur02)
                    assign(counts_cur12,20+counts_cur12)
                    
                    # save(counts_gate1_apd_0, counts_st_apd_0)
                    # save(counts_gate1_apd_1, counts_st_apd_1)
                    
                # if num_apds == 1:
                
                #     measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd, apd_readout_time, counts_gate1_apd))
                #     save(counts_gate1_apd, counts_st_apd)
                #     save(empty_var,empty_stream)
            save(counts_cur0, counts_st_apd)
            save(counts_cur02, counts_st_apd)
            save(counts_cur1, counts_st_apd)
            save(counts_cur12, counts_st_apd)
     
        if num_apds == 2:
            with stream_processing():
                # counts_st_apd_0.buffer(num_readouts).buffer(num_gates).save_all("counts_apd0") 
                # counts_st_apd_1.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1")
                counts_st_apd.buffer(2).buffer(num_apds).save_all("counts_apd") 
                # counts_st_apd_1.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1")
                
        # if num_apds == 1:    
        #     with stream_processing():
        #         counts_st_apd.buffer(num_readouts).buffer(num_gates).save_all("counts_apd0") 
        #         empty_stream.buffer(num_readouts).buffer(num_gates).save_all("counts_apd1") 
            
    return seq


def get_seq(opx, config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(opx, config, args, num_reps=1)
    final =''
    period = 0
    return seq, final, [period]

def get_full_seq(opx, config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(opx,config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
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
    x_voltage_list,y_voltage_list,z_voltage_list = [],[],[]
    num_repeat=100000
    delay = 200
    args = [delay, readout_time, 0,'do_laserglow_532_dm',1]
    config = []
    apd_indices = [0,1]
    seq , f, p = get_full_seq(apd_indices,config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # plt.xlim(100,12000)

    job = qm.execute(seq)
    
    results_end = fetching_tool(job, data_list = ["counts_apd"], mode="live")
    
    while results_end.is_processing():
        
        counts = results_end.fetch_all()
        print(np.shape(counts[0]))
    
    
    
    # counts_apd0, counts_apd1 = results_end.fetch_all()
    # counts_apd0 = np.sum(counts_apd0,2).tolist()
    # counts_apd1 = np.sum(counts_apd1,2).tolist()
    # counts_full = []
    # for i in range(len(counts_apd0)):
    #     counts_full.append([counts_apd0[i],counts_apd1[i]])
    # print(counts_full)
    # #this gives the counts_full as a list of samples. Each sample is a list of 
    
    # results = fetching_tool(self.experiment_job, data_list = ["counts_apd0","counts_apd1"], mode="wait_for_all")

    # counts_apd0, counts_apd1 = results_end.fetch_all() #just not sure if its gonna put it into the list structure we want
    
    # if len(apd_indices)==2:
    #     pass
    # elif 0 not in apd_indices:
    #     counts_apd1 = np.copy(counts_apd0)
    #     counts_apd0 = np.zeros(np.shape(counts_apd1))
    # elif 1 not in apd_indices:
    #     counts_apd0 = np.copy(counts_apd0)
    #     counts_apd1 = np.zeros(np.shape(counts_apd0))
        
    # counts_apd0 = np.sum(counts_apd0,2).tolist()
    # counts_apd1 = np.sum(counts_apd1,2).tolist()
    # return_counts = []
    # for i in range(len(counts_apd0)):
    #     return_counts.append([counts_apd0[i],counts_apd1[i]])
    # print(return_counts)
        