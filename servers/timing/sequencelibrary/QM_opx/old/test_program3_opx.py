import time

from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from opx_configuration_file import *
import matplotlib.pyplot as plt
import numpy as np
import numpy


def qua_sequence(config, args, num_reps, x_voltage_list=[], y_voltage_list=[], z_voltage_list=[]):
    
    delay, readout_time, apd_index, laser_name, laser_power = args
    

    delay = numpy.int64(delay)
    delay_cc = int(delay // 4)
    readout_time = numpy.int64(readout_time)

    period = numpy.int64(delay + readout_time + 300)
    period_cc = int(period // 4)
    
    num_gates = 1
    apd_indices=[0,1]
    num_apds = len(apd_indices)
    timetag_list_size = int(150 / num_gates / num_apds)
    
    intrinsic_time_between_gates = int(108 + 36*num_apds)
    desired_time_between_gates = 300 
    time_between_gates = desired_time_between_gates - intrinsic_time_between_gates
    time_between_gates_cc = time_between_gates // 4

    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        if (num_apds == 2):
            counts_gate1_apd_0 = declare(int)  # variable for number of counts
            counts_gate1_apd_1 = declare(int)
            counts_cur0 = declare(int)
            counts_cur1 = declare(int)
            times_gate1_apd_0 = declare(int, size=timetag_list_size)  # why input a size??
            times_gate1_apd_1 = declare(int, size=timetag_list_size)
            
        
        if (num_apds == 1):
            counts_gate1_apd = declare(int)  # variable for number of counts
            counts_cur = declare(int)
            times_gate1_apd = declare(int, size=timetag_list_size)  # why input a size??
            times_gate1_apd = declare(int, size=timetag_list_size)
            
            
        counts_st = declare_stream()  # stream for counts
        
                
        i = declare(int)
        n = declare(int)
        
        num_readouts=1
        max_readout = 1000 #1000000
        if readout_time > max_readout:
            num_readouts = int(readout_time / max_readout)
            readout_time = int(readout_time / num_readouts)
    
        with for_(n,0,n<num_reps,n+1): 
            
            
            ### this is one readout section for long readout
            with for_(i,0,i<num_readouts,i+1):    
                
                if (len(apd_indices) == 2):
                    measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_cur0))
                    # assign(counts_cur0,1)
                    assign(counts_gate1_apd_0,counts_cur0+counts_gate1_apd_0)
                    
                    measure("readout", "do_apd_1_gate", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_cur1))
                    # assign(counts_cur1,10)
                    assign(counts_gate1_apd_1,counts_cur1+counts_gate1_apd_1)
                    
                if (len(apd_indices) == 1):
                    measure("readout", "do_apd_{}_gate".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd, readout_time, counts_cur))
                    assign(counts_gate1_apd,counts_cur+counts_gate1_apd)
                        
                
                # align()
                # wait(time_between_gates_cc)
                wait(100)  #how do i control this time?
            ###
            
            if (len(apd_indices) == 2):
                save(counts_gate1_apd_0,counts_st)
                save(counts_gate1_apd_1,counts_st)
            if (len(apd_indices) == 1):
                save(counts_gate1_apd,counts_st)
        
        with stream_processing():
            counts_st.buffer(num_gates).buffer(len(apd_indices)).buffer(num_reps).save_all("counts")
        
          
    return seq

def get_seq(config, args): #so this will give just the sequence, no repeats
    
    seq = qua_sequence(config, args, num_reps=1)
    final = ''
    period = 0
    return seq, final, [period]

def get_full_seq(config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_sequence(config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    final = ''
    period = 0
    return seq, final, [period]
    

if __name__ == '__main__':
    
    print('hi')
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    readout_time = 4000
    qm = qmm.open_qm(config_opx)
    simulation_duration = 8000 // 4 # clock cycle units - 4ns
    x_voltage_list,y_voltage_list,z_voltage_list = [],[],[]
    num_repeat=1
    
    args = [200,readout_time,0,'do_laserglow_532_dm',1]
    config = []
    seq , f, p = get_full_seq(config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # plt.xlim(100,12000)

    job = qm.execute(seq)
    
    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    counts = res_handles.get("counts").fetch_all()
    # times = res_handles.get("times").fetch_all()
    print(counts)
    # counts = res_handles.get("counts").fetch_all()
    
    
    # res_handles_tagstream = job.result_handles
    # res_handles_tagstream.wait_for_all_values()
    # counts_data = res_handles_tagstream.get("counts").fetch_all()
    # times_data = res_handles_tagstream.get("times").fetch_all()
    
    # counts_data = results.res_handles.counts.fetch_all()
    # times_data = results.res_handles.times.fetch_all()
    
    # counts_data = counts_data[0][0].tolist()
    
    
    # print(np.asarray(counts_data))
    # print('')
    # print(times_data.tolist())
# time.sleep(2)
# job.halt()
# Execute does not block python! As this is an infinite loop, the job would run forever. In this case, we've put a 10
# seconds sleep and then halted the job.
# time.sleep(2)
# job.halt()
