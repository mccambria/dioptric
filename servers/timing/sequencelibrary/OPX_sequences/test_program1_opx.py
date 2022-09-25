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
    num_apds = int(len(apd_indices))
    timetag_list_size = int(15000 / num_gates / num_apds)
    
    
    intrinsic_time_between_gates = int(88 + 56*num_apds)
    desired_time_between_gates = 300 
    time_between_gates = desired_time_between_gates - intrinsic_time_between_gates
    time_between_gates_cc = time_between_gates // 4


    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        if (num_apds == 2):
            counts_gate1_apd_0 = declare(int)  # variable for number of counts
            counts_gate1_apd_1 = declare(int)
            counts_gate2_apd_0 = declare(int) 
            counts_gate2_apd_1 = declare(int)
            times_gate1_apd_0 = declare(int, size=timetag_list_size)  
            times_gate1_apd_1 = declare(int, size=timetag_list_size)
            times_gate2_apd_0 = declare(int, size=timetag_list_size) 
            times_gate2_apd_1 = declare(int, size=timetag_list_size)
            
        if (num_apds == 1):
            counts_gate1_apd = declare(int)
            counts_gate2_apd = declare(int)
            times_gate1_apd = declare(int, size=timetag_list_size)
            times_gate2_apd = declare(int, size=timetag_list_size)
        
        
        counts_st = declare_stream()  # stream for counts
        times_st = declare_stream()
                
        n = declare(int)
        i = declare(int)
        j = declare(int)        
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()  
          
            if (num_apds == 2):
                measure("readout", "APD_0", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))  
                measure("readout", "APD_1", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_gate1_apd_1))
            
            if (num_apds == 1):
                measure("readout", "APD_{}".format(apd_indices[0]), None, time_tagging.analog(times_gate1_apd, readout_time, counts_gate1_apd))  
                
                
            align()  
            wait(time_between_gates_cc)
           
            if (num_apds == 2):
                save(counts_gate1_apd_0, counts_st)
                with for_(i, 0, i < (counts_gate1_apd_0), i + 1):
                    save(times_gate1_apd_0[i], times_st)
                save(counts_gate1_apd_1, counts_st)
                with for_(j, 0, j < (counts_gate1_apd_1), j + 1):
                    save(times_gate1_apd_1[j], times_st)
               
            if (num_apds == 1):
                save(counts_gate1_apd, counts_st)
                with for_(i, 0, i < (counts_gate1_apd), i + 1):
                    save(times_gate1_apd[i], times_st)
            
            
            
        with stream_processing():
            counts_st.buffer(num_gates).buffer(num_apds).buffer(num_reps).save_all("counts")
            times_st.save_all("times")
        
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
    from qualang_tools.results import fetching_tool

    print('hi')
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    readout_time = 3000
    qm = qmm.open_qm(config_opx)
    simulation_duration =  12000 // 4 # clock cycle units - 4ns
    x_voltage_list,y_voltage_list,z_voltage_list = [],[],[]
    num_repeat=5
    
    args = [200,readout_time,0,'green_laser_do',1]
    config = []
    seq , f, p = get_full_seq(config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # plt.xlim(100,12000)

    # job = qm.execute(seq)
    
    # res_handles = job.result_handles
    # res_handles.wait_for_all_values()
    # counts = res_handles.get("counts").fetch_all()
    # times = res_handles.get("times").fetch_all()
    
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
