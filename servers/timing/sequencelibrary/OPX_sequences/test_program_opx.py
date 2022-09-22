import time

from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from opx_configuration_file import *
import matplotlib.pyplot as plt
import numpy as np
import numpy


# with program() as hello_QUA:
#     a = declare(fixed)
#     with for_(a, 0, a < 1.1, a + 0.05):
#         play("pi" * amp(a), "NV")
# apd_indices = [0,1]
green_laser_name = 'green_laser_do'
red_laser_name = 'red_laser_do'
period = 800 
period_cc = period // 4 #clock cycles
readout_time = 100 #ns
num_reps = 6
delay = 0
delay_cc = delay // 4 #clock cycles
num_gates = 1
apd_indices = [0,1]
num_apds = len(apd_indices)
time_of_flight_delay = detection_delay
time_of_flight_delay_cc = time_of_flight_delay // 4
dead_time = 200
dead_time_cc = dead_time // 4

desired_time_between_gates = 2600

intrinsic_time_between_gates = 124 - 12   #124ns delay + 12ns because the first 16ns in the wait command here don't contribute
time_between_gates = desired_time_between_gates - intrinsic_time_between_gates
time_between_gates_cc = time_between_gates // 4


# def qua_program(args, num_reps, x_voltage_list, y_voltage_list, z_voltage_list):
def qua_program(args, num_reps, x_voltage_list=[], y_voltage_list=[], z_voltage_list=[]):
    
    delay, readout_time, apd_index, laser_name, laser_power = args

    delay = numpy.int64(delay)
    delay_cc = int(delay // 4)
    readout_time = numpy.int64(readout_time)

    period = numpy.int64(delay + readout_time + 300)
    period_cc = int(period // 4)
    
    num_gates = 2
    num_apds = len(apd_indices)
    timetag_list_size = int(100 / num_gates / num_apds)
    print(config_opx['pulses']['readout_pulse']['length'])
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_gate2_apd_0 = declare(int)  # variable for number of counts
        counts_gate2_apd_1 = declare(int)
        counts_gate3_apd_0 = declare(int)  # variable for number of counts
        counts_gate3_apd_1 = declare(int)
        counts_st = declare_stream()  # stream for counts
        
        times_gate1_apd_0 = declare(int, size=timetag_list_size)  # why input a size??
        times_gate1_apd_1 = declare(int, size=timetag_list_size)
        times_gate2_apd_0 = declare(int, size=timetag_list_size)  # why input a size??
        times_gate2_apd_1 = declare(int, size=timetag_list_size)
        times_gate3_apd_0 = declare(int, size=timetag_list_size)  # why input a size??
        times_gate3_apd_1 = declare(int, size=timetag_list_size)
        times_st = declare_stream()
                
        n = declare(int)
        i = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()  
            
            ###green laser
            # play("laser_ON",laser_name,duration=int(period_cc))  
            
            ###gate 1 of apds
            # wait(time_between_gates_cc)
            if 0 in apd_indices:
                # wait(time_between_gates_cc) # wait for the delay before starting apds
                measure("readout", "APD_0", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))  
            if 1 in apd_indices:
                # wait(time_between_gates_cc,"APD_1")
                measure("readout", "APD_1", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_gate1_apd_1))
                
            
            
            ###gate 2 of apds
            if num_apds == 2:  # wait for them both to finish if we are using two apds
                align("APD_0","APD_1")
                
            wait(time_between_gates_cc) #make the total wait time between gates the desired time
            if 0 in apd_indices:
                measure("readout", "APD_0", None, time_tagging.analog(times_gate2_apd_0, readout_time, counts_gate2_apd_0))
            if 1 in apd_indices:
                measure("readout", "APD_1", None, time_tagging.analog(times_gate2_apd_1, readout_time, counts_gate2_apd_1))
            
                
            # save all the data
            if 0 in apd_indices:
                save(counts_gate1_apd_0, counts_st)
                with for_(i, 0, i < (counts_gate1_apd_0), i + 1):
                    save(times_gate1_apd_0[i], times_st)
                save(counts_gate2_apd_0, counts_st)
                with for_(i, 0, i < (counts_gate2_apd_0), i + 1):
                    save(times_gate2_apd_0[i], times_st)
                    
            if 1 in apd_indices:
                save(counts_gate1_apd_1, counts_st)
                with for_(i, 0, i < (counts_gate1_apd_1), i + 1):
                    save(times_gate1_apd_1[i], times_st)
                save(counts_gate2_apd_1, counts_st)
                with for_(i, 0, i < (counts_gate2_apd_1), i + 1):
                    save(times_gate2_apd_1[i], times_st)
            
            # if 0 in apd_indices:
            #     save(counts_gate2_apd_0, counts_st)
            #     with for_(i, 0, i < (counts_gate2_apd_0), i + 1):
            #         save(times_gate2_apd_0[i], times_st)
            # if 1 in apd_indices:
            #     save(counts_gate2_apd_1, counts_st)
            #     with for_(i, 0, i < (counts_gate2_apd_1), i + 1):
            #         save(times_gate2_apd_1[i], times_st)
            
            
            
        with stream_processing():
            counts_st.buffer(num_gates).buffer(num_apds).buffer(num_reps).save_all("counts")
            times_st.save_all("times")
        
    return seq

def get_seq(args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    final = ''
    return seq, final, [period]

def get_full_seq(args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    final = ''
    return seq, final, [period]
    

if __name__ == '__main__':
    
    print('hi')
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    readout_time = 12000
    config_opx['pulses']['readout_pulse']['length']=readout_time
    qm = qmm.open_qm(config_opx)
    simulation_duration = 30000 // 4 # clock cycle units - 4ns
    x_voltage_list,y_voltage_list,z_voltage_list = [],[],[]
    num_repeat=1
    
    args = [200,readout_time,0,'green_laser_do',1]
    
    seq , f, p = get_full_seq(args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    
    # job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    # Simulate blocks python until the simulation is done
    # job_sim.get_simulated_samples().con1.plot()
    # plt.xlim(100,12000)

# job = qm.execute(seq)
# time.sleep(2)
# job.halt()
# Execute does not block python! As this is an infinite loop, the job would run forever. In this case, we've put a 10
# seconds sleep and then halted the job.
# time.sleep(2)
# job.halt()
