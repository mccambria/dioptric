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

desired_time_between_gates = 324

intrinsic_time_between_gates = 124 - 16  #124ns delay + 16ns because the first 16ns in the wait command here don't contribute
time_between_gates = desired_time_between_gates - intrinsic_time_between_gates
time_between_gates_cc = time_between_gates // 4

# def qua_program(args, num_reps, x_voltage_list=None, y_voltage_list=None, z_voltage_list=None):
    
#     with program() as seq:
        
#         # counts_gate1_apd_0 = declare(int)  # variable for number of counts
#         # counts_gate1_apd_1 = declare(int)
#         # counts_gate2_apd_0 = declare(int)  # variable for number of counts
#         # counts_gate2_apd_1 = declare(int)
#         # counts_st = declare_stream() 
        
#         # times = declare(int, size=int(15900 / num_apds / num_gates ))  
#         # times2 = declare(int, size=int(15900 / num_apds / num_gates))  
#         # times_st = declare_stream()
        
#         n = declare(int)
#         # i = declare(int)
            
#         # with for_(n, 0, n < num_reps, n + 1):
            
#             # align()  
            
#         with for_(n,0,n<100,n+1):
#             play("laser_ON",green_laser_name,duration=int(5000 // 4))
#             wait(5000//4)

#     return seq

# def qua_program(args, num_reps, x_voltage_list, y_voltage_list, z_voltage_list):
def qua_program(args, num_reps, x_voltage_list=[], y_voltage_list=[], z_voltage_list=[]):
    
    delay, readout_time, apd_index, laser_name, laser_power = args

    delay = numpy.int64(delay)
    delay_cc = int(delay // 4)
    readout_time = numpy.int64(readout_time)

    period = numpy.int64(delay + readout_time + 300)
    period_cc = int(period // 4)
    
    num_gates = 1
    num_apds = len(apd_indices)
    timetag_list_size = int(15900 / num_gates / num_apds)
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        
        counts_gate1_apd_0 = declare(int)  # variable for number of counts
        counts_gate1_apd_1 = declare(int)
        counts_st = declare_stream()  # stream for counts
        
        times_gate1_apd_0 = declare(int, size=timetag_list_size)  # why input a size??
        times_gate1_apd_1 = declare(int, size=timetag_list_size)
        times_st = declare_stream()
                
        n = declare(int)
        i = declare(int)
        
        with for_(n, 0, n < num_reps, n + 1):
            
            align()  
            
            ###green laser
            play("laser_ON",laser_name,duration=int(period_cc))  
            
            ###apds
            if 0 in apd_indices:
                wait(delay_cc, "APD_0") # wait for the delay before starting apds
                measure("readout", "APD_0", None, time_tagging.analog(times_gate1_apd_0, readout_time, counts_gate1_apd_0))
                
            if 1 in apd_indices:
                wait(delay_cc, "APD_1") # wait for the delay before starting apds
                measure("readout", "APD_1", None, time_tagging.analog(times_gate1_apd_1, readout_time, counts_gate1_apd_1))
        
            wait(100,laser_name)
            # save the sample to the count stream. sample is a list of gates, which is a list of counts from each apd
            # if there is only one gate, it will be in the same structure as read_counter_simple wants so we are good
           
            ###trigger piezos
            if (len(x_voltage_list) > 0):
                wait((period - 200) // 4, "x_channel")
                play("ON", "x_channel", duration=100)  
            if (len(y_voltage_list) > 0):
                wait((period - 200) // 4, "y_channel")
                play("ON", "y_channel", duration=100)  
            if (len(z_voltage_list) > 0):
                wait((period - 200) // 4, "z_channel")
                play("ON", "z_channel", duration=100)  
                
            
            ###saving
            if 0 in apd_indices:
                save(counts_gate1_apd_0, counts_st)
                with for_(i, 0, i < counts_gate1_apd_0, i + 1):
                    save(times_gate1_apd_0[i], times_st)
                        
            if 1 in apd_indices:
                save(counts_gate1_apd_1, counts_st)
                with for_(i, 0, i < counts_gate1_apd_1, i + 1):
                    save(times_gate1_apd_1[i], times_st)
                    
            
            
        with stream_processing():
            counts_st.buffer(num_gates).buffer(num_apds).buffer(num_reps).save_all("counts")
            times_st.save_all("times")
        
    return seq

def get_seq(args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    final = ''
    return seq, final, [period]

def get_full_seq(args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(args, num_reps, x_voltage_list,y_voltage_list,z_voltage_list)
    final = ''
    return seq, final, [period]
    

if __name__ == '__main__':
    
    print('hi')
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    qm = qmm.open_qm(config_opx)
    simulation_duration = 12000 // 4 # clock cycle units - 4ns
    x_voltage_list,y_voltage_list,z_voltage_list = [],[],[]
    num_repeat=1
    args = [1000,400,0,'green_laser_do',1]
    
    seq , f, p = get_full_seq('opx', args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    # Simulate blocks python until the simulation is done
    job_sim.get_simulated_samples().con1.plot()
    # plt.xlim(100,12000)

# job = qm.execute(seq)
# time.sleep(2)
# job.halt()
# Execute does not block python! As this is an infinite loop, the job would run forever. In this case, we've put a 10
# seconds sleep and then halted the job.
# time.sleep(2)
# job.halt()
