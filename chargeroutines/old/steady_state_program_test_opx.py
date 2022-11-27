import time

from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from opx_configuration_file import *
import matplotlib.pyplot as plt
import numpy as np


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

def qua_program():
    
    with program() as seq:
 
        n = declare(int)
        
            
        with for_(n,0,n<100000,n+1):
            play("laser_ON",green_laser_name,duration=int(500000 // 4))
            wait(10000//4)
        
        

    return seq


def get_steady_state_seq(opx): #so this will give just the sequence, no repeats
    
    seq = qua_program()

    return seq


if __name__ == '__main__':
    
    print('hi')