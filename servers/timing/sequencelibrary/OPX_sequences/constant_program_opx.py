import time
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from opx_configuration_file import *
import matplotlib.pyplot as plt
import numpy as np
import numpy
from qualang_tools.units import unit

period =0

def qua_program(opx, config,args, num_reps, x_voltage_list=[], y_voltage_list=[], z_voltage_list=[]):
    
    opx_wiring = config['Wiring']['QM_OPX']
    
    # need to figure out how to grab to elements from the opx wiring and the channel numbers
    
    high_digital_channels = args[0]
    analog_channels_to_set = args[1]
    analog_channel_values = args[2]
        
    
    with program() as seq:
                
        for element in channel:
            
            with infinite_loop_():
                play('cw',element)
            
    return seq
        
        
def get_seq(opx,config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(opx, config,args, num_reps=1)
    final = ''
    return seq, final, [period]

def get_full_seq(opx,config, args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(opx,config,args, num_repeat, x_voltage_list,y_voltage_list,z_voltage_list)
    final = ''
    return seq, final, [period]

