"""
Sequence for generating constants outputs with the opx, such as for leaving a laser on during alignment. 
"""

import time
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from opx_configuration_file import *
import matplotlib.pyplot as plt
import numpy as np
import utils.tool_belt as tool_belt
import numpy
from qualang_tools.units import unit

period = 0 

def qua_program(opx, config,args, num_reps):
    
    opx_wiring = config['Wiring']['QmOpx']    
    high_digital_channels = args[0]
    analog_elements_to_set = args[1]
    analog_frequencies = args[2]
    analog_amplitudes = args[3]
    
    with program() as seq:
        
        play('zero_clock_pulse',"do_sample_clock")
    
        for dig_element in high_digital_channels:
            with infinite_loop_():
                play('constant_HIGH',dig_element)
                
        for an_element,an_freq,an_amp in zip(analog_elements_to_set, analog_frequencies, analog_amplitudes):
            update_frequency(an_element,an_freq)
            with infinite_loop_():
                play("cw"*amp(an_amp),an_element)
       
    return seq
        
        
def get_seq(opx,config, args, num_repeat): #so this will give just the sequence, no repeats
    
    seq = qua_program(opx, config,args, num_reps=num_repeat)
    final = ''
    period = ''
    num_gates = 0
    sample_size = None
    return seq, final, [period], num_gates, sample_size


if __name__ == '__main__':
    from opx_configuration_file import *

    from qualang_tools.results import fetching_tool, progress_counter
    import matplotlib.pylab as plt
    import time
    
    config = tool_belt.get_config_dict()
    qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
    qm = qmm.open_qm(config_opx)
    
    simulation_duration =  10000 // 4 # clock cycle units - 4ns
    num_repeat=3
    delay = 1000
    args = [['do_laserglow_532_dm', 'do_signal_generator'], ['AOD_1X', 'AOD_1Y'], [0.0, 10000000.0], [1.0, 0.5]]
    args = [],['laserglow_589'],[0],[.5]
    seq , f, p, ng, ss = get_seq([],config, args, num_repeat)
    
    job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))
    job_sim.get_simulated_samples().con1.plot()
    # plt.show()
# 
    # job = qm.execute(seq)
    
    # print('job.halt() to end infinite loop')
    