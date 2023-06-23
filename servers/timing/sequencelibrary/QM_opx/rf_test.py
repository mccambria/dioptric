#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test sequence for driving AODs with the OPX

Created on June 21st, 2023

@author: mccambria
"""


import numpy
import qm
from qm import qua
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig
from qm.qua import program, declare, declare_stream, stream_processing
from qm.qua import measure, wait, save, play, align, fixed, assign
from qm.qua import infinite_loop_, while_
from utils.tool_belt import States
import utils.common as common
import utils.tool_belt as tb
import utils.kplotlib as kpl
import matplotlib.pyplot as plt


def qua_program(element, freq, amp, duration, num_reps=1):
    
    with program() as seq:
        
        ### Non-loop stuff here
        qua.update_frequency(element, freq * 1e6)
        a = declare(fixed, value=amp)
        
        ### Define one pass through the loop - call it in boilerplate below
        def one_loop():
            qua.play("continuous" * qua.amp(a), element, duration=duration)
        
        ### Boilerplate for handling num_reps
        if num_reps == -1:
            with infinite_loop_():
                one_loop()
        else:
            ind = declare(fixed)
            assign(ind, 0)
            with while_(ind<num_reps):
                one_loop()
                assign(ind, ind+1)
                
    return seq


def get_seq(opx_config, config, args, num_reps=1):
    element, freq, amp, duration = args
    seq = qua_program(element, freq, amp, duration, num_reps)
    final = ""
    # specify what one 'sample' means for  readout
    sample_size = "all_reps"
    num_gates = 0
    return seq, final, [], num_gates, sample_size


if __name__ == "__main__":

    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    
    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(ip_address)
    opx = qmm.open_qm(opx_config)
    
    try:
    
        args = ["laserglow_589_x", 10, 0.2, 100, 5]
        seq = qua_program(*args)
    
        sim_config = SimulationConfig(duration=5000 // 4)
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        
    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
