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
import matplotlib.pylab as plt

apd_readout_time = 300


with program() as hello_qua:
    times_gate1_apd_0 = declare(int,size=100)
    counts_gate1_apd_0 = declare(int)
    
    times_gate2_apd_0 = declare(int,size=100)
    counts_gate2_apd_0 = declare(int)
    
    counts_st = declare_stream()
    # times_st = declare_stream()
    # j = declare(int)
    # n = declare(int)
    # k = declare(int)
    play("laser_ON_ANALOG"*amp(1),'laserglow_589',duration=100)
    
    # with for_(n, 0, n < 100, n + 1):
    #     play("laser_ON","do_laserglow_532_dm",duration=1000 // 4)
    # play("cw","AOD_1X",duration=1000 // 4)
    # play('uwave_ON','signal_generator_tsg4104a',duration=100)
    # measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate1_apd_0, apd_readout_time, counts_gate1_apd_0))
    # 
    # save(counts_gate1_apd_0,counts_st)
    # with for_(j, 0, j < counts_gate1_apd_0, j + 1):
    #     save(j, times_st) 
    # measure("readout", "do_apd_0_gate", None, time_tagging.analog(times_gate2_apd_0, apd_readout_time, counts_gate2_apd_0))
    
    # save(counts_gate2_apd_0,counts_st)
    # with for_(k, 0, j < counts_gate2_apd_0, j + 1):
    #     save(k, times_st) 
    
qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
qm = qmm.open_qm(config_opx)

simulation_duration = 1000
job_sim = qm.simulate(hello_qua, SimulationConfig(simulation_duration))
job_sim.get_simulated_samples().con1.plot()
# plt.show()

# job = qm.execu1te(hello_qua)