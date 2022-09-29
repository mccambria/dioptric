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

with program() as hello_qua:
    
    play("laser_ON","do_laserglow_532_dm",duration=300)
    
    
qmm = QuantumMachinesManager(host="128.104.160.117",port="80")
qm = qmm.open_qm(config_opx)
job = qm.execute(hello_qua)