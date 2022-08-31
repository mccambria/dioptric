#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:58:57 2022

Example sequence file for the OPX. This should serve as a template

@author: carterfox
"""

import numpy
import utils.tool_belt as tool_belt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *


def get_seq(pulse_streamer, config, args):
    
    
    with program() as seq:
    
    #play() #using args parameters...
    #measure()
    #save()
    
    
    return seq, final, [period]


    
    