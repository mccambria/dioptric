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
from configuration import *

def qua_program(opx):
    
    
    with program() as seq:
        
        # I make two of each because we can have up to two APDs (two analog inputs), It will save all the streams that are actually used
        with for_each_(el,config["elements"]):
            play("steady_state",el)
        
    return seq


def get_steady_state_seq(opx): #so this will give just the sequence, no repeats
    
    seq = qua_program(opx)
    
    return seq


if __name__ == '__main__':
    
    print('hi')

