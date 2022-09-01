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


def qua_program(args, num_reps):
    
    with program() as seq:
        
        counts = declare(int)  # variable for number of counts
        counts_st = declare_stream()  # stream for counts
        
        n = declare(int)
        times = declare(int, size=100)
        readout_time = args[4]
        
        with for_(n, 0, n < num_reps, n+1):
            
            play('NV','pi',args[0]) #using args parameters...
            wait() #wait for pi pulse to end.
            play('laser_ON','AOM')
            measure("readout", "SPCM", None, time_tagging.analog(times, readout_time, counts))
            save(counts, counts_st)
            wait(25) #wait some time before taking the reference 
            #using args parameters...
            play('laser_ON','AOM')
            measure("readout", "SPCM", None, time_tagging.analog(times, readout_time, counts))
            save(counts, counts_st)
            #sig counts are indices 0,2,4,6,...
            #ref counts are indices 1,3,5,7,...
            
    return seq


def get_seq(opx, config, args): #so this will give just the sequence, no repeats
    
    seq = qua_program(args, num_reps=1)
    
    return seq, final, [period]

def get_full_seq(opx, config, args, num_repeat): #so this will give the full desired sequence, with however many repeats are intended repeats

    seq = qua_program(args, num_reps)

    return seq, final, [period]
    
    