# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:47:21 2019

@author: mccambria
"""

import labrad
import time
import numpy
from pulsestreamer import PulseStreamer as Pulser
from pulsestreamer import TriggerStart
from pulsestreamer import TriggerRearm
from pulsestreamer import OutputState
from pulsestreamer import Sequence

x_center, y_center, z_center = [0.0, 1.0, 50.0]
scan_range = 0.2
x_range = scan_range
y_range = scan_range
period = int(0.5*10**9)


with labrad.connect() as cxn:
    num_steps = 3
    num_samples = num_steps**2
    x_voltages, y_voltages = cxn.galvo.load_sweep_scan(x_center, y_center, x_range, y_range, num_steps, period)
    
#    num_steps = 5
#    num_samples = num_steps*2
#    x_voltages, y_voltages = cxn.galvo.load_cross_scan(x_center, y_center, scan_range, num_steps, period)
    
    print(x_voltages, y_voltages)
            
    pulser = Pulser('128.104.160.11')
    pulser.setTrigger(start=TriggerStart.SOFTWARE)
    seq = Sequence()
    train = [(period - 100, 0), (100, 1)]
    seq.setDigital(0, train)
    train = [(period, 1)]
    seq.setDigital(3, train)
    pulser.stream(seq, num_samples-2, OutputState([3], 0, 0))
    
#    time.sleep(5.0)
    
    pulser.startNow()
    
    time.sleep(((period / 10**9) * num_samples) + 1)
    
    print(cxn.galvo.read())
    