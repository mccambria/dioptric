# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:55:33 2019
Working on how to read voltages on AO on DAQ

@author: gardill
"""

import nidaqmx
import Utils.tool_belt as tool_belt

xPosition = 0.0
xChan = 0

DAQ = tool_belt.get_daq("dev1")

#aiPhysChans = DAQ.ai_physical_chans

#for chan in aiPhysChans:
#    print(chan)
    
with nidaqmx.Task() as task:
    
    task.ao_channels.add_ao_voltage_chan('Dev1/ao3')
    
    task.write(0.45)

with nidaqmx.Task() as task:
    
    task.ai_channels.add_ai_voltage_chan('Dev1/_ao3_vs_aognd')
    
    print(task.read())
        
    task.stop()
