# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 10:54:42 2018

Exersize for myself

@author: aGardill (with help from NI)
"""
import nidaqmx

xPosition=0.0
yPosition=0.0

with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan('Dev1/ao0')

    task.ao_channels.add_ao_voltage_chan('Dev1/ao1')
  
    print('Channel 0 Voltage: '  + str(xPosition))
    print('Channel 1 Voltage: '  + str(yPosition))
    task.write([xPosition, yPosition])
    task.stop()

    task.stop()
