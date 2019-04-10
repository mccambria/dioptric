# -*- coding: utf-8 -*-
"""
Set the DAQ voltage. (Modified NI code)

Created on Mon Dec  3 10:54:42 2018

@author: aGardill (with help from NI)
"""
import nidaqmx

xPosition = 0.2381
xChan = 0

yPosition = 0.2123
yChan = 1

with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan('Dev1/ao' + str(xChan))

    task.ao_channels.add_ao_voltage_chan('Dev1/ao' + str(yChan))

    print('Channel ' + str(xChan) + ' Voltage: ' + str(xPosition))
    print('Channel ' + str(yChan) + ' Voltage: ' + str(yPosition))
    task.write([xPosition, yPosition])

    task.stop()
