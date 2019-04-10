# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:41:41 2019

@author: mccambria
"""
print(__file__.title().split('.')[0])
import Outputs.piezo as piezo
import Utils.tool_belt as tool_belt
import numpy
import time

DAQ_NAME = "Dev1"
DAQ_AO_PIEZO = 2
DAQ_DI_PULSER_CLOCK = 12
PULSE_STREAMER_IP = "128.104.160.11"
PULSER_DO_DAQ_CLOCK = 0
PULSER_DO_DAQ_GATE = 2
PULSER_DO_AOM = 3

print("\ncurrent voltage: " + str(piezo.read_daq(DAQ_NAME, DAQ_AO_PIEZO)))

voltage = 6.0
piezo.write_daq(DAQ_NAME, DAQ_AO_PIEZO, voltage)

print("\nshould be: " + str(voltage))
time.sleep(1)
print("actually is: " + str(piezo.read_daq(DAQ_NAME, DAQ_AO_PIEZO)))

numSamps = 20
voltages = numpy.arange(numSamps)
voltages = (voltages * 0.1) + 5.0
numpy.append(voltages, [voltages[len(voltages)-1]])
period = numpy.int64(10 * 10**6)

piezoTask = piezo.stream_write_daq(DAQ_NAME, DAQ_AO_PIEZO, DAQ_DI_PULSER_CLOCK,
                                   voltages, period)

print("\nshould be: " + str(voltages[0]))
time.sleep(1)
print("actually is: " + str(piezo.read_daq(DAQ_NAME, DAQ_AO_PIEZO)))

tool_belt.pulser_readout_cont_illum(PULSE_STREAMER_IP, PULSER_DO_DAQ_CLOCK,
                                    PULSER_DO_DAQ_GATE, PULSER_DO_AOM,
                                    period, period, numSamps)

print("\nshould be: " + str(voltages[len(voltages)-1]))
time.sleep(1)
print("actually is: " + str(piezo.read_daq(DAQ_NAME, DAQ_AO_PIEZO)))

piezoTask.close()

print("\nshould be: " + str(voltages[len(voltages)-1]))
time.sleep(1)
print("actually is: " + str(piezo.read_daq(DAQ_NAME, DAQ_AO_PIEZO)))
