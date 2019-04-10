# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:02:57 2019

@author: kolkowitz
"""
import threading
import os
import datetime
import numpy

voltage = 3.0

CurrentVoltage = 4.0
    
print(CurrentVoltage)
    
    # Define a step size for the voltage to increase incrementally
    # Recall the scaling is 465 nm/V
StepSize = 0.01
    
    # Define the steps of voltage as starting with the current voltage 
NextVoltage = CurrentVoltage
    
    # Set up a loop to add the step size to the previous voltage, slowly 
    # stepping up to the desired voltage

while abs(NextVoltage - voltage) >= StepSize:
    if NextVoltage - voltage < 0.0:
        NextVoltage -= StepSize
        #task.write([NextVoltage] * len(daqAOPiezo))
        print(NextVoltage)
    else:
        NextVoltage += NextVoltage + StepSize
        #task.write([NextVoltage] * len(daqAOPiezo))
        print(NextVoltage)
        
print("piezo has arrived")