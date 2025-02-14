#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:28:48 2024

@author: sean
"""
import time
import numpy as np

from godirect import GoDirect
import fieldcontrol as fc

  
  

def measure():
    godirect = GoDirect(use_ble=False, use_usb=True)
    device = godirect.get_device(threshold=-100)
    device.open()
    device.enable_sensors([1,2,3])
    device.start(period=1000) 
    sensors = device.get_enabled_sensors() 
    
    data = {}
    for sensor in sensors:
        data[sensor.sensor_description] = []

    for i in range(0,10):
        if device.read():
            for sensor in sensors:
                data[sensor.sensor_description].extend(sensor.values)
                sensor.clear()
    device.stop()
    device.close()
    return data
    
currents = np.arange(0.1,1.6,0.1)

data = []
for i in range(0,len(currents)):
    
    a = fc.RS_NGC103(IP='128.32.239.90',start_open = True)
    a.set_current(1,0.05)
    a.activateChannel(1)
    a.activateMaster()
    
    a.set_current(1, currents[i])
    a.maintain_current(1, currents[i])
    
    time.sleep(2)
    
    currentCurrent = a.get_current(1)
    measurement = measure()
    data.append([currentCurrent, measurement])
    print(currentCurrent,measurement)
    
    time.sleep(2)
    
    
    a.deactivateMaster()
    a.deactivateAll()
    a.close_connection()

