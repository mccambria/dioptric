#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:28:48 2024

@author: sean
"""
import time
import numpy as np

from gdx import gdx
import fieldcontrol as fc

gdx = gdx.gdx()
  
  

def measure():
    gdx.open(connection='usb', device_to_open='GDX-3MG 015002C9')
    gdx.select_sensors([1,2,3])
    gdx.start(50) 
    
    data = []
    for i in range(0,20):
        measurements = gdx.read()
        if measurements == None:
            break 
        data.append(measurements)

    gdx.stop()
    gdx.close()
    return np.mean(data,0)
    
currents = np.linspace(0.1,2,20)
a = fc.RS_NGC103(IP='128.32.239.90',start_open = True)
a.set_current(1,0.05)
a.activateChannel(1)
a.activateMaster()

for i in range(0,20):
    a.set_current(1, currents[i])
    print(a.get_current(1),measure())
    time.sleep(5)
    
    
a.deactivateMaster()
a.deactivateAll()
a.close_connection()

