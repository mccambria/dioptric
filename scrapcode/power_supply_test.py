# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:07:32 2021

@author: kolkowitz
"""

import pyvisa as visa
# import visa
import time

address = "USB0::0x5345::0x1235::2101004::INSTR"

resource_manager = visa.ResourceManager()
with resource_manager.open_resource(address) as power_supply:
    power_supply.baud_rate = 115200
    power_supply.read_termination = '\n'
    power_supply.write_termination = '\n'
    for i in range(100):
        # for i in range(10):
        #     test = power_supply.read_bytes(1)
        #     print(test)
        # test = power_supply.read()
        # print(test == "")
        # power_supply.clear()
        # break
        
        # power_supply.write("*RST")
        response = power_supply.query("*IDN?")
        print(response)
        
        # time.sleep(0.1)
    
        response = power_supply.query("MEAS:VOLT?")
        print(repr(response))
        
        # time.sleep(0.1)
        
        response2 = power_supply.query("MEAS:CURR?")
        print(repr(response2))
        
        # time.sleep(0.1)
