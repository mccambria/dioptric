# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:07:32 2021

@author: kolkowitz
"""

import pyvisa as visa
# import visa
import time
import labrad

# address = "USB0::0x5345::0x1235::2101004::INSTR"  # power supply
address = "USB0::0x5345::0x1234::2101156::INSTR"  # multimeter

resource_manager = visa.ResourceManager()
with resource_manager.open_resource(address) as device:
    device.baud_rate = 115200
    device.read_termination = '\n'
    device.write_termination = '\n'
    device.timeout = 2000
    # for i in range(100):
    #     # for i in range(10):
    #     #     test = power_supply.read_bytes(1)
    #     #     print(test)
    #     # test = power_supply.read()
    #     # print(test == "")
    #     # power_supply.clear()
    #     # break
        
    #     # power_supply.write("*RST")
    #     response = power_supply.query("*IDN?")
    #     print(response)
        
    #     # time.sleep(0.1)
    
    #     response = power_supply.query("MEAS:VOLT?")
    #     print(repr(response))
        
    #     # time.sleep(0.1)
        
    #     response2 = power_supply.query("MEAS:CURR?")
    #     print(repr(response2))
        
    #     # time.sleep(0.1)
    
    detector_type = "PT100"
    units = "K"
    
    # device.write("*RST")
    # test = device.query('SENS:FUNC?')
    # print(test)
    
    # device.write("SYST:LOC")
    # test = device.write("SYST:REM")
    # print(test)
        
    # Set the measurement mode
    # time.sleep(0.1)
    # device.write('SENS:FUNC "RES"')
    test = device.query('SENS:FUNC?')
    print(test)
    # cmd = 'CONF:SCAL:RES 50e3'
    # device.write(cmd)
    
    # Reset the measurement parameters and supposedly set the detector
    # type, but as far as I can tell this doesn't actually do... anything
    # cmd = 'CONF:SCAL:TEMP:RTD {}'.format(detector_type)
    # device.write(cmd)
    
    # cmd = "SYST:REM"
    # cmd = "RANGE 1"
    # device.write(cmd)
    
    # cmd = "AUTO 1"
    # test = device.write(cmd)
    # cmd = "AUTO?"
    # test = device.query(cmd)
    # print(test)
    
    # Set the detector type
    # cmd = "SENS:TEMP:RTD:TYP {}".format(detector_type)
    # device.write(cmd)
    
    # # Set the display type - just show the temperature
    # device.write("SENS:TEMP:RTD:SHOW TEMP")
    
    # # Set the units
    # cmd = "SENS:TEMP:RTD:UNIT {}".format(units)
    # device.write(cmd)
    
    # # Set the update rate to fast (maximum speed)
    # device.write("RATE F")
    
    # time.sleep(0.1)
    for i in range(5):
        time.sleep(0.5)
        val = device.query("MEAS1?")
        print(val)
    # test = device.query("SENS:TEMP:RTD:TYP?")
    # test = device.query("SYST:TIME?")
    # print(test)

# with labrad.connect() as cxn:
    
#     cxn.multimeter_mp730028.reset()
#     cxn.multimeter_mp730028.config_temp_measurement("PT100", "K")
#     val = cxn.multimeter_mp730028.measure()
#     print(val)
