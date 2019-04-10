# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 18:06:50 2019

@author: mccambria
"""

# The PyVISA library is a python wrapper around NI's VISA IO standard.
# Docs here: https://pyvisa.readthedocs.io/en/master/
import visa
import Utils.tool_belt as tool_belt

resourceManager = visa.ResourceManager()

sigGen = tool_belt.get_VISA_instr("TCPIP0::128.104.160.12::5025::SOCKET")

sigGen.write("FREQ %fGHZ" % (2.875))
sigGen.write("POW %fDBM" % (-80.0))
sigGen.write("ENBR 1")

input("Press enter to stop...")

sigGen.write("ENBR 0")
