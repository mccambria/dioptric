# -*- coding: utf-8 -*-
"""
XYZ control, encompassing the galvo and the piezo

Created on Mon Mar  4 22:30:12 2019

@author: mccambria
"""


# %% Imports


# User modules
import Outputs.galvo as galvo
import Outputs.objective_piezo as objective_piezo

# Library modules


# %% Writes

def write_daq(daqName, daqAOGalvoX, daqAOGalvoY, piezoSerial,
              xVoltage, yVoltage, zVoltage):
    """
    Set the galvo and piezo to the specified voltages.

    Params:
        daqName: string
            The name of the DAQ
        daqAOGalvoX: int
            DAQ AO carrying galvo X signal
        daqAOGalvoY: int
            DAQ AO carrying galvo Y signal
        piezoSerial: string
            Objective piezo serial number
        xVoltage: float
            Galvo x voltage
        yVoltage: float
            Galvo y voltage
        zVoltage: float
            Piezo voltage
    """

    galvo.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, xVoltage, yVoltage)

    objective_piezo.write_single_open_loop(piezoSerial, zVoltage)
