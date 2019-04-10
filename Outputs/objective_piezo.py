# -*- coding: utf-8 -*-
"""
Objective piezo control functions

Created on Tue Mar 19 09:58:06 2019

@author: mccambria
"""


# %% Imports


# User modules

# Library modules
import os
from pipython import GCSDevice
from pipython import pitools


# %% Constants


DEV_NAME = 'E709'
GCS_DLL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "GCSTranslator",
                                         "PI_GCS2_DLL_x64.dll"))


# %% Writes


def write_single_closed_loop(piezoSerial, position):
    """
    Set the piezo to the specified position.

    Params:
    """

    piezo = GCSDevice("E-709")
#    piezo.ConnectUSB(piezoSerial)
    piezo.InterfaceSetupDlg(key="sample")
    axis = piezo.axes()[0]  # Just one axis for this device

    # Turn on the servo for closed loop feedback
    piezo.SVO(axis, True)

    # Write the value and wait until we get there
    piezo.MOV(axis, position)
    pitools.waitontarget(piezo)


#def write_many_closed_loop(piezoSerial, positions):
#    piezo = GCSDevice("E-709")
#    piezo.ConnectUSB(piezoSerial)
#    axis = piezo.axes()[0]  # Just one axis for this device
#
#    # Write the first value to prevent hysteresis
#    write_single_closed_loop(piezoSerial, positions[0])
#
#    # Set up the trigger input
#    lines = [0]
#    piezo.CTI(lines, )
#    piezo.TRI(lines, True)


def write_single_open_loop(piezoSerial, voltage):
    """
    Set the piezo to the specified voltage.

    Params:
    """

    with GCSDevice(devname=DEV_NAME, gcsdll=GCS_DLL_PATH) as piezo:
        piezo.ConnectUSB(piezoSerial)
        axis = piezo.axes[0]  # Just one axis for this device
    
        # Turn off the servo for closed loop feedback
        piezo.SVO(axis, False)
    
        # Write the value and wait until we get there
        piezo.SVA(axis, voltage)
#        pitools.waitonoma(piezo, axis)



# %% Reads


def read_position(piezoSerial):
    """
    Return the current piezo position.

    Params:
        daqName: string
            The name of the DAQ
        daqAOPiezo: int
            DAQ AO carrying piezo signal
        voltage: float
            Piezo voltage to set
    """

    with GCSDevice(devname=DEV_NAME, gcsdll=GCS_DLL_PATH) as piezo:
        piezo.ConnectUSB(piezoSerial)
        axis = piezo.axes[0]  # Just one axis for this device
        return piezo.qPOS()[axis]


# %% Clean up


def close(piezoSerial):
    """
    Turn off feedback and set to 0V.

    Params:
        piezoUSBAddress: string
    """

    piezo = GCSDevice("E-709")
    piezo.ConnectUSB(piezoSerial)

    piezo.SVO("Z", 0)


