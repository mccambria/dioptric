# -*- coding: utf-8 -*-
"""
Input server for the "HNu 512 Gamma" EMCCD camera from NUVU.

Created on July 31st, 2023

@author: Saroj B Chand

### BEGIN NODE INFO
[info]
name = slm_THOR_exulus_hd2
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 10

[shutdown]
message = 987654321
timeout = 
### END NODE INFO
"""

import os
import sys
import ctypes
import numpy as np
import logging
from labrad.server import LabradServer, setting
from labrad import util
import socket
# Configure logging to output more detailed information
#logging.basicConfig(level=logging.DEBUG,  # Set logging level to DEBUG for verbose output
#                    format="%(asctime)s %(levelname)-8s %(message)s",  # Specify log format
#                    datefmt="%y-%m-%d %H:%M:%S")  # Specify date format

# sys.path.append('c:/Users/Saroj Chand/Documents/dioptric')
from utils import tool_belt as tb
from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.Thorlabs_EXULUS_PythonSDK.Thorlabs_EXULUS_Python_SDK.EXULUS_COMMAND_LIB import *
from slmsuite.hardware.Thorlabs_EXULUS_PythonSDK.Thorlabs_EXULUS_CGHDisplay.Thorlabs_EXULUS_CGHDisplay import *

DEFAULT_SDK_PATH = "C:/Users/Saroj Chand/Documents/dioptric/Thorlabs_EXULUS_PythonSDK"


class SlmThorExulusHd2(LabradServer):
    name = "slm_THOR_exulus_hd2"
    pc_name = socket.gethostname()

    def initServer(self):
        """
        Initializes the ThorSLMServer.
        """
        tb.configure_logging(self)
        
        self.serialNumber = None
        self.device_hdl = None
        self.window_hdl = None
        logging.info("init complete")

    def initSLM(self, c, serialNumber):
        """
        Initializes an instance of a Thorabs SLM.

        Args:
            c (Context): The Labrad context object.
            serialNumber (str): Serial number of the SLM to initialize.

        Returns:
            str: Initialization result message.
        """
        self.device_hdl = EXULUSOpen(serialNumber, 38400, 3)
        if self.device_hdl < 0:
            return f"Connect {serialNumber} failed"

        result = EXULUSIsOpen(serialNumber)
        if result < 0:
            return "Open failed"

        width, height = 1920, 1080
        self.window_hdl = CghDisplayCreateWindow(2, width, height, "SLM window")
        if self.window_hdl < 0:
            return "Create window failed"

        result = CghDisplaySetWindowInfo(self.window_hdl, width, height, 1)
        if result < 0:
            return "Set Window Info failed"

        result = CghDisplayShowWindow(self.window_hdl, None)
        if result < 0:
            return "Show window failed"

        self.serialNumber = serialNumber
        return f"Connect {serialNumber} successfully"

    @setting(1, returns='s')
    def _write_phase(self, phase):
        """
        Low-level hardware interface to write phase data onto the SLM.

        Args:
            phase (bytes): Phase data to set on the SLM.

        Returns:
            str: Result message.
        """
        if self.device_hdl is None or self.window_hdl is None:
            return "SLM not initialized"

        phase_array = np.frombuffer(phase, dtype=np.uint8)
        phase_matrix = phase_array.reshape((1080, 1920))
        flattened_matrix = phase_matrix.flatten()
        c_data = ctypes.cast(flattened_matrix.ctypes.data, ctypes.POINTER(ctypes.c_ubyte))

        result = CghDisplayShowWindow(self.window_hdl, c_data)
        if result < 0:
            return "Show failed"

        return "Show successfully"

    @setting(2, returns='s')
    def closeSLM(self):
        """
        Close the SLM connection.

        Returns:
            str: Result message.
        """
        self.close_window()
        self.close_device()
        return "SLM connection closed"
    
    @setting(3, returns='s')
    def close_device(self):
        """Close SLM connection."""
        if self.device_hdl:
            EXULUSClose(self.device_hdl)
            self.device_hdl = None

    @setting(4, returns='s')
    def close_window(self):
        """Close SLM window."""
        if self.window_hdl:
            CghDisplayCloseWindow(self.window_hdl)
            self.window_hdl = None

    @setting(5, returns='*s')
    def info(self, verbose=True):
        """
        Discovers all SLMs detected by an SDK.

        Args:
            verbose (bool): Whether to print the discovered information.

        Returns:
            list of str: List of serial numbers or identifiers.
        """
        raise NotImplementedError()
        serial_list = get_serial_list()  # TODO: Fill in proper function.
        if verbose:
            print("Discovered SLMs:", serial_list)
        return serial_list


__server__ = SlmThorExulusHd2()

if __name__ == "__main__":
    from labrad import util
    serial_number = '00429430'  # Replace this with the actual serial number of your SLM
    __server__.initSLM(None, serial_number)
    util.runServer(__server__)
