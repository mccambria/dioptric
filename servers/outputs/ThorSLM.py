# -*- coding: utf-8 -*-
"""
Input server for the "HNu 512 Gamma" EMCCD camera from NUVU.

Created on July 31st, 2023

@author: Saroj B Chand

### BEGIN NODE INFO
[info]
name = ThorSLM
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

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

# sys.path.append('c:/Users/Saroj Chand/Documents/dioptric')
from utils import tool_belt as tb
from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.Thorlabs_EXULUS_PythonSDK.Thorlabs_EXULUS_Python_SDK.EXULUS_COMMAND_LIB import *
from slmsuite.hardware.Thorlabs_EXULUS_PythonSDK.Thorlabs_EXULUS_CGHDisplay.Thorlabs_EXULUS_CGHDisplay import *

DEFAULT_SDK_PATH = "C:/Users/Saroj Chand/Documents/dioptric/Thorlabs_EXULUS_PythonSDK"


class ThorSLMServer(LabradServer):
    name = "ThorSLMServer"
    pc_name = socket.gethostname()

    def initServer(self):
        """
        Initializes the ThorSLMServer.
        """
        tb.configure_logging(self)
        self.serialNumber = None
        self.device_hdl = None
        self.window_hdl = None
        logging.info("ThorSLMServer initialized")

    # @setting(1, serialNumber='s', returns='s')
    def initSLM(self, c, serialNumber):
        """
        Initializes an instance of a Thorabs SLM.

        Args:
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

    @setting(2, phase='y', returns='s')
    def setPhase(self, c, phase):
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

    @setting(3, returns='s')
    def closeSLM(self, c):
        """
        Close the SLM connection.

        Returns:
            str: Result message.
        """
        self.close_window()
        self.close_device()
        return "SLM connection closed"

    def close_device(self):
        """Close SLM connection."""
        if self.device_hdl:
            EXULUSClose(self.device_hdl)
            self.device_hdl = None

    def close_window(self):
        """Close SLM window."""
        if self.window_hdl:
            CghDisplayCloseWindow(self.window_hdl)
            self.window_hdl = None

    @setting(4, verbose='b', returns='*s')
    def info(self, c, verbose=True):
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


__server__ = ThorSLMServer()

if __name__ == "__main__":
    util.runServer(__server__)
