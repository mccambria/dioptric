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
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%y-%m-%d %H:%M:%S")

from utils import tool_belt as tb
from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.Thorlabs_EXULUS_PythonSDK.Thorlabs_EXULUS_Python_SDK.EXULUS_COMMAND_LIB import *
from slmsuite.hardware.Thorlabs_EXULUS_PythonSDK.Thorlabs_EXULUS_CGHDisplay.Thorlabs_EXULUS_CGHDisplay import *

DEFAULT_SDK_PATH = "C:/Users/Saroj Chand/Documents/dioptric/Thorlabs_EXULUS_PythonSDK"

class SlmThorExulusHd2(LabradServer, SLM):
    name = "slm_THOR_exulus_hd2"
    pc_name = socket.gethostname()

    def initServer(self):
        """
        Initializes the ThorSLMServer.
        Initializes an instance of a Thorabs SLM.

        Args:
            c (Context): The Labrad context object.
            serialNumber (str): Serial number of the SLM to initialize.

        Returns:
            str: Initialization result message.
        """
        tb.configure_logging(self)
        
        self.serialNumber = None
        self.device_hdl = None
        self.window_hdl = None
        logging.info("init complete")

        # Initialize SLM on server startup
        serialNumber = '00429430'
        logging.info(f"Attempting to connect to SLM with serial number {serialNumber}")
        self.device_hdl = EXULUSOpen(serialNumber, 38400, 3)
        if self.device_hdl < 0:
            logging.error(f"Connect {serialNumber} failed with error code {self.device_hdl}")
            return f"Connect {serialNumber} failed"

        result = EXULUSIsOpen(serialNumber)
        if result < 0:
            logging.error(f"SLM with serial number {serialNumber} is not open, error code {result}")
            return "Open failed"

        width, height = 1920, 1080
        self.window_hdl = CghDisplayCreateWindow(2, width, height, "SLM window")
        if self.window_hdl < 0:
            logging.error(f"Create window failed with error code {self.window_hdl}")
            return "Create window failed"

        result = CghDisplaySetWindowInfo(self.window_hdl, width, height, 1)
        if result < 0:
            logging.error(f"Set Window Info failed with error code {result}")
            return "Set Window Info failed"

        result = CghDisplayShowWindow(self.window_hdl, None)
        if result < 0:
            logging.error(f"Show window failed with error code {result}")
            return "Show window failed"

        self.serialNumber = serialNumber
        logging.info(f"Connected to SLM with serial number {serialNumber} successfully")
        return f"Connect {serialNumber} successfully"

    def is_connected(self):
        """
        Checks if the SLM is connected and operational.

        Returns:
            bool: True if connected, False otherwise.
        """
        if self.device_hdl is None:
            logging.warning("SLM device handle is None, device is not connected")
            return False

        if self.window_hdl is None:
            logging.warning("SLM window handle is None, window is not created")
            return False

        result = EXULUSIsOpen(self.serialNumber)
        if result < 0:
            logging.error(f"SLM with serial number {self.serialNumber} is not open, error code {result}")
            return False

        logging.info("SLM is connected and operational")
        return True

    @setting(1, returns='s')
    def _write_phase(self, phase):
        """
        Low-level hardware interface to write phase data onto the SLM.

        Args:
            phase (bytes): Phase data to set on the SLM.

        Returns:
            str: Result message.
        """
        if not self.is_connected():
            return "SLM not initialized"

        phase_array = np.frombuffer(phase, dtype=np.uint8)
        phase_matrix = phase_array.reshape((1080, 1920))
        flattened_matrix = phase_matrix.flatten()
        c_data = ctypes.cast(flattened_matrix.ctypes.data, ctypes.POINTER(ctypes.c_ubyte))

        result = CghDisplayShowWindow(self.window_hdl, c_data)
        if result < 0:
            logging.error(f"Show failed with error code {result}")
            return "Show failed"

        logging.info("Show successfully")
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
        logging.info("SLM connection closed")
        return "SLM connection closed"
    
    @setting(3, returns='s')
    def close_device(self):
        """Close SLM connection."""
        if self.device_hdl:
            EXULUSClose(self.device_hdl)
            logging.info(f"Closed SLM device with handle {self.device_hdl}")
            self.device_hdl = None

    @setting(4, returns='s')
    def close_window(self):
        """Close SLM window."""
        if self.window_hdl:
            CghDisplayCloseWindow(self.window_hdl)
            logging.info(f"Closed SLM window with handle {self.window_hdl}")
            self.window_hdl = None

    @setting(5, verbose='b', returns='*s')
    def info(self, c, verbose=True):
        """
        Retrieves information about the SLM.

        Args:
            c (Context): The LabRAD context object.
            verbose (bool): Whether to print verbose information (default True).

        Returns:
            list: List of strings containing information about the SLM.
        """
        serial_list = ["00429430"]  # Example serial number
        info = [
            f"PC Name: {self.pc_name}",
            f"SLM Serial Number: {self.serialNumber}",
            f"SLM Device Handle: {self.device_hdl}",
            f"SLM Window Handle: {self.window_hdl}",
        ]
        if verbose:
            for line in info:
                print(line)
        logging.info(f"SLM info: {info}")
        return info

__server__ = SlmThorExulusHd2()

if __name__ == "__main__":
    from labrad import util
    __server__.initServer()
    util.runServer(__server__)
