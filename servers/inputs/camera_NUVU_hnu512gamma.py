# -*- coding: utf-8 -*-
"""
Input server for the "HNu 512 Gamma" EMCCD camera from NUVU.

Created on July 31st, 2023

@author: mccambria

### BEGIN NODE INFO
[info]
name = camera_NUVU_hnu512gamma
version = 1.0
description =
[startup]
cmdline = %PYTHON% %FILE%
timeout = 60
[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""


from labrad.server import LabradServer
from labrad.server import setting
from utils import common
from utils import tool_belt as tb
import numpy as np
import socket
import logging

# Keep the C stuff in the nuvu_camera folder - for simplicity, don't put any in this file
from servers.inputs.nuvu_camera.nc_camera import NcCamera
from servers.inputs.nuvu_camera.defines import TriggerMode, ReadoutMode, ProcessingType


class CameraNuvuHnu512gamma(LabradServer):
    name = "camera_NUVU_hnu512gamma"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)

        # Instantiate the software camera and connect to the hardware camera
        self.cam = NcCamera()
        self.cam.connect()  # Assumes there's just one camera available

        # Configure the camera
        self.cam.set_target_detector_temp(-60)
        self.cam.set_readout_mode(ReadoutMode.EM)
        self.cam.set_processing_type(ProcessingType.BACKGROUND_SUBTRACTION)
        self.cam.update_bias()
        self.cam.set_trigger_mode(TriggerMode.EXT_LOW_HIGH_EXP)
        self.cam.set_timeout(-1)
        self.cam.get_size()

    def stopServer(self):
        self.reset(None)
        self.cam.disconnect()

    @setting(6)
    def get_detector_temp(self, c):
        return self.cam.get_detector_temp()

    @setting(0, num_images="i")
    def arm(self, c, num_images=0):
        self.cam.open_shutter()
        self.cam.start(num_images)

    @setting(1)
    def disarm(self, c):
        self.cam.stop()
        self.cam.close_shutter()

    @setting(3)
    def disarm(self, c):
        self.cam.stop()
        self.cam.close_shutter()

    @setting(2, returns="*2i")
    def read(self, c):
        img_array = self.cam.read()
        return img_array

    @setting(5)
    def reset(self, c):
        self.disarm(c)


__server__ = CameraNuvuHnu512gamma()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
