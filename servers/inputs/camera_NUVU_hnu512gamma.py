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
timeout = 20
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

# Keep the C stuff in the nuvu_camera folder - for simplicity, don't put any in this file
from servers.inputs.nuvu_camera.nc_camera import NcCamera
from servers.inputs.nuvu_camera.defines import TriggerMode, ReadoutMode


class CameraNuvuHnu512gamma(LabradServer):
    name = "camera_NUVU_hnu512gamma"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)

        # Instantiate the software camera and connect to the harware camera
        self.cam = NcCamera()
        self.cam.connect()  # Assumes there's just one camera available

        # Configure the camera
        self.cam.set_readout_mode(ReadoutMode.EM)
        self.cam.set_trigger_mode(TriggerMode.CONT_LOW_HIGH)

    def stopServer(self):
        self.cam.disconnect()

    @setting(0)
    def arm(self, c, num_images=0):
        self.cam.open_shutter()
        self.cam.start(num_images)

    @setting(1)
    def disarm(self, c):
        self.cam.stop()
        self.cam.close_shutter()

    @setting(2)
    def read(self, c):
        return self.cam.read()

    @setting(5)
    def reset(self, c):
        self.disarm()


__server__ = CameraNuvuHnu512gamma()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
