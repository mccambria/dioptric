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


class CameraNuvuHnu512gamma(LabradServer):
    name = "camera_NUVU_hnu512gamma"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)

        self.cam = NcCamera()
        self.cam.open_cam()  # Assumes there's just one camera available

    @setting(0)
    def get_img_array(self, c):
        return self.cam.get_img_array()

    @setting(5)
    def reset(self, c):
        self.cam.close_cam()


__server__ = CameraNuvuHnu512gamma()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
