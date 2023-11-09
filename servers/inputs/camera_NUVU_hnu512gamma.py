# -*- coding: utf-8 -*-
"""
Input server for the "HNu 512 Gamma" EMCCD camera from NUVU.

Created on July 31st, 2023

@author: mccambria

### BEGIN NODE INFO
[info]
name = camera_NUVU_hnu512gamma
version = 1.0
description = 120
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
import time

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
        # self.cam.set_heartbeat(int(10e3))

        # Configure the camera
        self.cam.set_target_detector_temp(-60)

        # self.cam.set_readout_mode(6)

        self.cam.set_readout_mode(1)
        self.cam.setCalibratedEmGain(100)
        # self.cam.setCalibratedEmGain(10)

        self.cam.set_processing_type(ProcessingType.BACKGROUND_SUBTRACTION)
        self.cam.update_bias()
        self.cam.set_trigger_mode(TriggerMode.EXT_LOW_HIGH_EXP)
        # self.cam.set_timeout(-1)
        self.cam.set_timeout(1000)
        self.cam.get_size()
        # self.cam.set_buffer_count(1000)
        # logging.info(self.cam.get_dynamic_buffer_count())

    def stopServer(self):
        self.reset(None)
        self.cam.disconnect()

    @setting(9)
    def clear_buffer(self, c):
        self._clear_buffer()

    def _clear_buffer(self):
        self.cam.flushReadQueue()

    @setting(6)
    def get_detector_temp(self, c):
        return self.cam.get_detector_temp()

    @setting(0, num_images="i")
    def arm(self, c, num_images=0):
        self._clear_buffer()
        self.cam.open_shutter()
        self.cam.start(num_images)

    @setting(1)
    def disarm(self, c):
        self.cam.stop()
        self.cam.close_shutter()

    # @setting(2, returns="*2i")
    @setting(2, returns="y")
    def read(self, c):
        """For efficiency, the int-type numpy array returned by read will be sent over LabRAD
        as a byte string, which then must be reconstructed on the client. There's a function
        for this in the widefield library:

        img_str = camera.read()
        img_array = widefield.img_str_to_array(img_str)
        """
        return self.cam.read()

    @setting(5)
    def reset(self, c):
        self.disarm(c)
        self._clear_buffer()

    @setting(8, readout_mode="i")
    def set_readout_mode(self, c, readout_mode):
        """
        Set the camera's readout mode, including amplifier and vertical/horizontal frequencies.
        Good defaults starred below

        readout_mode options:
            EM amplifier
                *Mode:  1; vertical frequency: 2000000; horizontal frequency: 10000000
                 Mode:  2; vertical frequency: 3333000; horizontal frequency: 10000000
                 Mode:  3; vertical frequency: 1000000; horizontal frequency: 10000000
                 Mode:  4; vertical frequency:  200000; horizontal frequency: 10000000
                 Mode: 16; vertical frequency: 2000000; horizontal frequency: 20000000
                 Mode: 17; vertical frequency: 3333000; horizontal frequency: 20000000
                 Mode: 18; vertical frequency: 1000000; horizontal frequency: 20000000
                 Mode: 19; vertical frequency:  200000; horizontal frequency: 20000000
            Conventional amplifier
                 Mode:  5; vertical frequency: 1000000; horizontal frequency:  3333000
                *Mode:  6; vertical frequency: 3333000; horizontal frequency:  3333000
                 Mode:  7; vertical frequency: 2000000; horizontal frequency:  3333000
                 Mode:  8; vertical frequency:  200000; horizontal frequency:  3333000
                 Mode:  9; vertical frequency: 1000000; horizontal frequency:  1000000
                 Mode: 10; vertical frequency: 3333000; horizontal frequency:  1000000
                 Mode: 11; vertical frequency: 2000000; horizontal frequency:  1000000
                 Mode: 12; vertical frequency:  200000; horizontal frequency:  1000000
                 Mode: 13; vertical frequency: 1000000; horizontal frequency:   100000
                 Mode: 14; vertical frequency: 2000000; horizontal frequency:   100000
                 Mode: 15; vertical frequency: 3333000; horizontal frequency:   100000
        """
        self.cam.stop()  # Make sure the camera is stopped or else this won't work
        self.cam.set_readout_mode(readout_mode)

    @setting(7, returns="s")
    def get_readout_mode(self, c):
        readout_mode = self.cam.get_readout_mode()
        return readout_mode

    @setting(10, returns="i")
    def get_num_readout_modes(self, c):
        return self.cam.get_num_readout_modes()


__server__ = CameraNuvuHnu512gamma()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
