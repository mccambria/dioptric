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
from utils import widefield
import time
from enum import Enum, IntEnum, auto

"""
Readout modes. Readout mode specifies EM vs conventional, as well as vertical and horizontal 
readout frequencies. Good defaults starred below. Frequencies in Hz

EM amplifier
    * 1; vertical frequency: 2000000; horizontal frequency: 10000000
      2; vertical frequency: 3333000; horizontal frequency: 10000000
      3; vertical frequency: 1000000; horizontal frequency: 10000000
      4; vertical frequency:  200000; horizontal frequency: 10000000
     16; vertical frequency: 2000000; horizontal frequency: 20000000
     17; vertical frequency: 3333000; horizontal frequency: 20000000
     18; vertical frequency: 1000000; horizontal frequency: 20000000
     19; vertical frequency:  200000; horizontal frequency: 20000000
Conventional amplifier
     13; vertical frequency: 1000000; horizontal frequency:   100000
     14; vertical frequency: 2000000; horizontal frequency:   100000
     15; vertical frequency: 3333000; horizontal frequency:   100000
      9; vertical frequency: 1000000; horizontal frequency:  1000000
     10; vertical frequency: 3333000; horizontal frequency:  1000000
     11; vertical frequency: 2000000; horizontal frequency:  1000000
     12; vertical frequency:  200000; horizontal frequency:  1000000
      5; vertical frequency: 1000000; horizontal frequency:  3333000
    * 6; vertical frequency: 3333000; horizontal frequency:  3333000
      7; vertical frequency: 2000000; horizontal frequency:  3333000
      8; vertical frequency:  200000; horizontal frequency:  3333000
"""

# Each readout mode has a specific k gain associated with it. See certificate
# of conformity for details
k_gain_dict = {
    # EM
    1: 20.761,
    2: 20.761,
    3: 20.761,
    4: 20.761,
    16: 23.917,
    17: 23.917,
    18: 23.917,
    19: 23.917,
    # Conventional
    13: 4.574,
    14: 4.574,
    15: 4.574,
    9: 3.809,
    10: 3.809,
    11: 3.809,
    12: 3.809,
    5: 4.357,
    6: 4.357,
    7: 4.357,
    8: 4.357,
}


class CameraNuvuHnu512gamma(LabradServer):
    name = "camera_NUVU_hnu512gamma"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)

        # For readability keep the C stuff in the nuvu_camera folder. To allow this file
        # to be easily imported in post-processing contexts, only import the C stuff if
        # we're actually initializating the server
        from servers.inputs.nuvu_camera.nc_camera import NcCamera
        from servers.inputs.nuvu_camera.defines import TriggerMode, ProcessingType

        # Instantiate the software camera and connect to the hardware camera
        self.cam = NcCamera()
        self.cam.connect()  # Assumes there's just one camera available
        # self.cam.set_heartbeat(int(10e3))

        temp = widefield._get_camera_temp()
        self.cam.set_target_detector_temp(temp)

        # See readout modes block comment above
        self.cam.set_readout_mode(1)
        em_gain = widefield._get_camera_em_gain()
        self.cam.setCalibratedEmGain(em_gain)

        self.cam.set_processing_type(ProcessingType.BIAS_SUBTRACTION)
        self.cam.update_bias()
        self.cam.set_trigger_mode(TriggerMode.EXT_LOW_HIGH_EXP)
        # self.cam.set_timeout(-1)
        timeout = widefield._get_camera_timeout()
        self.cam.set_timeout(timeout)
        self.cam.get_size()
        # self.cam.set_buffer_count(1000)
        # logging.info(self.cam.get_dynamic_buffer_count())
        # waiting_time = self.get_waiting_time()
        # logging.info(f"Waiting time: {waiting_time}")
        exposure_time = self.get_exposure_time()
        logging.info(f"Exposure time: {exposure_time}")
        logging.info("Init complete")

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
        self._clear_buffer()

    @setting(2, returns="y")
    def read(self, c):
        """For efficiency, the int-type numpy array returned by read will be sent over LabRAD
        as a byte string, which then must be reconstructed on the client. There's a function
        for this in the widefield library:

        img_str = camera.read()
        img_array = widefield.img_str_to_array(img_str)
        """
        # start = time.time()
        # img_str = self.cam.read()
        # stop = time.time()
        # logging.info(f"self.cam.read(): {stop-start}")
        # return img_str

        # img_str, read_time, proc_time = self.cam.read()
        # logging.info(f"_read: {read_time}")
        # logging.info(f"processing: {proc_time}")
        # return img_str

        return self.cam.read()

    @setting(5)
    def reset(self, c):
        self.disarm(c)
        self._clear_buffer()

    @setting(8, readout_mode="i")
    def set_readout_mode(self, c, readout_mode):
        """Set the camera's readout mode, including amplifier and vertical/horizontal
        frequencies.

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

    def get_waiting_time(self):
        return self.cam.getWaitingTime()


__server__ = CameraNuvuHnu512gamma()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
