# -*- coding: utf-8 -*-
"""
Output server for the Thorlabs ELL9K filter slider.

Created on Thu Apr  4 15:58:30 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = rotation_stage_THOR_ell18k
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
from twisted.internet.defer import ensureDeferred
import serial
import time
import logging
import socket
from utils import common
from utils import tool_belt as tb


class RotationStageThorEll18k(LabradServer):
    name = "rotation_stage_THOR_ell18k"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        config = common.get_config_dict()
        device_id = config["DeviceIDs"][f"{self.name}_com"]
        # Get the slider
        try:
            self.stage = serial.Serial(
                device_id,
                9600,
                serial.EIGHTBITS,
                serial.PARITY_NONE,
                serial.STOPBITS_ONE,
            )
        except Exception as e:
            logging.debug(e)
            del self.stage
        time.sleep(0.1)
        self.stage.flush()
        time.sleep(0.1)
        logging.debug("Init complete")

    @setting(0, returns="v[]")
    def get_angle(self, c):
        return self._get_angle()

    def _get_angle(self):
        self.stage.write("0gp".encode())
        ret = self.stage.readline().decode()
        # First 3 characters are header so skip those
        digi_angle_hex = ret[3:]
        digi_angle = int(digi_angle_hex, 16)
        # If we get a giant value, then the device has just overshot 0 -
        # we can consider this as a negative value and take two's complement
        if digi_angle > int("F0000000", 16):
            digi_angle -= 2**32
        angle = digi_angle * (360 / 143360)
        return angle

    @setting(1, angle="v[]")
    def set_angle(self, c, angle):
        current_angle = self._get_angle()
        # logging.info(current_angle)
        # logging.info(angle)
        # Max speed is 430 deg/s
        min_response_time = abs(angle - current_angle) / 430
        # Convert the angle to a command - angles are digitized at a
        # resolution of 143360/rev
        digi_angle = int(angle * (143360 / 360))
        cmd = "0ma{:08X}".format(digi_angle)
        cmd = cmd.encode()  # Convert to bytes
        self.stage.write(cmd)
        # Always read after every write to clear the buffer
        self.stage.readline()
        # Allow double the minimum possible response time and tack on another
        #         tenth of a second for rise and fall
        time.sleep((2 * min_response_time) + 0.1)


__server__ = RotationStageThorEll18k()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
