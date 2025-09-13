# -*- coding: utf-8 -*-
"""
Output server for the SRS SR474 shutter controller

Created on August 5th, 2025

@author: mccambria

### BEGIN NODE INFO
[info]
name = shutter_STAN_sr474
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

import logging
import socket
import time

import pyvisa as visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
from labrad.server import LabradServer, setting
from twisted.internet.defer import ensureDeferred

from utils import common
from utils import tool_belt as tb


class ShutterStanSr474(LabradServer):
    name = "shutter_STAN_sr474"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        config = common.get_config_dict()
        device_id = config["DeviceIDs"][f"{self.name}_visa"]
        resource_manager = visa.ResourceManager()
        self.shutter = resource_manager.open_resource(device_id)
        # Set the VISA read and write termination. This is specific to the
        # instrument - you can find it in the instrument's programming manual
        self.shutter.read_termination = "\r\n"
        self.shutter.write_termination = "\r\n"
        self.reset(None)
        self.shutter.write("AUXC 0,0")  # Set all channels to manual mode
        logging.info("Init complete")

    @setting(0, channel="i")
    def enable(self, c, channel):
        self.shutter.write(f"ENAB {channel},1")

    @setting(1, channel="i")
    def disable(self, c, channel):
        self.shutter.write(f"ENAB {channel},0")

    @setting(2, channel="i")
    def open(self, c, channel):
        self.write_state(channel, 1)

    @setting(3, channel="i")
    def close(self, c, channel):
        self.write_state(channel, 0)

    def write_state(self, channel, state):
        num_attempts = 0
        max_attempts = 10
        while self.read_state(channel) != state:
            self.shutter.write(f"STAT {channel},{state}")
            num_attempts += 1
            if num_attempts == max_attempts:
                msg = f"Failed to set shutter within {max_attempts} attempts"
                raise RuntimeError(msg)

    def read_state(self, channel):
        """Read the state of the passed channel

        Parameters
        ----------
        channel : int
            Channel number, 1-4

        Returns
        -------
        int
            1 if open, 0 if closed, -1 if indeterminate (e.g. not done switching yet)
        """
        return int(self.shutter.query(f"SPOS? {channel}"))

    @setting(6)
    def reset(self, c):
        self.shutter.write("STAT 0,0")
        self.disable(None, 0)


__server__ = ShutterStanSr474()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
