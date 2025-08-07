# -*- coding: utf-8 -*-
"""
Input server for a Keithley DAQ6510 benchtop multimeter

Created on August 7th, 2025

@author: mccambria

### BEGIN NODE INFO
[info]
name = multimeter_KEIT_daq6510
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

import logging
import socket
import time

import numpy
import pyvisa as visa
from labrad.server import LabradServer, setting
from twisted.internet.defer import ensureDeferred

from utils import common
from utils import tool_belt as tb


class MultimeterKeitDaq6510(LabradServer):
    name = "multimeter_KEIT_daq6510"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        config = common.get_config_dict()

        resource_manager = visa.ResourceManager()
        visa_address = config["DeviceIDs"][f"{self.name}_visa"]
        # logging.info(visa_address)
        self.multimeter = resource_manager.open_resource(visa_address)
        self.multimeter.read_termination = "\n"
        self.multimeter.write_termination = "\n"
        self.multimeter.write("*RST")
        logging.info("Init complete")

    # def measure_internal(self):
    #     value = self.multimeter.query("MEAS1?")
    #     if value == "":
    #         logging.info("Read blank string in measure_internal!")
    #         while value == "":
    #             time.sleep(0.1)
    #             value = self.multimeter.query("MEAS1?")
    #         logging.info("Recovered!")
    #     return float(value)

    @setting(5, returns="v[]")
    def read(self, c):
        """Return the value from the main display."""
        value = self.multimeter.query(":MEASure:VOLTage:DC?")
        return value

    @setting(6)
    def reset_cfm_opt_out(self, c):
        """This setting is just a flag for the client. If you include this
        setting on a server, then the server won't be reset along with the
        rest of the instruments when we call tool_belt.reset_cfm.
        """
        pass

    @setting(7)
    def reset(self, c):
        """Fully reset to factory defaults"""
        self.multimeter.write("*RST")


__server__ = MultimeterKeitDaq6510()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
