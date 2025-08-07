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


class MultimeterKeitDaq6510(LabradServer):
    name = "multimeter_KEIT_daq6510"
    pc_name = socket.gethostname()

    def initServer(self):
        ### Logging

        filename = (
            "E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log"
        )
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%y-%m-%d_%H-%M-%S",
            filename=filename,
        )

        ### Configure

        self.measuring_temp = False
        config = common.get_config_dict()

        resource_manager = visa.ResourceManager()
        visa_address = config["DeviceIDs"][f"{self.name}_visa_address"]
        self.multimeter = resource_manager.open_resource(visa_address)
        self.multimeter.baud_rate = 115200
        self.multimeter.read_termination = "\n"
        self.multimeter.write_termination = "\n"
        self.multimeter.write("*RST")
        # test = self.multimeter.query("*IDN?")
        # logging.info(test)
        logging.info("Init complete")

    @setting(1, res_range="s", four_wire="b")
    def config_res_measurement(self, c, res_range, four_wire=False):
        if four_wire:
            meas_name = "FRES"
        else:
            meas_name = "RES"
        self.multimeter.write(f'SENS:FUNC "{meas_name}"')
        # res_range_options = ["500", "5E3", "50E3", "500E3", "5E6", "50E6", "500E6"]
        cmd = f"CONF:SCAL:{meas_name} {res_range}"
        self.multimeter.write(cmd)
        # Set the update rate to fast (maximum speed)
        self.multimeter.write("RATE F")
        self.measuring_temp = False
        # Query the device until it finishes setting up and starts
        # returning valid data
        start = time.time()
        while True:
            time.sleep(0.25)
            if self.measure_internal() > 0:
                break
            if time.time() - start > 5:
                raise RuntimeError(
                    "multimeter_mp730028 timed out configuring resistance measurement."
                )

    @setting(2, detector="s")
    def config_temp_measurement(self, c, detector):
        """There is an option to measure temperature directly on this
        multimeter but it's buggy. In particular, the measurement
        stops returning values much outside room temperature if the unit is
        left on for several days. By monitioring the resistance directly
        we avoid this problem."""

        detector_ranges = {"PT100": "500"}
        self.config_res_measurement(c, detector_ranges[detector], four_wire=True)
        self.measuring_temp = True
        self.detector = detector

    def convert_res_to_temp(self, value):
        # From this Texas Instruments whitepaper: https://www.ti.com/lit/an/sbaa275/sbaa275.pdf?ts=1630965124219&ref_url=https%253A%252F%252Fwww.google.com%252F
        # We have R(T) = 100 (1 + (3.9083E-3 T) + (-5.775E-7 T**2)) in C
        # Inverted this gives:
        if self.detector == "PT100":
            return 3656.96 - 0.287154 * numpy.sqrt(159861899 - 210000 * value)

    def measure_internal(self):
        value = self.multimeter.query("MEAS1?")
        if value == "":
            logging.info("Read blank string in measure_internal!")
            while value == "":
                time.sleep(0.1)
                value = self.multimeter.query("MEAS1?")
            logging.info("Recovered!")
        return float(value)

    @setting(5, returns="v[]")
    def measure(self, c):
        """Return the value from the main display."""
        value = self.measure_internal()
        if self.measuring_temp:
            return self.convert_res_to_temp(value)
        else:
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
