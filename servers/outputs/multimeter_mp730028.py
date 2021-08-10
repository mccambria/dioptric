# -*- coding: utf-8 -*-
"""
Output server for Multicomp Pro's 5.5 digit benchtop multimeter

Created on August 10th, 2021

@author: mccambria

### BEGIN NODE INFO
[info]
name = multimeter_mp730028
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


from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
import logging
import socket
import visa


class MultimeterMp730028(LabradServer):
    name = "multimeter_mp730028"
    pc_name = socket.gethostname()
    reset_cfm_opt_out = True

    def initServer(self):
        filename = (
            "E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log"
        )
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%y-%m-%d_%H-%M-%S",
            filename=filename,
        )
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(["", "Config", "DeviceIDs"])
        p.get("{}_visa_address".format(self.name))
        result = await p.send()
        return result["get"]

    def on_get_config(self, config):
        # Note that this instrument works with pyvisa's default
        # termination assumptions
        resource_manager = visa.ResourceManager()
        visa_address = config
        self.multimeter = resource_manager.open_resource(visa_address)
        logging.debug(self.multimeter)
        self.multimeter.write("*RST")
        logging.debug("Init complete")

    @setting(0)
    def meas_resistance(self, c):
        return self.multimeter.query("")

    @setting(6)
    def reset(self, c):
        pass


__server__ = MultimeterMp730028()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
