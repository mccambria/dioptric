# -*- coding: utf-8 -*-
"""
Output server for Multicomp Pro's 0-60 V, 0-3 A benchtop linear power supply

Created on August 10th, 2021

@author: mccambria

### BEGIN NODE INFO
[info]
name = power_supply_mp710087
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
import visa  # Docs here: https://pyvisa.readthedocs.io/en/master/


class PowerSupplyMp710087(LabradServer):
    name = "power_supply_mp710087"
    pc_name = socket.gethostname()

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
        p.get("power_supply_mp710087_visa_address")
        result = await p.send()
        return result["get"]

    def on_get_config(self, config):
        # Note that this instrument works with pyvisa's default
        # termination assumptions
        resource_manager = visa.ResourceManager()
        visa_address = config
        self.power_supply = resource_manager.open_resource(visa_address)
        logging.debug(self.power_supply)
        self.power_supply.write("*RST")
        logging.debug("Init complete")


__server__ = PowerSupplyMp710087()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
