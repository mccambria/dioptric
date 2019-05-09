# -*- coding: utf-8 -*-
"""
Output server for the PI E709 objective piezo.

Created on Thu Apr  4 15:58:30 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = objective_piezo
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
from pipython import GCSDevice


class ObjectivePiezo(LabradServer):
    name = 'objective_piezo'

    def initServer(self):
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd('Config')
        p.get('objective_piezo_model')
        p.get('gcs_dll_path')
        p.get('objective_piezo_serial')
        result = await p.send()
        return result['get']

    def on_get_config(self, config):
        # Load the generic device
        self.piezo = GCSDevice(devname=config[0], gcsdll=config[1])
        # Connect the specific device with the serial number
        self.piezo.ConnectUSB(config[2])
        # Just one axis for this device
        self.axis = self.piezo.axes[0]

    @setting(0, voltage='v[]')
    def write_voltage(self, c, voltage):
        """Write a voltage to the piezo.

        Params:
            voltage: float
                The voltage to write
        """

        self.piezo.SVO(self.axis, False)  # Turn off the feedback servo
        self.piezo.SVA(self.axis, voltage)  # Write the voltage

    @setting(1, returns='v[]')
    def read_position(self, c):
        """Read the position of the piezo.

        Returns
            float
                The current position of the piezo in microns
        """

        return self.piezo.qPOS()[self.axis]


__server__ = ObjectivePiezo()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
