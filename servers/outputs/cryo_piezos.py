# -*- coding: utf-8 -*-
"""
Output server for the attodry800's piezos.

Created on Tue Dec 29 2020

@author: mccambria

### BEGIN NODE INFO
[info]
name = cryo_z_piezo
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

# telnetlib is a package for connecting to networked device over the telnet
# protocol. See the ANC150 section of the cryostat manual for more details
from telnetlib import Telnet  

import numpy
import logging


class CryoPiezos(LabradServer):
    name = 'cryo_piezos'
    logging.basicConfig(level=logging.DEBUG, 
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%y-%m-%d_%H-%M-%S',
                filename='E:/Shared drives/Kolkowitz Lab Group/nvdata/labrad_logging/{}.log'.format(name))


    def initServer(self):
        self.task = None
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)


    async def get_config(self):
        p = self.client.registry.packet()
        p.cd('Config')
        p.get('cryo_piezos_address')
        result = await p.send()
        return result


    def on_get_config(self, config):
        ip_address = config['get']
        # Connect via telnet
        try:
            self.piezos = Telnet(ip_address, 7230)
        except Exception as e:
            logging.debug(e)
            del self.piezos
        logging.debug('Init complete')


    @setting(2, voltage='v[]')
    def write_z(self, c, voltage):
        """"""


    @setting(1, returns='v[]')
    def read_z(self, c):
        """"""
        voltage = 1/0
        return voltage
    
    
    @setting(3, center='v[]', scan_range='v[]',
             num_steps='i', period='i', returns='*v[]')
    def load_z_scan(self, c, center, scan_range, num_steps, period):
        """Load a linear sweep with the DAQ"""

        half_scan_range = scan_range / 2
        low = center - half_scan_range
        high = center + half_scan_range
        voltages = numpy.linspace(low, high, num_steps)
        self.load_stream_writer(c, 'ObjectivePiezo-load_z_scan',
                                voltages, period)
        return voltages


__server__ = CryoPiezos()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
