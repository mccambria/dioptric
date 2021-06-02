# -*- coding: utf-8 -*-
"""
Output server for the attodry800 cryostat. 

Created on Tue Dec 29 2020

@author: mccambria

### BEGIN NODE INFO
[info]
name = cryostat
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
import socket
import logging
import cryostat_dll.attoDRYLib as attoDRYLib


class Cryostat(LabradServer):
    name = 'cryostat'
    pc_name = socket.gethostname()
    logging.basicConfig(level=logging.DEBUG, 
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%y-%m-%d_%H-%M-%S',
                filename='E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log'.format(pc_name, name))


    def initServer(self):
        msg = attoDRYLib.load()
        logging.debug(msg)
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)


    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(['', 'Config'])
        p.get('cryostat_address')
        result = await p.send()
        return result['get']


    def on_get_config(self, com_port):
        logging.debug(com_port)
        # Done
        logging.debug('Init complete')


    @setting(2, pos_in_steps='i')
    def write_z(self, c, pos_in_steps):
        """
        Specify the absolute position in steps relative to 0. There will be 
        hysteresis on this value, but it's repeatable enough for the 
        common and important routines (eg optimize)
        """
        
        self.write_ax(pos_in_steps, 3)


        


__server__ = Cryostat()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
