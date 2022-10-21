# -*- coding: utf-8 -*-
"""
Output server for the attodry800 cryostat. 

Created on Tue Dec 29 2020

@author: mccambria

### BEGIN NODE INFO
[info]
name = cryostat_attodry800
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


class CryostatAttodry800(LabradServer):
    name = 'cryostat_attodry800'
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


    @setting(1)
    def activate_temp_control(self, c):
        """
        
        """
        
        pass
    
    
    @setting(2)
    def deactivate_temp_control(self, c):
        """
        
        """
        
        pass
    
    
    @setting(3, temp="v[]")
    def set_temp(self, c, temp):
        """
        
        """
        
        pass
    
    
    @setting(4, returns="v[]")
    def get_temp(self, c, temp):
        """
        
        """
        
        pass
    
        
    @setting(6)
    def reset_cfm_opt_out(self, c):
        """This setting is just a flag for the client. If you include this 
        setting on a server, then the server won't be reset along with the 
        rest of the instruments when we call tool_belt.reset_cfm.
        """
        pass

        


__server__ = CryostatAttodry800()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)