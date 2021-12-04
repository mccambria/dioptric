# -*- coding: utf-8 -*-
"""
Output server for the Thorlabs ELL9K filter slider.

Created on Thu Apr  4 15:58:30 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = filter_slider_ell9k
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


class FilterSliderEll9k(LabradServer):
    name = 'filter_slider_ell9k'
    pc_name = socket.gethostname()

    def initServer(self):
        filename = 'E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log'
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d_%H-%M-%S', filename=filename)
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(['', 'Config', 'DeviceIDs'])
        logging.debug('{}_address'.format(self.name))
        p.get('{}_address'.format(self.name))
        result = await p.send()
        return result

    def on_get_config(self, config):
        # Get the slider
        try:
            self.slider = serial.Serial(config['get'], 9600, serial.EIGHTBITS,
                                serial.PARITY_NONE, serial.STOPBITS_ONE)
        except Exception as e:
            logging.debug(e)
            del self.slider
        time.sleep(0.1)
        self.slider.flush()
        time.sleep(0.1)
        # Set up the mapping from filter position to move command
        self.move_commands = {0: '0ma00000000'.encode(),
                              1: '0ma00000020'.encode(),
                              2: '0ma00000040'.encode(),
                              3: '0ma00000060'.encode()}
        logging.info('Init complete')
        
    def get_status(self):
        cmd = "0gs".encode()
        self.slider.write(cmd)
        res = self.slider.readline()
        status = int(res.decode().split("0GS")[1])
        return status
    
    def wait_for_move_complete(self):
        # Poll the status until we get the OK indicating we're done moving
        status = None
        while status != 0:
            status = self.get_status()
            logging.info(status)
            time.sleep(5)
        
    
    @setting(0, pos='i')
    def set_filter(self, c, pos):
        cmd = self.move_commands[pos]
        incomplete = True
        while incomplete:
            self.slider.write(cmd)
            time.sleep(0.1)
            res = self.slider.readline()
            # The device returns
            incomplete = ("0GS" in res.decode())


__server__ = FilterSliderEll9k()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
