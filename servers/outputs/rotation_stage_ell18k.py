# -*- coding: utf-8 -*-
"""
Output server for the Thorlabs ELL9K filter slider.

Created on Thu Apr  4 15:58:30 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = rotation_stage_ell18k
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


class RotationStageEll18k(LabradServer):
    name = 'rotation_stage_ell18k'
    logging.basicConfig(level=logging.DEBUG, 
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%y-%m-%d_%H-%M-%S',
                filename='E:/Shared drives/Kolkowitz Lab Group/nvdata/labrad_logging/{}.log'.format(name))

    def initServer(self):
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd('Config')
        p.get('filter_slider_ell9k_address')
        p.cd('FilterSliderEll9kFilterMapping')
        p.dir()
        result = await p.send()
        return result

    def on_get_config(self, config):
        # Get the slider
        try:
            self.stage = serial.Serial(config['get'], 9600, serial.EIGHTBITS,
                                serial.PARITY_NONE, serial.STOPBITS_ONE)
        except Exception as e:
            logging.debug(e)
            del self.stage
        time.sleep(0.1)
        self.stage.flush()
        time.sleep(0.1)
        logging.debug('Init complete')

    @setting(0, angle='v[]')
    def set_angle(self, c, angle):
        # Convert the angle to a command - angles are digitized at a
        # resolution of 143360/rev
        digi_angle = angle * (143360 / 360)
        cmd = '0ma{:08X}'.format(digi_angle)
        self.stage.write(cmd)


__server__ = RotationStageEll18k()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
