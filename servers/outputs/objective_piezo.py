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
from pipython import pitools 
import time
import logging


class ObjectivePiezo(LabradServer):
    name = 'objective_piezo'
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
        logging.debug('Init complete')
        
    def waitonoma(self, pidevice, axes=None, timeout=300, predelay=0, postdelay=0):
        """This is ripped from the pitools module and adapted for Python 3
        (ie I changed xrange to range...). It hangs until the device completes
        an absolute open-piezo motion.
        """
        axes = pitools.getaxeslist(pidevice, axes)
        numsamples = 5
        positions = []
        maxtime = time.time() + timeout
        pitools.waitonready(pidevice, timeout, predelay)
        while True:
            positions.append(pidevice.qPOS(axes).values())
            positions = positions[-numsamples:]
            if len(positions) < numsamples:
                continue
            isontarget = True
            for vals in zip(*positions):
                isontarget &= 0.01 > sum([abs(vals[i] - vals[i + 1]) for i in range(len(vals) - 1)])
            if isontarget:
                return
            if time.time() > maxtime:
                pitools.stopall(pidevice)
                raise SystemError('waitonoma() timed out after %.1f seconds' % timeout)
            time.sleep(0.01)
        time.sleep(postdelay)

    @setting(0, voltage='v[]')
    def write_voltage(self, c, voltage):
        """Write a voltage to the piezo.

        Params:
            voltage: float
                The voltage to write
        """

        self.piezo.SVO(self.axis, False)  # Turn off the feedback servo
        self.piezo.SVA(self.axis, voltage)  # Write the voltage
        # Wait until the device completes the motion
        logging.debug('Began motion at {}'.format(time.time()))
#        self.waitonoma(self.piezo, timeout=5)
        try:
            self.waitonoma(self.piezo, timeout=5)
        except Exception as e:
            logging.error(e)
        logging.debug('Completed motion at {}'.format(time.time()))

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
