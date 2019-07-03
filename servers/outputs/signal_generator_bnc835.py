# -*- coding: utf-8 -*-
"""
Output server for the Berkeley Nucleonics 835 microwave signal generator.

Created on Wed Apr 10 12:53:38 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = signal_generator_bnc835
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
import visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
import logging


class SignalGeneratorBnc835(LabradServer):
    name = 'signal_generator_bnc835'
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
        p.get('signal_generator_bnc835_visa_address')
        result = await p.send()
        return result['get']

    def on_get_config(self, visa_address):
        resource_manager = visa.ResourceManager()
        self.sig_gen = resource_manager.open_resource(visa_address)
        # Note that this instrument works with pyvisa's default
        # termination assumptions
        self.reset(None)
        logging.debug('init complete')

    @setting(0)
    def uwave_on(self, c):
        """Turn on the signal. This is like opening an internal gate on
        the signal generator.
        """

        self.sig_gen.write('ENBR 1')

    @setting(1)
    def uwave_off(self, c):
        """Turn off the signal. This is like closing an internal gate on
        the signal generator.
        """

        self.sig_gen.write('ENBR 0')

    @setting(2, freq='v[]')
    def set_freq(self, c, freq):
        """Set the frequency of the signal.

        Params
            freq: float
                The frequency of the signal in GHz
        """

        # Determine how many decimal places we need
        precision = len(str(freq).split('.')[1])
        self.sig_gen.write('FREQ {0:.{1}f}GHZ'.format(freq, precision))

    @setting(3, amp='v[]')
    def set_amp(self, c, amp):
        """Set the amplitude of the signal.

        Params
            amp: float
                The amplitude of the signal in dBm
        """

        # Determine how many decimal places we need
        precision = len(str(amp).split('.')[1])
        self.sig_gen.write('AMPR {0:.{1}f}DBM'.format(amp, precision))

    @setting(6)
    def reset(self, c):
        self.sig_gen.write('FDEV 0')
        self.uwave_off(c)


__server__ = SignalGeneratorBnc835()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
