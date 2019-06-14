# -*- coding: utf-8 -*-
"""
Output server for the arbitrary waveform generator.

Created on Wed Apr 10 12:53:38 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = arbitrary_waveform_generator
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


class ArbitraryWaveformGenerator(LabradServer):
    name = 'arbitrary_waveform_generator'

    def initServer(self):
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd('Config')
        p.get('arb_wave_gen_visa_address')
        result = await p.send()
        return result['get']

    def on_get_config(self, config):
        resource_manager = visa.ResourceManager()
        self.wave_gen = resource_manager.open_resource(config)
        
    @setting(4)
    def test_sin(self, c):
        for chan in [1, 2]:
            source_name = 'SOUR{}:'.format(chan)
            self.wave_gen.write('{}FUNC SIN'.format(source_name))
            self.wave_gen.write('{}FREQ 10000'.format(source_name))
            self.wave_gen.write('{}VOLT:HIGH +2.0'.format(source_name))
            self.wave_gen.write('{}VOLT:LOW 0.0'.format(source_name))
        self.wave_gen.write('OUTP1 ON')
        self.wave_gen.write('SOUR2:PHAS 90')
        self.wave_gen.write('OUTP2 ON')
        
    @setting(5)
    def wave_off(self, c):
        self.wave_gen.write('OUTP1 OFF')
        self.wave_gen.write('OUTP2 OFF')


__server__ = ArbitraryWaveformGenerator()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
