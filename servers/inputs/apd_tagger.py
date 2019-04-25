# -*- coding: utf-8 -*-
"""
Input server for APDs running into the Time Tagger.

Created on Wed Apr 24 22:07:25 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = apd_tagger
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
import numpy
import TimeTagger


class ApdTagger(LabradServer):
    name = 'apd_tagger'

    def initServer(self):
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd('Config')
        p.get('time_tagger_serial')
        p.cd(['Wiring', 'Tagger'])
        p.dir()
        result = await p.send()
        return result

    def on_get_config(self, config):
        self.tagger = TimeTagger.createTimeTagger(config['get'])
        self.tagger.reset()
        # Determine how many APDs we need to set up
        apd_keys = []
        apd_indices = []
        keys = config['dir'][1]
        for key in keys:
            if key.startswith('di_apd_'):
                apd_keys.append(key)
                apd_indices.append(int(key.split('_')[1]))
        if len(apd_keys) > 0:
            wiring = ensureDeferred(self.get_wiring(apd_keys))
            wiring.addCallback(self.on_get_wiring, apd_indices)

    async def get_wiring(self, apd_keys):
        p = self.client.registry.packet()
        for key in apd_keys:
            p.get(key)
        result = await p.send()
        return result['get']

    def on_get_wiring(self, wiring, apd_indices):
        self.tagger_di_apd = {}
        # Loop through the possible counters
        for loop_index in range(len(apd_indices)):
            apd_index = apd_indices[loop_index]
            self.tagger_di_apd[apd_index] = wiring[loop_index]

    @setting(0, apd_indices='*i')
    def start_tag_stream(apd_indices):
        self.buffer = TimeTagger.TimeTagStreamBuffer()
        buffer_size = int(10**6 / len(apd_indices))  # A million total
        apd_chans = []
        for ind in apd_indices:
            apd_chans.append(self.tagger_di_apd[ind])
        self.stream = TimeTagger.TimeTagStream(tagger, buffer_size, apd_chans)

    @setting(1, returns='*i')
    def read_tag_stream():
        return self.stream.getData(self.buffer)


__server__ = ApdTagger()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
