# -*- coding: utf-8 -*-
"""
Timing server for the Swabian Pulse Stream 8/2.

Created on Tue Apr  9 17:12:27 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = Pulse Streamer
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
from pulsestreamer import PulseStreamer as Pulser
from pulsestreamer import TriggerStart
from pulsestreamer import TriggerRearm
from pulsestreamer import OutputState
import importlib
import os
import sys


class PulseStreamer(LabradServer):
    name = 'Pulse Streamer'

    def initServer(self):
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)
        self.seq = None
        self.loaded_seq_streamed = False

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd('Config')
        p.get('pulse_streamer_ip')
        p.get('sequence_library_path')
        p.cd(['Wiring', 'Pulser'])
        p.dir()
        result = await p.send()
        return result

    def on_get_config(self, config):
        get_result = config['get']
        self.pulser = Pulser(get_result[0])
        sys.path.append(get_result[1])
        reg_keys = config['dir'][1]  # dir returns subdirs followed by keys
        wiring = ensureDeferred(self.get_wiring(reg_keys))
        wiring.addCallback(self.on_get_wiring, reg_keys)

    async def get_wiring(self, reg_keys):
        p = self.client.registry.packet()
        for reg_key in reg_keys:
            p.get(reg_key, key=reg_key)  # Return as a dictionary
        result = await p.send()
        return result

    def on_get_wiring(self, wiring, reg_keys):
        self.pulser_wiring = {}
        for reg_key in reg_keys:
            self.pulser_wiring[reg_key] = wiring[reg_key]
        # The default output state is to have the AOM on
        pulser_do_aom = wiring['do_aom']
        self.default_output_state = OutputState([pulser_do_aom], 0, 0)

    def get_seq(self, seq_file, args):
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == '.py':  # py: import as a module
            seq_module = importlib.import_module(file_name)
            seq, period = seq_module.get_seq(self.pulser_wiring, args)
        return seq, period

    @setting(0, seq_file='s', num_repeat='i', args='*?')
    def stream_immediate(self, c, seq_file, num_repeat=1, args=None):
        period = self.stream_load(seq_file, args)
        self.stream_start(num_repeat)
        return period

    @setting(1, seq_file='s', args='*?', returns='i')
    def stream_load(self, c, seq_file, args=None):
        self.pulser.setTrigger(start=TriggerStart.SOFTWARE)
        seq, period = self.get_seq(seq_file, args)
        if seq is not None:
            self.seq = seq
            self.loaded_seq_streamed = False
        return period

    @setting(2, num_repeat='i')
    def stream_start(self, c, num_repeat=1):
        if self.seq == None:
            raise RuntimeError('Stream started with no sequence.')
        if not self.loaded_seq_streamed:
            self.pulser.stream(self.seq, num_repeat, self.default_output_state)
            self.loaded_seq_streamed = True
        self.pulser.startNow()

    @setting(3)
    def constant_default(self, c):
        self.pulser.constant(self.default_output_state)


__server__ = PulseStreamer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
