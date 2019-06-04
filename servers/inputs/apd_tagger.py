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
import TimeTagger
import numpy


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
                apd_indices.append(int(key.split('_')[2]))
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
        # Create an inversion of tagger_di_apd to pass back to the client
        self.inverted_tagger_di_apd = {}
        # Loop through the possible counters
        for loop_index in range(len(apd_indices)):
            apd_index = apd_indices[loop_index]
            wire = wiring[loop_index]
            self.tagger_di_apd[apd_index] = wire
            self.inverted_tagger_di_apd[wire] = apd_index

    def read_raw_stream():
        buffer = self.stream.getData()
        timestamps = buffer.getTimestamps()
        channels = buffer.getChannels()
        return timestamps, channels

    @setting(0, apd_indices='*i')
    def start_tag_stream(self, c, apd_indices):
        """Expose a raw tag stream which can be read with read_tag_stream and
        closed with stop_tag_stream.
        """
        buffer_size = int(10**6 / len(apd_indices))  # A million total
        channels = []
        for ind in apd_indices:
            channels.append(self.tagger_di_apd[ind])
            channels.append(self.tagger_di_apd_gate[ind])
        channels.append(self.tagger_di_clock)
        self.stream = TimeTagger.TimeTagStream(self.tagger, buffer_size, apd_chans)

    @setting(1, returns='*s*i')
    def read_tag_stream(self, c):
        """Read the stream started with start_tag_stream. Returns two lists,
        each as long as the number of counts that have occurred since the
        buffer was refreshed. First list is timestamps in ps, second is apd
        indices.
        """
        timestamps, channels = self.read_raw_stream()
        # Convert timestamps to strings since labrad does not support int64s
        timestamps = buffer.getTimestamps().astype(str).tolist()
        # Convert channels to APD indices
        indices = list(map(lambda x: self.inverted_tagger_di_apd[x], channels))
        return timestamps, indices

    @setting(2)
    def stop_tag_stream(self, c):
        """Closes the stream started with start_tag_stream."""
        self.stream.stop()

    @setting(3, apd_index='i', returns='*w')
    def read_counter(self, c, apd_index):
        timestamps, channels = self.read_raw_stream()

        # There will be 4 channels: APD, clock, gate open, and gate close
        apd_channel = self.tagger_di_apd[apd_index]
        clock_channel = self.tagger_di_clock
        gate_open_channel = self.tagger_di_apd_gate[apd_index]
        gate_close_channel = -gate_open_channel

        # Find clock clicks (sample breaks)
        result = numpy.nonzero(channels == clock_channel)
        clock_click_inds = result[0].aslist()

        previous_sample_end_ind = None

        counts = []

        for clock_click_ind in clock_click_inds:

            # Clock clicks end samples, so they should be included with the
            # sample itself
            sample_end_ind = clock_click_ind + 1

            if previous_sample_end_ind == None:
                sample_timestamps = self.leftover_timestamps
                sample_timestamps.extend(timestamps[0:sample_end_ind])
                sample_channels = self.leftover_channels
                sample_channels.extend(channels[0:sample_end_ind])
            else:
                sample_timestamps = timestamps[previous_sample_end_ind:
                    sample_end_ind]
                sample_channels = channels[previous_sample_end_ind:
                    sample_end_ind]

            # Find gate open clicks
            result = numpy.nonzero(sample_channels == gate_open_channel)
            gate_open_click_inds = result[0].aslist()

            # Find gate close clicks
            # Gate close channel is negative of gate open channel,
            # signifying the falling edge
            result = numpy.nonzero(sample_channels == gate_close_channel)
            gate_close_click_inds = result[0].aslist()

            # The number of APD clicks is simply the number of items in the
            # buffer between gate open and gate close clicks
            count = 0
            for ind in range(len(gate_open_click_inds)):
                gate_open_click_ind = gate_open_click_inds[ind]
                gate_close_click_ind = gate_close_click_inds[ind]
                count += sample_channels.count(apd_channel)
            counts.extend(count)

            previous_sample_end_ind = sample_end_ind

        return counts

__server__ = ApdTagger()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
