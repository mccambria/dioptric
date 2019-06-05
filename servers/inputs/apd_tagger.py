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
import logging
import re


class ApdTagger(LabradServer):
    name = 'apd_tagger'
    logging.basicConfig(level=logging.DEBUG, 
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%y-%m-%d_%H-%M-%S',
                filename='E:/Team Drives/Kolkowitz Lab Group/nvdata/labrad_logging/{}.log'.format(name))

    def initServer(self):
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)
        self.stream = None
        self.leftover_timestamps = []
        self.leftover_channels = []

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd('Config')
        p.get('time_tagger_serial')
        p.cd(['Wiring', 'Tagger'])
        p.get('di_clock')
        p.get('di_apd_gate')
        p.dir()
        result = await p.send()
        return result

    def on_get_config(self, config):
        get_result = config['get']
        self.tagger = TimeTagger.createTimeTagger(get_result[0])
        self.tagger.reset()
        self.tagger_di_clock = get_result[1]
        self.tagger_di_apd_gate = get_result[2]
        # Create a mapping from tagger channels to semantic channels
        self.channel_mapping = {self.tagger_di_clock: 'clock',
                                self.tagger_di_apd_gate: 'gate_open',
                                -self.tagger_di_apd_gate: 'gate_close'}
        # Determine how many APDs we need to set up
        apd_keys = []
        apd_indices = []
        keys = config['dir'][1]
        for key in keys:
            # Regular expression for keys of the form di_apd_1
            if re.fullmatch(r'di_apd_[0-9]+', key) is not None:
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
        # Loop through the possible counters
        for loop_index in range(len(apd_indices)):
            apd_index = apd_indices[loop_index]
            channel = wiring[loop_index]
            self.tagger_di_apd[apd_index] = channel
            self.channel_mapping[channel] = 'apd_{}'.format(apd_index)
        logging.debug('init complete')

    def read_raw_stream(self):
        if self.stream is None:
            logging.error('read_raw_stream attempted while stream is None.')
            return
        
        buffer = self.stream.getData()
        timestamps = buffer.getTimestamps()
        channels = buffer.getChannels()
        return timestamps, channels
        
    def read_counter_internal(self, apd_index, num_to_read=None):
        if self.stream is None:
            logging.error('read_counter_internal attempted while stream ' /
                          'is None.')
            return
        
        timestamps, channels = self.read_raw_stream()

        # There will be 4 channels: APD, clock, gate open, and gate close
        apd_channel = self.tagger_di_apd[apd_index]
        clock_channel = self.tagger_di_clock
        gate_open_channel = self.tagger_di_apd_gate
        gate_close_channel = -gate_open_channel

        # Find clock clicks (sample breaks)
        result = numpy.nonzero(channels == clock_channel)
        clock_click_inds = result[0].tolist()

        previous_sample_end_ind = None
        sample_end_ind = None

        # Counts will be a list of lists - the first dimension will divide
        # samples and the second will divide gatings within samples
        counts = []

        for clock_click_ind in clock_click_inds:

            # Clock clicks end samples, so they should be included with the
            # sample itself
            sample_end_ind = clock_click_ind + 1

            if previous_sample_end_ind is None:
                sample_timestamps = self.leftover_timestamps
                sample_timestamps.extend(timestamps[0: sample_end_ind])
                sample_channels = self.leftover_channels
                sample_channels.extend(channels[0: sample_end_ind])
            else:
                sample_timestamps = timestamps[previous_sample_end_ind:
                    sample_end_ind]
                sample_channels = channels[previous_sample_end_ind:
                    sample_end_ind]
                
            # Make sure we've got arrays or else comparison won't produce
            # the boolean array we're looking for when we find gate clicks
            sample_timestamps = numpy.array(sample_timestamps)
            sample_channels = numpy.array(sample_channels)
                    
            # Find gate open clicks
            result = numpy.nonzero(sample_channels == gate_open_channel)
            gate_open_click_inds = result[0].tolist()

            # Find gate close clicks
            # Gate close channel is negative of gate open channel,
            # signifying the falling edge
            result = numpy.nonzero(sample_channels == gate_close_channel)
            gate_close_click_inds = result[0].tolist()

            # The number of APD clicks is simply the number of items in the
            # buffer between gate open and gate close clicks
            sample_counts = []
            for ind in range(len(gate_open_click_inds)):
                gate_open_click_ind = gate_open_click_inds[ind]
                gate_close_click_ind = gate_close_click_inds[ind]
                gate_window = sample_channels[gate_open_click_ind:
                    gate_close_click_ind]
                gate_window = gate_window.tolist()
                gate_counts = gate_window.count(apd_channel)
                sample_counts.append(gate_counts)
            counts.append(sample_counts)

            previous_sample_end_ind = sample_end_ind
        
        if sample_end_ind is not None:
            self.leftover_timestamps = timestamps[sample_end_ind:].tolist()
            self.leftover_channels = channels[sample_end_ind:].tolist()

        return counts
    
    def stop_tag_stream_internal(self):
        if self.stream is None:
            logging.error('stop_tag_stream_internal attempted while stream ' /
                          'is None.')
        else:
            self.stream.stop()
            self.stream = None
        self.leftover_timestamps = []
        self.leftover_channels = []

    @setting(0, apd_indices='*i')
    def start_tag_stream(self, c, apd_indices):
        """Expose a raw tag stream which can be read with read_tag_stream and
        closed with stop_tag_stream.
        """
        if self.stream is not None:
            logging.warning('New stream started before existing stream was ' /
                            'stopped. Stopping existing stream.')
            self.stop_tag_stream_internal()
        # Hardware-limited max buffer is a million total samples
        buffer_size = int(10**6 / len(apd_indices))  
        channels = []
        for ind in apd_indices:
            channels.append(self.tagger_di_apd[ind])
        channels.append(self.tagger_di_apd_gate)  # rising edge
        channels.append(-self.tagger_di_apd_gate)  # falling edge
        channels.append(self.tagger_di_clock)
        self.stream = TimeTagger.TimeTagStream(self.tagger,
                                               buffer_size, channels)
        
    @setting(2)
    def stop_tag_stream(self, c):
        """Closes the stream started with start_tag_stream. Resets
        leftovers.
        """
        self.stop_tag_stream_internal()

    @setting(1, returns='*s*s')
    def read_tag_stream(self, c):
        """Read the stream started with start_tag_stream. Returns two lists,
        each as long as the number of counts that have occurred since the
        buffer was refreshed. First list is timestamps in ps, second is apd
        indices.
        """
        if self.stream is None:
            logging.error('read_tag_stream attempted while stream is None.')
            return
        timestamps, channels = self.read_raw_stream()
        # Convert timestamps to strings since labrad does not support int64s
        # It must be converted to int64s back on the client
        timestamps = timestamps.astype(str).tolist()
        # Map channels to semantic channels
        semantic_channels = [self.channel_mapping[chan] for chan in channels]
        return timestamps, semantic_channels

    @setting(3, apd_index='i', num_to_read='i', returns='*w')
    def read_counter(self, c, apd_index, num_to_read=None):
        if self.stream is None:
            logging.error('read_counter attempted while stream is None.')
            return
        if num_to_read is None:
            # Poll once and return the result
            counts =  self.read_counter_internal(apd_index)
        else:
            # Poll until we've read the requested number of samples
            counts = []
            while len(counts) < num_to_read:
                counts.extend(self.read_counter_internal(apd_index,
                                                         num_to_read))
            if len(counts) > num_to_read:
                msg = 'Read {} samples, only requested{}'.format(len(counts),
                            num_to_read)
                logging.error(msg)
        return counts

__server__ = ApdTagger()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
