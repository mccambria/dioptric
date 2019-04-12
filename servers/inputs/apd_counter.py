# -*- coding: utf-8 -*-
"""
Input server for the LASER COMPONENTS COUNT-100C APD. Communicates via the DAQ.

Created on Tue Apr  9 08:52:34 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = APD Counter
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
import nidaqmx
import nidaqmx.stream_readers as stream_readers
from nidaqmx.constants import TriggerType
from nidaqmx.constants import Level
from nidaqmx.constants import AcquisitionType


class ApdCounter(LabradServer):
    name = 'APD Counter'

    def initServer(self):
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)
        self.tasks = {}
        self.stream_reader_state = {}

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(['Config', 'Wiring', 'Daq'])
        p.get('di_clock')
        p.dir()
        result = await p.send()
        return result

    def on_get_config(self, config):
        # The counters share a clock, but everything else is distinct
        self.daq_di_clock = config['get']
        # Determine how many APDs we're supposed to set up
        apd_sub_dirs = []
        apd_indices = []
        sub_dirs = config['dir'][0]
        for sub_dir in sub_dirs:
            if sub_dir.startswith('Apd_'):
                apd_sub_dirs.append(sub_dir)
                apd_indices.append(int(sub_dir.split('_')[1]))
        if len(apd_sub_dirs) > 0:
            wiring = ensureDeferred(self.get_wiring(apd_sub_dirs))
            wiring.addCallback(self.on_get_wiring, apd_indices)

    async def get_wiring(self, apd_sub_dirs):
        p = self.client.registry.packet()
        for sub_dir in apd_sub_dirs:
            p.cd(['', 'Config', 'Wiring', 'Daq', sub_dir])
            p.get('ctr_apd')
            p.get('ci_apd')
            p.get('di_apd_gate')
        result = await p.send()
        return result['get']

    def on_get_wiring(self, wiring, apd_indices):
        self.daq_ctr_apd = {}
        self.daq_ci_apd = {}
        self.daq_di_apd_gate = {}
        # Loop through the possible counters
        for loop_index in range(len(apd_indices)):
            apd_index = apd_indices[loop_index]
            wiring_index = 3 * loop_index
            self.daq_ctr_apd[apd_index] = wiring[wiring_index]
            self.daq_ci_apd[apd_index] = wiring[wiring_index+1]
            self.daq_di_apd_gate[apd_index] = wiring[wiring_index+2]

    @setting(0, apd_index='i', period='i', total_num_to_read='i')
    def load_stream_reader(self, c, apd_index, period, total_num_to_read):
        try:
            self.try_load_stream_reader(c, apd_index,
                                        period, total_num_to_read)
        except:
            self.close_task(c, apd_index)
            raise

    def try_load_stream_reader(self, c, apd_index, period, total_num_to_read):

        # Close the task if it exists
        self.close_task(c, apd_index)
        task = nidaqmx.Task('Apd-load_stream_reader_{}'.format(apd_index))
        self.tasks[apd_index] = task

        chan_name = self.daq_ctr_apd[apd_index]
        chan = task.ci_channels.add_ci_count_edges_chan(chan_name)
        chan.ci_count_edges_term = self.daq_ci_apd[apd_index]

        # Set up the input stream
        input_stream = nidaqmx.task.InStream(task)
        reader = stream_readers.CounterReader(input_stream)
        # Just collect whatever data is available when we read
        reader.verify_array_shape = False

        # Set up the gate ('pause trigger')
        # Pause when low - i.e. read only when high
        task.triggers.pause_trigger.trig_type = TriggerType.DIGITAL_LEVEL
        task.triggers.pause_trigger.dig_lvl_when = Level.LOW
        gate_chan_name = self.daq_di_apd_gate[apd_index]
        task.triggers.pause_trigger.dig_lvl_src = gate_chan_name

        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        # We'll stop once we've run all the samples.
        freq = float(1/(period*(10**-9)))  # freq in seconds as a float
        task.timing.cfg_samp_clk_timing(freq, source=self.daq_di_clock,
                                        sample_mode=AcquisitionType.CONTINUOUS)

        # Initialize the state dictionary for this stream
        self.stream_reader_state[apd_index] = {}
        state_dict = self.stream_reader_state[apd_index]
        state_dict['reader'] = reader
        state_dict['num_read_so_far'] = 0
        state_dict['total_num_to_read'] = total_num_to_read
        # Something funny is happening if we get more
        # than 1000 samples in one read
        state_dict['buffer_size'] = min(total_num_to_read, 1000)
        state_dict['last_value'] = None  # Last cumulative value we read

        # Start the task. It will start counting immediately so we'll have to
        # discard the first sample.
        task.start()

    @setting(1, apd_index='i', one_sample='b', returns='*i')
    def read_stream(self, c, apd_index, one_sample=False):

        # Unpack the state dictionary
        state_dict = self.stream_reader_state[apd_index]
        reader = state_dict['reader']
        num_read_so_far = state_dict['num_read_so_far']
        total_num_to_read = state_dict['total_num_to_read']
        buffer_size = state_dict['buffer_size']

        # The counter task begins counting as soon as the task starts.
        # The AO channel writes its first samples only on the first clock
        # signal after the task starts. This means that if we're running
        # AOs on the clock, then there's one sample from the counter stream
        # that we don't want to record. We do need it for calculations.
        if state_dict['last_value'] == None:
            # If we're just collecting one sample, then assume there's no
            # AO stream set up.
            if one_sample:
                state_dict['last_value'] = 0
            else:
                state_dict['last_value'] = reader.read_one_sample_uint32()

        # Initialize the read sample array to its maximum possible size.
        new_samples_cum = numpy.zeros(buffer_size,
                                      dtype=numpy.uint32)

        # Read the samples currently in the DAQ memory.
        if one_sample:
            num_new_samples = 1
            new_samples_cum[0] = reader.read_one_sample_uint32()
        else:
            num_new_samples = reader.read_many_sample_uint32(new_samples_cum)
        if num_new_samples >= buffer_size:
            raise Warning('The DAQ buffer contained more samples than '
                          'expected. Validate your parameters and '
                          'increase bufferSize if necessary.')

        # Check if we collected more samples than we need, which may happen
        # if the pulser runs longer than necessary. If so, just to throw out
        # excess samples.
        if num_read_so_far + num_new_samples > total_num_to_read:
            num_new_samples = total_num_to_read - num_read_so_far
        new_samples_cum = new_samples_cum[0: num_new_samples]

        # The DAQ counter reader returns cumulative counts, which is not what
        # we want. So we have to calculate the difference between samples
        # n and n-1 in order to get the actual count for the nth sample.
        new_samples_diff = numpy.zeros(num_new_samples)
        for index in range(num_new_samples):
            if index == 0:
                last_value = state_dict['last_value']
            else:
                last_value = new_samples_cum[index-1]

            new_samples_diff[index] = new_samples_cum[index] - last_value

        if num_new_samples > 0:
            state_dict['last_value'] = new_samples_cum[num_new_samples-1]

        # Update the current count
        state_dict['num_read_so_far'] = num_read_so_far + num_new_samples

        return new_samples_diff

    @setting(2, apd_index='i')
    def close_task(self, c, apd_index):
        try:
            task = self.tasks[apd_index]
            task.close()
            self.tasks.pop(apd_index)
            self.stream_reader_state.pop(apd_index)
        except Exception:
            pass


__server__ = ApdCounter()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
