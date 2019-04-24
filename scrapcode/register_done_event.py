# -*- coding: utf-8 -*-
"""
register_done_event test

Created on Wed Apr 24 10:46:59 2019

@author: mccambria
"""

import nidaqmx
import nidaqmx.stream_writers as stream_writers
import time
import numpy
from pulsestreamer import PulseStreamer as Pulser
from pulsestreamer import TriggerStart
from pulsestreamer import TriggerRearm
from pulsestreamer import OutputState
from pulsestreamer import Sequence

class Test:
    
    def close_task_internal(self, task_handle=None, status=None, callback_data=None):
        print('close_task_internal')
        return 0
    
#    def fill_buffer(self, task_handle=None, every_n_samples_event_type=None,
#                    number_of_samples=None, callback_data=None):
    def fill_buffer(self, task_handle, every_n_samples_event_type,
                    number_of_samples, callback_data):
        print('fill_buffer')
        
        if self.buffered is True:
            return
        
        self.buffered = True
        
#        one_d_voltages = [0.02, 0.03, 0.04, 0.05]
#        voltages = [one_d_voltages,
#                    one_d_voltages]
#        voltages = numpy.array(voltages)
#        self.writer.write_many_sample(voltages)
        
        return 0
    
    def run(self):
        
        self.buffered = False
        try:
            
            task = None
            num_samples = 11
            
#            one_d_voltages = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0,
#                              0.01, 0.02, 0.03, 0.04, 0.05]
            one_d_voltages = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0]*750
            voltages = [one_d_voltages,
                        one_d_voltages]
            voltages = numpy.array(voltages)
            
            task = nidaqmx.Task("task")
            
            # Set up the output channels
            task.ao_channels.add_ao_voltage_chan('dev1/ao0', min_val=-5.0, max_val=5.0)
            task.ao_channels.add_ao_voltage_chan('dev1/ao1', min_val=-5.0, max_val=5.0)
            
            # Set up the output stream to the galvo
            outputStream = nidaqmx.task.OutStream(task)
            writer = stream_writers.AnalogMultiChannelWriter(outputStream)
            self.writer = writer
            
            task.timing.cfg_samp_clk_timing(1, 
                                            source='PFI12',
#                                            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                            samps_per_chan=len(one_d_voltages))
                    
            
            task.register_done_event(self.close_task_internal)
#            task.register_every_n_samples_transferred_from_buffer_event(6, self.fill_buffer)
            
            # Write the galvo voltages to the stream
            writer.write_many_sample(voltages)
            print(numpy.size(voltages))
            
            self.task = task
            print('start')
            task.start()
            
            pulser = Pulser('128.104.160.11')
            pulser.setTrigger(start=TriggerStart.SOFTWARE)
            seq = Sequence()
            train = [((10**9) - 100, 0), (100, 1)]
            seq.setDigital(0, train)
            train = [(10**8, 1)]
            seq.setDigital(3, train)
            pulser.stream(seq, num_samples, OutputState([3], 0, 0))
            pulser.startNow()
            
            time.sleep(15.0)
            
        finally:
            
            if task is not None:
                task.close()
        
test = Test()
test.run()
