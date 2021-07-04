# -*- coding: utf-8 -*-
"""Template for scripts that should be run directly from the files themselves
(as opposed to from the control panel, for example).

Created on Sun Jun 16 11:22:40 2019

@author: mccambria
"""


# %% Imports
    

from pulsestreamer import PulseStreamer as Pulser
from pulsestreamer import TriggerStart
from pulsestreamer import OutputState
import numpy
from pulsestreamer import Sequence
import labrad
import utils.tool_belt as tool_belt
import nidaqmx
import nidaqmx.stream_writers as stream_writers


# %% Constants


LOW = 0
HIGH = 1


# %% Functions


def load_laser_stream(cxn, pulse_length):

    # Create a new task
    task = nidaqmx.Task('MCCTEST2')
    
    try:

        # Set up the output channels
#        task.do_channels.add_do_chan('dev1/port0/line0')
        task.do_channels.add_do_chan('dev1/port0')
        
#        task.write(False)
#    
#        input('Press enter to stop...')
#        
#        task.close()
#        
#        return
    
        # Set up the output stream
        output_stream = nidaqmx.task.OutStream(task)
        writer = stream_writers.DigitalSingleChannelWriter(output_stream)
#        stream_voltages = numpy.array(100*[False, True], dtype=bool)
        stream_voltages = numpy.array(100*[1, 0], dtype=numpy.uint32)
    
        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        # We'll stop once we've run all the samples.
        freq = float(2/(pulse_length*(10**-9)))  # freq in seconds as a float
        sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
#        sample_mode = nidaqmx.constants.AcquisitionType.FINITE
        task.timing.cfg_samp_clk_timing(freq, source='PFI0', 
                                        sample_mode=sample_mode)#,
#                                        samps_per_chan=len(stream_voltages))
    
        writer.write_many_sample_port_uint32(stream_voltages)
    
        # Close the task once we've written all the samples
        task.register_done_event(close_task_callback)
    
        task.start()
        
    except Exception as e:
        
        print('error: {}'.format(e))
        task.close()
        
    return task

def close_task_callback(task_handle=None, status=None, callback_data=None):
#    print(task_handle)
    if task_handle is not None:
        task_handle.close()
    return 0
    


# %% Main


def main(channels):
    """When you run the file, we'll call into main, which should contain the
    body of the script.
    """
    
    pulse_length = 500
    
#    period = numpy.int64(100)
#    period = numpy.int64(300)
    period = numpy.int64(10**4)
#    period = numpy.int64(10**5)
#    period = numpy.int64(10**9)
    half_period = period // 2
    
    task = load_laser_stream(cxn, pulse_length)
    
    try:
    
        seq = Sequence()
    
    #    train = [(half_period, HIGH), (half_period, HIGH)]
    #    train = [(half_period, HIGH), (half_period, LOW)]
    #    train = [(10**4, HIGH), (500, LOW),
    #             (10**4, HIGH), (5000, LOW),
    #             ]
        train = [(pulse_length, HIGH), (pulse_length, LOW),
                 (pulse_length, HIGH), (pulse_length, LOW),
                 ]
        for chan in channels:
            seq.setDigital(chan, train)
    #    laser_high = 1.0
    #    laser_low = 0
    #    train = [(10**4, laser_high), (10**4, laser_low),
    #             (10**4, laser_high), (10**4, laser_low),
    #             ]
    ##    for chan in channels:
    ##        seq.setAnalog(chan, train)
    #    seq.setAnalog(0, train)
        
        pulser = Pulser('128.104.160.111')
        pulser.constant(OutputState([]))
        pulser.setTrigger(start=TriggerStart.SOFTWARE)
        pulser.stream(seq, Pulser.REPEAT_INFINITELY)
        pulser.startNow()
        
        input('Press enter to stop...')
        
        pulser.constant(OutputState([]))
        
    finally:
        
        task.close()


# %% Run the file


# The __name__ variable will only be '__main__' if you run this file directly.
# This allows a file's functions, classes, etc to be imported without running
# the script that you set up here.
if __name__ == '__main__':

    # Set up your parameters to be passed to main here
    
    # Rabi
    laser_names = ['laser_515']
    pos = [0.0, 0.0, 5.0]
    
    # Hahn
#    laser_names = ['laser_532', 'laser_589', 'laser_638']
#    pos = [0.0, 0.0, 0]
    
    chans = []
    with labrad.connect() as cxn:
        tool_belt.set_xyz(cxn, pos)
        for el in laser_names:
            # tool_belt.set_filter(cxn, optics_name=laser_name, filter_name='nd_0.5')
            chan = tool_belt.get_registry_entry(cxn, 'do_{}_dm'.format(el),
                                         ['', 'Config', 'Wiring', 'Pulser'])
            chans.append(chan)
#    constant(chans, 0.0, 0.0)
    main(chans)
