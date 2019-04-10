# -*- coding: utf-8 -*-
"""
Set up the stream reader for the photometer aka power meter.

Created on Wed Dec 19 17:54:52 2018

@author: mccambria
"""

import cfm_utils
import nidaqmx
import nidaqmx.stream_readers as niStreamReaders
from nidaqmx.constants import Edge
from nidaqmx.constants import AcquisitionType


def main(daqName, daqAIPhotometer,
         daqDIPulserClock, daqDIPulserStart, period):
    """
    Set up the photometer stream to the DAQ

    Params:

        daqName: string
            The name of the DAQ that we'll be using

        daqCIApd: int
            DAQ CI for photometer

        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        daqDIPulserStart: int
            DAQ DI for start signal from pulser

        period: int
            Period of a sample in ns
    """

    # Create the task
    # Append it to the task list so we can clean it up later
    photoTask = nidaqmx.Task("photoTask")
    taskList = cfm_utils.get_task_list()
    taskList.append(photoTask)

    # We'll set up an edge counter to determine how many of TTL pulses the APD
    # sends during the readout time. This approximates the photoluminescence,
    # ie the number of photons emitted by the sample while we're looking.
    # Myers uses a readout time of 350 ns.
    chanName = daqName + "/ai" + str(daqAIPhotometer)
    photoTask.ai_channels.add_ai_voltage_chan(chanName,
                                              min_val=0.0, max_val=2.0)

    # Set up the input stream
    inputStream = nidaqmx.task.InStream(photoTask)
    streamReader = niStreamReaders.AnalogSingleChannelReader(inputStream)
    # Just collect whatever data is available when we read
    streamReader.verify_array_shape = False

    # Start the task on the first rising edge of our clock signal
    # For rather unimportant technical reasons, counters use arm start
    # triggers, while analog/digital use start triggers. You have to set up
    # arm start triggers manually, which is kind of ridiculous, but it is
    # what it is.
    daqDIPulserStartName = "PFI" + str(daqDIPulserStart)
    photoTask.triggers.start_trigger.cfg_dig_edge_start_trig(daqDIPulserStartName)

    # Configure the sample to advance on the falling edge of the PFI input.
    # By using the falling edge, we'll read when the galvo is in the middle of
    # a sample rather than just as it is switching samples.
    # The frequency specified is just the max expected rate in this case.
    # We'll stop once we've run all the samples.
    daqDIPulserClockName = "PFI" + str(daqDIPulserClock)
    freq = float(1/(period*(10**-9)))  # freq in seconds as a float
    photoTask.timing.cfg_samp_clk_timing(freq,
                                         source=daqDIPulserClockName,
                                         sample_mode=AcquisitionType.CONTINUOUS,
                                         active_edge=Edge.FALLING)

    # "Start" the task. It won't actually start until it gets the trigger
    # signal the we set up above.
    photoTask.start()

    return streamReader
