# -*- coding: utf-8 -*-
"""
DAQ task for a three value analog output (e.g. the galvo and the piezo)
I'll refer to the first set of values as the A values, the second set of
values as the B values, and the third set of values as the C values

Created on Sun Feb 24 13:00:37 2019

@author: mccambria
"""

# User modules
import Utils.tool_belt as tool_belt

# Library modules
import nidaqmx
import nidaqmx.stream_writers as stream_writers
from nidaqmx.constants import AcquisitionType


def load(daqName, daqAOWriteToA, daqAOWriteToB, daqAOWriteToC,
         daqDIPulserClock, voltages, period):
    """
    Load the task. The task will be started, but the first sample won't be
    written until the first clock rising edge.

    Params:
        daqName: string
            The name of the DAQ that we'll be using
        daqAOWriteToA: tuple(int)
            DAQ AO channels to write the A values to
        daqAOWriteToB: tuple(int)
            DAQ AO channels to write the B values to
        daqAOWriteToC: tuple(int)
            DAQ AO channels to write the B values to
        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        voltages: numpy.ndarray
            Array of voltages to be written to the AO channels. The number of
            row must equal the total number of channels we'll be writing to.
        period: numpy.int64
            Period of a sample in ns

    Returns:
        nidaqmx.task
    """

    # Create the task
    # Append it to the task list so we can clean it up later
    task = nidaqmx.Task("triple_value_analog_output")
    taskList = tool_belt.get_task_list()
    taskList.append(task)

    # Set up the output channels. Assume the first two are for the galvo
    # (-10.0, +10.0) and the third is for the piezo (0.0, +10.0)
    for chan in daqAOWriteToA:
        chanName = daqName + "/AO" + str(chan)
        task.ao_channels.add_ao_voltage_chan(chanName,
                                             min_val=-10.0, max_val=10.0)
    for chan in daqAOWriteToB:
        chanName = daqName + "/AO" + str(chan)
        task.ao_channels.add_ao_voltage_chan(chanName,
                                             min_val=-10.0, max_val=10.0)
    for chan in daqAOWriteToC:
        chanName = daqName + "/AO" + str(chan)
        task.ao_channels.add_ao_voltage_chan(chanName,
                                             min_val=0.0, max_val=10.0)

    # Set up the output stream to the galvo
    outputStream = nidaqmx.task.OutStream(task)
    streamWriter = stream_writers.AnalogMultiChannelWriter(outputStream)

    # Configure the sample to advance on the rising edge of the PFI input.
    # The frequency specified is just the max expected rate in this case.
    # We'll stop once we've run all the samples.
    daqDIPulserClockName = "PFI" + str(daqDIPulserClock)
    freq = float(1/(period*(10**-9)))  # freq in seconds as a float
    task.timing.cfg_samp_clk_timing(freq, source=daqDIPulserClockName,
                                    sample_mode=AcquisitionType.CONTINUOUS)

    # Write the galvo voltages to the stream
    streamWriter.write_many_sample(voltages)

    # Start the task. The first sample won't actually be written until the
    # first clock signal.
    task.start()

    return task
