# -*- coding: utf-8 -*-
"""
DAQ task for an edge count (e.g. an APD)

Created on Sun Dec  2 12:40:55 2018

@author: mccambria
"""

# User modules
import Utils.tool_belt as tool_belt

# Library modules
import nidaqmx
import nidaqmx.stream_readers as stream_readers
from nidaqmx.constants import TriggerType
from nidaqmx.constants import Level
from nidaqmx.constants import AcquisitionType


def load(daqName, daqCIReadFrom,
         daqDIPulserClock, daqDIPulserGate, period):
    """
    Load the task. The task starts immediately, so you'll want to
    discard the first value you read. Note that the task returns cumulative
    counts over the entire task. 

    Params:
        daqName: string
            The name of the DAQ that we'll be using
        daqCIReadFrom: int
            DAQ CI channel to read from
        daqCIEdgeCounter: int
            DAQ DI for clock signal from pulser
        daqDIPulserGate: int
            DAQ DI for gate signal from pulser
        period: numpy.int64
            Period of a sample in ns

    Returns:
        nidaqmx.stream_reader
    """

    # Create the task
    # Append it to the task list so we can clean it up later
    task = nidaqmx.Task("edge_count_counter_input")
    taskList = tool_belt.get_task_list()
    taskList.append(task)

    chanName = daqName + "/ctr" + str(daqCIReadFrom)
    task.ci_channels.add_ci_count_edges_chan(chanName)

    # Set up the input stream
    inputStream = nidaqmx.task.InStream(task)
    streamReader = stream_readers.CounterReader(inputStream)
    # Just collect whatever data is available when we read
    streamReader.verify_array_shape = False

    # Set up the gate ("pause trigger")
    if daqDIPulserGate is not None:
        daqDIPulserGateName = "PFI" + str(daqDIPulserGate)
        # Pause when low - i.e. read only when high
        task.triggers.pause_trigger.trig_type = TriggerType.DIGITAL_LEVEL
        task.triggers.pause_trigger.dig_lvl_when = Level.LOW
        task.triggers.pause_trigger.dig_lvl_src = daqDIPulserGateName

    # Configure the sample to advance on the rising edge of the PFI input.
    # The frequency specified is just the max expected rate in this case.
    # We'll stop once we've run all the samples.
    daqDIPulserClockName = "PFI" + str(daqDIPulserClock)
    freq = float(1/(period*(10**-9)))  # freq in seconds as a float
    task.timing.cfg_samp_clk_timing(freq, source=daqDIPulserClockName,
                                    sample_mode=AcquisitionType.CONTINUOUS)

    # Start the task. It will start counting immediately so we'll have to
    # discard the first sample.
    task.start()

    return streamReader
