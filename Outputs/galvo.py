# -*- coding: utf-8 -*-
"""
Galvo control functions

Created on Mon Mar  4 08:52:20 2019

@author: mccambria
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import nidaqmx.stream_writers as stream_writers
from nidaqmx.constants import AcquisitionType

# Library modules
import nidaqmx
import numpy


# %% Writes


def write_daq(daqName, daqAOGalvoX, daqAOGalvoY, xVoltage, yVoltage):
    """
    Set the galvo AOs to the specified voltages.

    Params:
        daqName: string
            The name of the DAQ
        daqAOGalvoX: int
            DAQ AO carrying galvo X signal
        daqAOGalvoY: int
            DAQ AO carrying galvo Y signal
        xVoltage: float
            Galvo x voltage
        yVoltage: float
            Galvo y voltage
    """

    task = nidaqmx.Task()

    # Set up the output channels
    chanName = daqName + "/AO" + str(daqAOGalvoX)
    task.ao_channels.add_ao_voltage_chan(chanName, min_val=-10.0, max_val=10.0)
    chanName = daqName + "/AO" + str(daqAOGalvoY)
    task.ao_channels.add_ao_voltage_chan(chanName, min_val=-10.0, max_val=10.0)

    task.write([xVoltage, yVoltage])

    task.close()


def stream_write_daq(daqName, daqAOGalvoX, daqAOGalvoY, daqDIPulserClock,
                     xVoltages, yVoltages, period):
    """
    Stream the voltages in voltageArr to the galvo

    Params:
        daqName: string
            The name of the DAQ
        daqAOGalvoX: int
            DAQ AO carrying galvo X signal
        daqAOGalvoY: int
            DAQ AO carrying galvo Y signal
        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        xVoltages: numpy.ndarray(float)
            Galvo x voltages
        yVoltages: numpy.ndarray(float)
            Galvo y voltages
        period: numpy.int64
            Period of a sample in ns

    Returns:
        nidaqmx.stream_writer
        nidaqmx.task
    """

    # Create the task
    # Append it to the task list so we can clean it up later
    task = nidaqmx.Task("galvo-stream_write_daq")
    taskList = tool_belt.get_task_list()
    taskList.append(task)

    # Set up the output channels
    chanName = daqName + "/AO" + str(daqAOGalvoX)
    task.ao_channels.add_ao_voltage_chan(chanName, min_val=-10.0, max_val=10.0)
    chanName = daqName + "/AO" + str(daqAOGalvoY)
    task.ao_channels.add_ao_voltage_chan(chanName, min_val=-10.0, max_val=10.0)

    # Set up the output stream
    outputStream = nidaqmx.task.OutStream(task)
    streamWriter = stream_writers.AnalogMultiChannelWriter(outputStream)

    # Configure the sample to advance on the rising edge of the PFI input.
    # The frequency specified is just the max expected rate in this case.
    # We'll stop once we've run all the samples.
    daqDIPulserClockName = "PFI" + str(daqDIPulserClock)
    freq = float(1/(period*(10**-9)))  # freq in seconds as a float
    task.timing.cfg_samp_clk_timing(freq, source=daqDIPulserClockName,
                                    sample_mode=AcquisitionType.CONTINUOUS)
    
    # Start the task before writing so that the channel will sit on the last 
    # value when the task stops. The first sample won't actually be written
    # until the first clock signal.
    task.start()  

    # Write the voltages to the stream
    galvoVoltages = numpy.vstack((xVoltages, yVoltages))
    streamWriter.write_many_sample(galvoVoltages)

    return streamWriter, task


# %% Reads


def read_daq(daqName, daqAOGalvoX, daqAOGalvoY):
    """
    Get the current voltage of the galvo AO.

    Params:
        daqName: string
            The name of the DAQ
        daqAOGalvoX: int
            DAQ AO carrying galvo X signal
        daqAOGalvoY: int
            DAQ AO carrying galvo Y signal
    
    Returns:
        float: the current galvo X voltage
        float: the current galvo Y voltage
    """

    task = nidaqmx.Task()

    # Set up the internal channel
    chanName = daqName + "/_ao" + str(daqAOGalvoX) + "_vs_aognd"
    task.ai_channels.add_ai_voltage_chan(chanName, min_val=-10.0, max_val=10.0)
    chanName = daqName + "/_ao" + str(daqAOGalvoY) + "_vs_aognd"
    task.ai_channels.add_ai_voltage_chan(chanName, min_val=-10.0, max_val=10.0)

    voltages = task.read()
    task.close()

    return voltages[0], voltages[1]
