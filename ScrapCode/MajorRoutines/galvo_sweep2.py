# -*- coding: utf-8 -*-
"""
This routine sweeps the galvo over a square grid. The sweep is conducted in a
winding pattern (left to right, then down one, the right to left, then down
one...)

Created on Fri Nov 23 22:33:54 2018

@author: mccambria
"""

import numpy
import nidaqmx
import nidaqmx.stream_writers as niStreamWriters
from Utils import tool_belt, sweep_utils


def main(pulserIP, daqName,
         daqAOGalvoX, daqAOGalvoY, daqDIPulser, pulserDODaq,
         samplesPerDim, resolution, period, offset, initial):
    """
    Sweep the galvo

    Params:

        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        daqName: string
            The name of the DAQ that we'll be using

        daqAOGalvoX: list(int)
            DAQ AOs carrying galvo X signal
        daqAOGalvoY: list(int)
            DAQ AOs carrying galvo Y signal
        daqDIPulser: int
            DAQ DI for clock signal from pulser
        pulserDODaq: list(int)
            pulser DOs carrying DAQ clock signal

        samplesPerDim: int
            Number of samples in each direction
        resolution: float
            Volts per step between samples
        period: float
            Period of a sample in seconds
        offset: list(float)
            x, y offset voltages to align the top left of the grid
        initial: list(float)
            x, y initial values for the galvo to sit at before we start the
            PulseStreamer. This doesn't really matter, it should just be
            different than the first sample value
    """

    # %%  Get pulser

    pulser = tool_belt.get_pulser(pulserIP)
    if pulser is None:
        return

    # %%  Set up the galvo values

    # Calculate the total number of samples we'll collect
    totalSamples = samplesPerDim**2

    # Set up vectors for the number of samples in each direction
    # [0, 1, 2, ... samplesPerDim]
    stepsX = numpy.arange(samplesPerDim)
    stepsY = numpy.arange(samplesPerDim)

    # Apply scale and offsets to get the voltages we'll apply to the galvo
    # Note that the x/y angles, not the actual x/y positions are linear
    # in these voltages. We don't care about this since our angular range is
    # is small enough to warrant sin(theta) ~= theta.
    voltagesX = (resolution * stepsX) + offset[0]
    voltagesY = (resolution * stepsY) + offset[1]

    # Get the 2 x totalSamples array of voltages to apply to the galvo
    galvoVoltages = sweep_utils.winding_cartesian_product(voltagesX, voltagesY)
    print(galvoVoltages)

    # Add a dummy value that the galvo will sit at until
    # we start the PulseStreamer
    galvoInitials = numpy.array(initial)
    numpy.insert(galvoVoltages, galvoInitials, 0, axis=0)

    # %%  Set up the DAQ

    # Create the task
    # Append it to the task list so we can clean it up later
    sweepTask = nidaqmx.Task("sweepTask")
    taskList = tool_belt.get_task_list()
    taskList.append(sweepTask)

    # Set up the output channels
    daqAOGalvoXName = daqName + "/AO" + str(daqAOGalvoX[0])
    sweepTask.ao_channels.add_ao_voltage_chan(daqAOGalvoXName,
                                              min_val=-1.0,
                                              max_val=1.0)
    daqAOGalvoYName = daqName + "/AO" + str(daqAOGalvoY[0])
    sweepTask.ao_channels.add_ao_voltage_chan(daqAOGalvoYName,
                                              min_val=-1.0,
                                              max_val=1.0)

    # Set up the output stream to the galvo
    outputStream = nidaqmx.task.OutStream(sweepTask)
    streamWriter = niStreamWriters.AnalogMultiChannelWriter(outputStream,
                                                            auto_start=True)

    # Configure the sample to advance on the rising edge of the PFI0 input
    # Assume a max sample rate of 4 samples per channel per second
    # We'll stop once we've run all the samples
    freq = 1/period
    daqDIPulserName = "PFI" + str(daqDIPulser)
    sweepTask.timing.cfg_samp_clk_timing(freq,
                                         source=daqDIPulserName,
                                         samps_per_chan=totalSamples)

    # Write the galvo voltages to the stream
    streamWriter.write_many_sample(galvoVoltages)

    # %%  Run the PulseStreamer

    periodNano = int(period * (10**9))  # integer period in ns
    tool_belt.pulser_square_wave(pulser, periodNano,
                                 pulserDODaq, totalSamples)

    sweepTask.wait_until_done((totalSamples * period) + 1)
