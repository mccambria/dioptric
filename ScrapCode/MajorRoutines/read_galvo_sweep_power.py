# -*- coding: utf-8 -*-
"""
Just like find_nvs, except we read from the photometer instead of the APD.
Not really useful, but a good demonstration. The difference is that the APD
requires a counter input, whereas the photometer just uses an analog input.

Created on Fri Dec 2 12:24:54 2018

@author: mccambria
"""

from Utils import tool_belt, sweep_utils
from DaqIo import dual_value_analog_output, edge_count_counter_input
import numpy
import time
from PulseStreamer.pulse_streamer_jrpc import Start, Mode


# %% Functions


def run_pulser_stream(pulserIP,
                      pulserDODaqClock, pulserDODaqStart, pulserDODaqGate,
                      period, totalSamples):

    # Get pulser Note that it only supports one stream at a time
    pulser = tool_belt.get_pulser(pulserIP)
    if pulser is None:
        return

    # Set the mode
    pulser.setTrigger(start=Start.IMMEDIATE, mode=Mode.SINGLE)

    seq = [(period, pulserDODaqStart, 0, 0)]

    seqElem = [(period / 2, pulserDODaqClock, 0, 0),
               (period / 2, [], 0, 0)]

    for index in range(totalSamples):
        seq.extend(seqElem)

    pulser.stream(seq, 1)


# %% Main


def main(pulserIP, daqName,
         daqAOGalvoX, daqAOGalvoY, daqAIPhotometer,
         daqDIPulserClock, daqDIPulserStart, daqDIPulserGate,
         pulserDODaqClock, pulserDODaqStart, pulserDODaqGate,
         xLength, yLength,
         resolution, period, offset):
    """
    Find those NVs

    Params:

        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        daqName: string
            The name of the DAQ that we'll be using

        daqAOGalvoX: list(int)
            DAQ AOs carrying galvo X signal
        daqAOGalvoY: list(int)
            DAQ AOs carrying galvo Y signal

        daqAIPhotometer: int
            DAQ CI for photometer

        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        daqDIPulserStart: int
            DAQ DI for start signal from pulser
        daqDIPulserGate: int
            DAQ DI for gate signal from pulser

        pulserDODaqClock: list(int)
            pulser DOs carrying DAQ clock signal
        pulserDODaqStart: list(int)
            pulser DOs carrying DAQ start signal
        pulserDODaqGate: list(int)
            pulser DOs carrying DAQ gate signal

        xLength: int
            Number of samples in the x direction
        yLength: int
            Number of samples in the Y direction
        resolution: float
            Volts per step between samples
        offset: list(float)
            x, y offset voltages to align the top left of the grid
    """

    # %% Set up the galvo

    dual_value_analog_output.load(daqName,
                                  daqAOGalvoX, daqAOGalvoY,
                                  daqDIPulserClock, daqDIPulserStart,
                                  offset, xLength, yLength,
                                  resolution, period)

    # %% Set up the APD

    numTotalSamples = xLength * yLength

    streamReader = edge_count_counter_input.load(daqName, daqAIPhotometer,
                                                 daqDIPulserClock,
                                                 daqDIPulserStart,
                                                 period)

    # %% Set up the image display

    imgArray = numpy.zeros((yLength, xLength))

    left = offset[0]
    right = left + (xLength * resolution)
    top = offset[1]
    bottom = top + (yLength * resolution)

    imageExtent = [left, right, bottom, top]

    fig, ax, img = tool_belt.create_imagefigure(imgArray,
                                                imageExtent, resolution)

    # %% Run the PulseStreamer

    run_pulser_stream(pulserIP,
                      pulserDODaqClock, pulserDODaqStart, pulserDODaqGate,
                      period, numTotalSamples)

    # %% Collect the data

    # Initialize the necessary values
    numTotalRead = 0

    # The DAQ only fills values 0: numRead of whatever you pass it so let's
    # manually tack those values onto a running array for the total samples
    allSamples = numpy.zeros(numTotalSamples)

    imgWritePos = None

    routineTime = ((period*(10**-9)) * numTotalSamples) + 10
    startTime = time.time()

    print("Progress")

    while numTotalRead < numTotalSamples:

        # Wait for some samples to fill
        time.sleep(0.5)

        # Break out of the while if we've timed out
        runTime = time.time() - startTime
        if runTime > routineTime:
            print("Timed out after " + str(runTime) + "seconds")
            break

        # Initialize/reset the read sample array to it's maximum possible size.
        readSamples = numpy.zeros(numTotalSamples)

        # Read the samples currently in the buffer.
        numRead = streamReader.read_many_sample(readSamples)

        # Check if we collected more samples than we need.
        # This may happen as a consequence of the sleep below. We just need
        # to throw out the excess samples.
        if numTotalRead + numRead > numTotalSamples:
            numRead = numTotalSamples - numTotalRead

        # Tack the new samples onto our total samples
        newSamples = readSamples[0: numRead]
        allSamples[numTotalRead: numTotalRead + numRead] = newSamples

        imgArray, imgWritePos = sweep_utils.populate_img_array(newSamples,
                                                               imgArray,
                                                               imgWritePos)

        tool_belt.update_sweep_image(fig, ax, img, imgArray)

        # Update the totalRead count
        numTotalRead = numTotalRead + numRead
        print(str(int(100 * numTotalRead / numTotalSamples)) + "%")
