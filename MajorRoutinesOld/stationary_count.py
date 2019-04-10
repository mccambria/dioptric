# -*- coding: utf-8 -*-
"""
Count the APD pulses for totalNumSamples periods. Plot a line graph of the
results.

Created on Fri Dec 2 12:24:54 2018

@author: mccambria
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import Inputs.apd as apd
import Outputs.xyz as xyz

# Library modules
import numpy
import matplotlib.pyplot as plt


# %% Functions


def update_line_plot(newSamples, numReadSoFar, *args):
    """
    Update the line plot figure. Called once per loop through the APD stream
    read function.

    Params:
        newSamples: numpy.ndarray
            Array of new samples to populate the image with
        numReadSoFar: int
            Total number of samples read so far
        args: tuple
            fig: matplotlib.figure.Figure
                The matplotlib figure to update
                Should be from tool_belt.create_image_figure
            samples: numpy.ndarray
                Array of counts to plot
            writePos: list(int)
                Current write position in samples
    """

    fig, samples, writePos = args

    # Write to the samples array
    samples[writePos[0]: writePos[0] + len(newSamples)] = newSamples
    writePos[0] += len(newSamples)

    # Update the figure
    tool_belt.update_line_plot_figure(fig, samples)


# %% Main


def main(pulserIP, daqName,
         daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd,
         daqDIPulserClock, daqDIPulserGate,
         pulserDODaqClock, pulserDODaqGate, pulserDOAom,
         xCenter, yCenter, zCenter,
         totalNumSamples, period, readout):
    """
    Entry point for the routine

    Params:

        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        daqName: string
            The name of the DAQ that we'll be using

        daqAOGalvoX: int
            DAQ AO carrying galvo X signal
        daqAOGalvoY: int
            DAQ AO carrying galvo Y signal
        piezoSerial: string
            Objective piezo serial number

        daqCIApd: int
            DAQ CI for APD

        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        daqDIPulserGate: int
            DAQ DI for gate signal from pulser

        pulserDODaqClock: int
            pulser DOs carrying DAQ clock signal
        pulserDODaqGate: int
            pulser DOs carrying DAQ gate signal
        pulserDOAom: int
            pulser DOs carrying AOM gate signal

        xCenter: float
            Fixed voltage for the galvo x
        yCenter: float
            Fixed voltage for the galvo y
        zCenter: float
            Fixed voltage for the piezo
        totalNumSamples: int
            Number of samples to collect
        period: numpy.int64
            Total period of a sample in ns
        readout: numpy.int64
            Readout time of a sample (time APD channel is ungated) in ns
            If None, the gate channel will be low for the entire experiment
    """

    # %% Set up xyz

    xyz.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, piezoSerial,
                  xCenter, yCenter, zCenter)

    # %% Set up the APD

    streamReader, apdTask = apd.stream_read_load_daq(daqName, daqCIApd,
                                                     daqDIPulserClock,
                                                     daqDIPulserGate,
                                                     period)

    # %% Initialize the figure

    samples = numpy.empty(totalNumSamples)
    samples.fill(numpy.nan)  # Only floats support NaN
    writePos = [0]  # This is a list because we need a mutable variable

    # Set up the line plot
    fig = tool_belt.create_line_plot_figure(samples)
    
    # Maximize the window
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    # %% Run the PulseStreamer

    tool_belt.pulser_readout_cont_illum(pulserIP, pulserDODaqClock,
                                        pulserDODaqGate, pulserDOAom, 
                                        period, readout, totalNumSamples)

    # %% Collect the data

    timeout = ((period*(10**-9)) * totalNumSamples) + 10

    apd.stream_read_daq(streamReader, totalNumSamples,
                        timeout, update_line_plot, fig, samples, writePos)

    average = numpy.mean(samples[0:writePos[0]])
    print("average: {0:d}".format(int(average)))
    stDev = numpy.std(samples[0:writePos[0]])
    print("standard deviation: {0:.3f}".format(stDev))
