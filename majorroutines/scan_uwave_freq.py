# -*- coding: utf-8 -*-
"""
Scans the microwave frequency, taking counts at each point.

Created on Thu Apr 11 15:39:23 2019

@author: mccambria
"""

# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import Inputs.apd as apd
from PulseStreamer.pulse_streamer_jrpc import Start
from PulseStreamer.Sequence import Sequence
import Outputs.xyz as xyz
import MajorRoutines.find_nv_center as find_nv_center

# Library modules
import numpy
import time


# %% Main


def main(pulserIP, daqName, rfAddress,
         daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd,
         daqDIPulserClock, daqDIPulserGate,
         pulserDODaqClock, pulserDODaqGate,
         pulserDOAom, pulserDORf,
         name, xCenter, yCenter, zCenter,
         freqCenter, freqRange, freqResolution, rfPower,
         readout):
    """
    Entry point for the routine

    Params:

        pulserIP: string
            The IP of the PulseStreamer that we'll be using
        daqName: string
            The name of the DAQ that we'll be using
        rfAddress: string
            VISA address of the signal generator

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
            pulser DO carrying DAQ clock signal
        pulserDODaqGate: int
            pulser DO carrying DAQ gate signal
        pulserDOAom: int
            pulser DO carrying AOM gate signal
        pulserDORf: int
            pulser DO carrying RF gate signal

        name: string
            The file names consist of <date>_<time>_<name>.<ext>
        xCenter: float
            Fixed voltage for the galvo x
        yCenter: float
            Fixed voltage for the galvo y
        zCenter: float
            Fixed voltage for the piezo
        freqCenter: float
            Center frequency to scan about (GHz)
        freqRange: float
            Frequency range to scan about
        freqResolution: int
            Number of samples to take over the range
        rfPower: float
            Power setting of the signal generator (dBm)
        readout: numpy.int64
            Readout time of a sample (time APD channel is ungated) in ns
    """

    # %% Initial calculations and setup

    # Calculate the frequencies we need to set
    freqLow = freqCenter - (freqRange / 2)
    freqStepSize = freqRange / freqResolution
    freqSteps = numpy.arange(freqResolution)
    freqs = (freqStepSize * freqSteps) + freqLow

    # As a test, flip the freqs
#    freqs = numpy.flip(freqs)

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    counts = numpy.empty(freqResolution)
    counts[:] = numpy.nan

    # %% Run find_nv_center to optimize the position of the resonance scan

    # Find the optimized position in x, y ,z
    optiCenters = find_nv_center.main(pulserIP, daqName,
						daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd,
                        daqDIPulserClock, daqDIPulserGate,
						pulserDODaqClock, pulserDODaqGate,
                        pulserDOAom, name, xCenter, yCenter, zCenter, 0.1,
						0.02/60.0, None, 10.0/60.0, numpy.int64(10 * 10**6))

    # Set optimized positions based on the fins_nv_center to use for scan
    xCenterOptimized = optiCenters[0]
    yCenterOptimized = optiCenters[1]
    zCenterOptimized = 0

    time.sleep(5.0)

    # %% Set up the pulser

    pulser = tool_belt.get_pulser(pulserIP)

    # Set the PulseStreamer to start on python's command. We can run the
    # loaded stream repeatedly using startNow().
    pulser.setTrigger(start=Start.SOFTWARE)

    seq = Sequence()
    low = 0
    high = 1

    delay = numpy.int64(0.1 * 10**9)
    period = delay + readout

    # After delay, ungate the APD channel for readout.
    # The delay is to allow the signal generator to switch frequencies.
    readoutTrain = [(delay, low), (readout, high)]
    seq.setDigitalChannel(pulserDODaqGate, readoutTrain)

    # Collect a sample with rf off at the end of the first gating
    clockTrain = [(period, low), (100, high)]
    seq.setDigitalChannel(pulserDODaqClock, clockTrain)

    # The AOM should always be on
    staticOnTrain = [(period, high)]
    seq.setDigitalChannel(pulserDOAom, staticOnTrain)

    # The RF should always be on
    seq.setDigitalChannel(pulserDORf, staticOnTrain)
#    # The RF should be on during readout
#    seq.setDigitalChannel(pulserDORf, readoutTrain)
    # The RF should be on for half the period
#    halfPeriodTrain = [(period // 2, low), (period // 2, high)]
#    seq.setDigitalChannel(pulserDORf, halfPeriodTrain)


    # Run the sequence once per start, leave the AOM and RF on when complete
    pulser.stream(seq.getSequence(), 1, (0, [pulserDOAom, pulserDORf], 0, 0))
    # Run the sequence once per start, leave the AOM on when complete
#    pulser.stream(seq.getSequence(), 1, (0, [pulserDOAom], 0, 0))

    # %% Set the galvo and piezo position

    xyz.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, piezoSerial,
                  xCenterOptimized, yCenterOptimized, zCenterOptimized)

    # %% Load the APD task

    streamReader, apdTask = apd.stream_read_load_daq(daqName, daqCIApd,
                                                     daqDIPulserClock,
                                                     daqDIPulserGate,
                                                     period)

    # %% Get the signal generator

    sigGen = tool_belt.get_VISA_instr(rfAddress)

    # %% Set up the plot

    fig = tool_belt.create_line_plot_figure(counts, freqs)

    # %% Collect and plot the data

    # The timeout for each sample will be 1.1 * (the period in seconds)
    timeout = 1.1 * (float(period) * 10**-9)

    # Set previousSample to 0 since we don't have any samples yet
    previousSample = 0

    # Start "Press enter to stop..."
    tool_belt.init_safe_stop()

    # Take a sample and increment the frequency
    for ind in range(freqResolution):

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        sigGen.write("FREQ %fGHZ" % (freqs[ind]))

        # If this is the first sample then we have to enable the signal
        if ind == 0:
            sigGen.write("AMPR %fDBM" % (rfPower))
            sigGen.write("ENBR 1")

        # Start the timing stream
        pulser.startNow()

        # Read the sample - this will hang until the sample becomes available
        sample = streamReader.read_one_sample_uint32(timeout)

        # The counter task returns the cumulative count over the life of the
        # task. We want the individual count over each sample period.
        counts[ind] = sample - previousSample
        previousSample = sample

        tool_belt.update_line_plot_figure(fig, counts)

    # %% Turn off the RF and save the data

    sigGen.write("ENBR 0")

    timeStamp = tool_belt.get_time_stamp()

    rawData = {"timeStamp": timeStamp,
               "name": name,
               "xyzCenters": [xCenter, yCenter, zCenter],
               "freqCenter": freqCenter,
               "freqRange": freqRange,
               "freqResolution": freqResolution,
               "rfPower": rfPower,
               "readout": int(readout),
               "counts": counts.astype(int).tolist()}

    filePath = tool_belt.get_file_path("find_resonance", timeStamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)

