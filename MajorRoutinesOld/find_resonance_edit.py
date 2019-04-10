# -*- coding: utf-8 -*-
"""
This routine will increment the signal generator's frequency through a
specified range, counting fluorescence at each point. A static magnetic field
can be applied to split degenerate spins.

At each frequency, the counts with and without rf will be recorded. A 
normalized counts (without ESR / with ESR) will be calculated.

This function also has the ability to average over multiple runs.

Created on Sun Feb 10 12:38:55 2019

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
import matplotlib.pyplot as plt


# %% Main


def main(pulserIP, daqName, rfAddress,
         daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd,
         daqDIPulserClock, daqDIPulserGate,
         pulserDODaqClock, pulserDODaqGate,
         pulserDOAom, pulserDORf,
         name, xCenter, yCenter, zCenter,
         freqCenter, freqRange, freqResolution, rfPower,
         readout1, readout2, numAverage):
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
        readout1: numpy.int64
            Readout time for samples with RF off (time APD channel is ungated)
            in ns
        readout2: numpy.int64
            Readout time for samples with RF on (time APD channel is ungated)
            in ns
        numAVerage: numpy.int64
            The number of scan to average over
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
    
    # We define 2D arrays, with the horizontal dimension as the frequency and
    # veritical dimension as the averaging run.
    
    countsNorm = numpy.empty([numAverage, freqResolution])
    countsNorm[:] = numpy.nan
    
    countsESR = numpy.empty([numAverage, freqResolution])
    countsESR[:] = numpy.nan
    
    countsSub = numpy.empty([numAverage, freqResolution])
    countsSub[:] = numpy.nan

    # %% Get the pulser and define the sequence

    pulser = tool_belt.get_pulser(pulserIP)

    pulserSequence = Sequence()
    low = 0
    high = 1

    # Define the period to be the whole sequence
    period = readout1 + readout2 + 200

    # Ungate the APD channel for readout1 and readout 2
    readoutTrain = [(readout1, high), (100, low), (readout2, high), (100, low)]
    pulserSequence.setDigitalChannel(pulserDODaqGate, readoutTrain)

    # Collect a sample with rf off at the end of the first and second gating
    clockTrain = [(readout1, low), (100, high), (readout2, low), (100, high)]
    pulserSequence.setDigitalChannel(pulserDODaqClock, clockTrain)

    # The AOM should always be on
    staticOnTrain = [(period, high)]
    pulserSequence.setDigitalChannel(pulserDOAom, staticOnTrain)
    
    # The RF should be off for the first measurement, and then on for the second
    # measurement
    rfTrain = [(readout1, low), (period-readout1, high)]
    pulserSequence.setDigitalChannel(pulserDORf, rfTrain)
    
    pulserSequence = pulserSequence.getSequence()
    
    # The timeout for each sample will be 1.1 * (the period in seconds)
    timeout = 1.1 * (float(period) * 10**-9)
    
    # %% Get the task list
    
    taskList = tool_belt.get_task_list()

    # %% Get the signal generator

    sigGen = tool_belt.get_VISA_instr(rfAddress)
        
    # %% "Press enter to stop..."
    tool_belt.init_safe_stop()
    
    # %% Sequence to average over
    
    for indAvg in range(numAverage):
        
        print(indAvg)
        
        scanRange = 0.02
        depthRange = 5.0
        numOptiSteps = 30
            
        ## Find the optimized position in x, y ,z
        optiCenters = find_nv_center.main(pulserIP, daqName,
    						daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd, 
                            daqDIPulserClock, daqDIPulserGate,
    						pulserDODaqClock, pulserDODaqGate,
                            pulserDOAom, name, xCenter, yCenter, zCenter,
                            scanRange, scanRange/numOptiSteps,
                            depthRange, depthRange/numOptiSteps,
                            numpy.int64(10 * 10**6), doPlot=(indAvg==0))
        
        xCenter = optiCenters[0]
        yCenter = optiCenters[1]
        zCenter = optiCenters[2]
        
        # Set the galvo and piezo position
        xyz.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, piezoSerial,
                      xCenter, yCenter, zCenter)
        time.sleep(0.5)
        
        # Set up the APD task
        streamReader, apdTask = apd.stream_read_load_daq(daqName, daqCIApd,
                                                     daqDIPulserClock,
                                                     daqDIPulserGate,
                                                     period)
        
        # Set the PulseStreamer to start on python's command. We can run the
        # loaded stream repeatedly using startNow().
        pulser.setTrigger(start=Start.SOFTWARE)
        pulser.stream(pulserSequence, 1, (0, [pulserDOAom], 0, 0))
        
        ## Collect and plot the data
    
        # Define how many sampels will be taken in each loop (2)
        samples = numpy.empty(2, dtype=numpy.uint32)
    
        # Set previousSample to 0 since we don't have any samples yet
        previousSampleNorm = 0
    
        # Take a sample and increment the frequency
        for indFreq in range(freqResolution):
    
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break
    
            sigGen.write("FREQ %fGHZ" % (freqs[indFreq]))
    
            # If this is the first frequency and first run, then we have to enable the signal
            if (indFreq == 0) and (indAvg == 0):
                sigGen.write("AMPR %fDBM" % (rfPower))
                sigGen.write("ENBR 1")
    
            # Start the timing stream
            pulser.startNow()
    
            # Read the two samples corresponding to the two readouts. This will 
            # update the values in the samples array defined above
            streamReader.read_many_sample_uint32(samples, 2, timeout)
    
            # The counter task returns the cumulative count over the life of the
            # task. We want the individual count over each sample period.
            countsNorm[indAvg][indFreq] = samples[0] - previousSampleNorm
            previousSampleNorm = samples[0]
    
            # Do the same for the second readout counts
            countsESR[indAvg][indFreq] = samples[1] - previousSampleNorm
            previousSampleNorm = samples[1]
            
            # Subtract the counts without RF from the counts with RF
            countsSub[indAvg][indFreq] = countsESR[indAvg][indFreq] / countsNorm[indAvg][indFreq]
            
        for task in taskList:
            if task.name == apdTask.name:
                taskList.remove(task)
        apdTask.close()
    
    
    # %% Average the counts over the number of averaging runs we perform
    
    countsNormAveraged = numpy.average(countsNorm, axis = 0)
    countsESRAveraged = numpy.average(countsESR, axis = 0)
    countsSubAveraged = numpy.average(countsSub, axis = 0)
    
    # %% Set up the plot

    # Create an image with 2 plots on one row, with a specified size
    # Then draw the canvas and flush all the previous plots from the canvas
    fig, axesPack = plt.subplots(1, 2, figsize=(17, 8.5))
    
    # The first plot (axesPack[0]), we will display both the RF-off and RF-on
    # counts
    ax = axesPack[0]
    ax.plot(freqs, countsNormAveraged / readout1 * 10**6, 'r-', label = "rf off")
    ax.plot(freqs, countsESRAveraged / readout2 * 10**6, 'g-', label = "rf on")
    ax.set_title('Counts with/without rf')
    ax.set_xlabel('frequency (GHz)')
    ax.set_ylabel('kcts/sec')
    ax.legend()
    # The second plot will show their subtracted values
    ax = axesPack[1]
    ax.plot(freqs,countsSubAveraged, 'b-')
    ax.set_title('Electron Spin Resonance')
    ax.set_xlabel('frequency (GHz)')
    ax.set_ylabel('contrast (arb. units)')
    
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()
    
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
               "readout1": int(readout1),
               "readout2": int(readout2),
               "numRunsToAvg": numAverage,
               "averageCounts": [countsNormAveraged.astype(int).tolist(),
                        countsESRAveraged.astype(int).tolist(),
                        countsSubAveraged.astype(float).tolist()],
               "rawCounts": [countsNorm.astype(int).tolist(),
                        countsESR.astype(int).tolist(),
                        countsSub.astype(float).tolist()]}


    filePath = tool_belt.get_file_path("find_resonance", timeStamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
