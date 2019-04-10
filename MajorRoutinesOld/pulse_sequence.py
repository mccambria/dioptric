# -*- coding: utf-8 -*-
"""
This routine is utilizing multiple counter channels on the daq. 

Currently it is being used to see the delay time between the AOM and the gates.

Created on Tue Mar 26 12:38:55 2019

@author: kolkowitz
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
#import time
import matplotlib.pyplot as plt


# %% Main


def main(pulserIP, daqName, rfAddress,
         daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd0, daqCIApd1,
         daqDIPulserClock, daqDIPulserGate0,daqDIPulserGate1,
         pulserDODaqClock, pulserDODaqGate0, pulserDODaqGate1,
         pulserDOAom, pulserDORf,
         name, xCenter, yCenter, zCenter,
         startTime, endTime, numSteps,
         nSamples):
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

    t = numpy.linspace(startTime, endTime, num=numSteps)
    
    # As a test, flip the freqs
#    freqs = numpy.flip(freqs)

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    
    # We define 2D arrays, with the horizontal dimension as the frequency and
    # veritical dimension as the averaging run.
    
#    # %% Get the task list
#    
#    taskList = tool_belt.get_task_list()
#    
    # %% Get the signal generator

#    sigGen = tool_belt.get_VISA_instr(rfAddress)
        
    # %% "Press enter to stop..."
    tool_belt.init_safe_stop()
    
    #%% Optimize
    
    scanRange = 0.02
    depthRange = 5.0
    numOptiSteps = 30
        
    ## Find the optimized position in x, y ,z
    optiCenters = find_nv_center.main(pulserIP, daqName,
						daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd0, 
                        daqDIPulserClock, daqDIPulserGate0,
						pulserDODaqClock, pulserDODaqGate0,
                        pulserDOAom, name, xCenter, yCenter, zCenter,
                        scanRange, scanRange/numOptiSteps,
                        depthRange, depthRange/numOptiSteps,
                        numpy.int64(10 * 10**6), doPlot=True)
    
    xCenter = optiCenters[0]
    yCenter = optiCenters[1]
    zCenter = optiCenters[2]
    
    # Set the galvo and piezo position
    xyz.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, piezoSerial,
                  xCenter, yCenter, zCenter)

    # %% Get the pulser and define the sequence
    
    pulser = tool_belt.get_pulser(pulserIP)
    
        # Set up the first APD task
    streamReader0, apdTask0 = apd.stream_read_load_daq(daqName, daqCIApd0,
                                                 daqDIPulserClock,
                                                 daqDIPulserGate0,
                                                 1000, 0)
    
    streamReader1, apdTask1 = apd.stream_read_load_daq(daqName, daqCIApd1,
                                                 daqDIPulserClock,
                                                 daqDIPulserGate1,
                                                 1000, 1)
    counts0 = numpy.linspace(startTime, endTime, num=numSteps)
    counts1 = numpy.linspace(startTime, endTime, num=numSteps)
    timeArray = numpy.linspace(startTime, endTime, num=numSteps)
    print(timeArray)
    low = 0
    high = 1
    
    x=0
    
#    clockSequence = Sequence()
#    clockTrain = [(100, low), (100, high), (100, low)]
#    clockSequence.setDigitalChannel(pulserDODaqClock, clockTrain)
#
#    clockSequence = clockSequence.getSequence()
    
    counts0prev=0
    counts1prev=0
    
    for i in range(len(t)):
    
        pulserSequence = Sequence()
        tmpTime = t[i].astype(int)
        print(tmpTime)
        # Gating for the reference
        gate1Train = [(2000, low), (300, high), (700, low)]
        pulserSequence.setDigitalChannel(pulserDODaqGate1, gate1Train)
        
        #Gating for the signal
        gate0Train = [(tmpTime, low), (300, high), (100, low)]
        pulserSequence.setDigitalChannel(pulserDODaqGate0, gate0Train)

        # The AOM should always be on
        staticOnTrain = [(1000, low), (1000, high), (2000, low)]
        pulserSequence.setDigitalChannel(pulserDOAom, staticOnTrain)
        
        
        pulserSequence = pulserSequence.getSequence()
        
        # The timeout for each sample will be 1.1 * (the period in seconds)
#        timeout = 1.1 * (float(period) * 10**-9)
        
        # Set the PulseStreamer to start on python's command. We can run the
        # loaded stream repeatedly using startNow().
        pulser.setTrigger(start=Start.SOFTWARE)
        pulser.stream(pulserSequence, nSamples, (0, [pulserDOAom,pulserDODaqClock], 0, 0))
        # Start the timing stream
        pulser.startNow()
                
        pulser.setTrigger(start=Start.SOFTWARE)
        
#        
#        pulser.stream(clockSequence, 1, (0, [pulserDOAom], 0, 0))
#        
#        pulser.startNow()
#    
        # Read the two samples corresponding to the two readouts. This will 
        # update the values in the samples array defined above
        counts0tmp = streamReader0.read_one_sample_uint32()
        counts1tmp = streamReader1.read_one_sample_uint32()
        
        counts0[x]=counts0tmp-counts0prev
        counts0prev=counts0tmp
        
        counts1[x]=counts1tmp-counts1prev
        counts1prev=counts1tmp
        
        print(x)
        print(counts0[x])
        print(counts1[x])
        x=x+1
              # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break
    print(timeArray)
    print(counts0)
    print(counts1)
    countsNorm=counts0/counts1
    # %% Set up the plot

    # Create an image with 2 plots on one row, with a specified size
    # Then draw the canvas and flush all the previous plots from the canvas
    fig, axesPack = plt.subplots(1, 2, figsize=(17, 8.5))
    
    # The first plot (axesPack[0]), we will display both the RF-off and RF-on
    # counts
    ax = axesPack[0]
    ax.plot(timeArray, counts1, 'r-', label = "counts1")
    ax.plot(timeArray, counts0, 'g-', label = "counts0")
    ax.set_title('Counts')
    ax.set_xlabel('t')
    ax.set_ylabel('counts')
    ax.legend()
    # The second plot will show their normalized values
    ax = axesPack[1]
    ax.plot(t,countsNorm, 'b-')
    ax.set_title('Counts')
    ax.set_xlabel('t')
    ax.set_ylabel('counts')
    
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()
    
    # %% Turn off the RF and save the data
    
    timeStamp = tool_belt.get_time_stamp()

    rawData = {"timeStamp": timeStamp,
               "name": name,
               "xyzCenters": [xCenter, yCenter, zCenter],
               "times": timeArray.astype(int).tolist(),
               "Counts": [counts0.astype(int).tolist(),
                        counts1.astype(int).tolist(),
                        countsNorm.astype(float).tolist()]}


    filePath = tool_belt.get_file_path("timing_pulses_sequence", timeStamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
