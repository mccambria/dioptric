# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:26:41 2019

Pulsed ESR, with fixed rf frequency and variable rf on time

@author: gardill
"""

# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import Inputs.apd as apd
from pulsestreamer import PulseStreamer
from pulsestreamer import TriggerStart
from pulsestreamer import Sequence
from pulsestreamer import OutputState
import Outputs.xyz as xyz
import MajorRoutines.find_nv_center as find_nv_center

# Library modules
import numpy
import time
import matplotlib.pyplot as plt

# %% Main

def main(pulserIP, daqName, rfAddress,
         daqAOGalvoX, daqAOGalvoY, piezoSerial, 
         daqCIApd0, daqCIApd1, 
         daqDIPulserClock, daqDIPulserGate0, daqDIPulserGate1,
         pulserDODaqClock, pulserDODaqGate0, pulserDODaqGate1,
         pulserDOAom, pulserDORf,
         name, xCenter, yCenter, zCenter, AOMDelayTmp,
         rfFrequency, rfPower,
         rfMinTime, rfMaxTime, numTimeSteps, nSamples):
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

        daqCIApd0: int
            DAQ CI0 for APD
        daqCIApd1: int
            DAQ CI1 for APD
        daqCIApd3: int
            DAQ CI3 for APD

        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        daqDIPulserGate0: int
            DAQ DI for gate 0 signal from pulser
        daqDIPulserGate1: int
            DAQ DI for gate 1 signal from pulser
        daqDIPulserGate3: int
            DAQ DI for gate 3 signal from pulser

        pulserDODaqClock: int
            pulser DO carrying DAQ clock signal
        pulserDODaqGate0: int
            pulser DO carrying DAQ gate signal
        pulserDODaqGate1: int
            pulser DO carrying DAQ gate 0 signal
        pulserDODaqGate3: int
            pulser DO carrying DAQ gate 1 signal
        pulserDOAom: int
            pulser DO carrying AOM gate 3 signal
        pulserDORf: int
            pulser DO carrying RF gate signal

        name: string
            The file names consist of <date>_<time>_<name>.<ext>
        AOMDelay: int (?)
            The AOM and gate are delayed by this amount. This amount of time
            is added at the beginning of the gate and rf sequences, and added 
            to the end of of the AOM sequence
        rfFrequency: float
            The rf frequency to use (GHz)
        rfPower: float
            Power setting of the signal generator (dBm)
        rfMinTime: numpy.int64
            The rf will be applied starting at this length of time, up to rfManTime
        rfMaxTime: numpy.int64
            The longest time the rf will be applied
        numTimeSteps: float
            How many samples of time will be tested
        nSamples: float
            How many times a measurement will be taken for one value of rf time
    """
    
    # %% Initial calculations and setup
    
    # Let's make an array for the different rf on times we want to take
    tauArray = numpy.linspace(rfMinTime, rfMaxTime, num=numTimeSteps)

    # %% "Press enter to stop..."
    
    tool_belt.init_safe_stop()
    
    # %% Get the pulser and define the sequence

    pulser = PulseStreamer(pulserIP)


    low = 0
    high = 1
    
    # %% Create arrays for the signal, reference, and background counts
    
    countsSignal = numpy.empty(len(tauArray))

    countsReference = numpy.empty(len(tauArray))
    
    countsBackground = numpy.empty(len(tauArray))
    
    # %% Get the task list
    
    taskList = tool_belt.get_task_list()

    # %% Get the signal generator
    
    sigGen = tool_belt.get_VISA_instr(rfAddress)
     
    # Set the rf frequency
    sigGen.write("FREQ %fGHZ" % (rfFrequency))
    
    sigGen.write("AMPR %fDBM" % (rfPower))
    sigGen.write("ENBR 1")
    
    # %% Define the sarting counts as 0 for all counters
    
    countsSignalPrevious=0
    countsReferencePrevious=0
    countsBackgroundPrevious=0
    
    # %%
    
    # Define some times
        
    polarizationTime = numpy.int64(3 * 10**3)
    referenceTime = numpy.int64(1 * 10**3)
    signalWaitTime = numpy.int64(1 * 10**3)
    referenceWaitTime = numpy.int64(2 * 10**3)
    backgroundWaitTime = numpy.int64(1 * 10**3)
    AOMDelay = numpy.int64(AOMDelayTmp)
    gateTime = numpy.int64(300)

    
    # Total period
    totalTime = AOMDelay + polarizationTime + referenceWaitTime + referenceWaitTime + polarizationTime + referenceWaitTime + referenceTime + rfMaxTime


#    # The timeout for each sample will be 1.1 * (the period in seconds)
#    timeout = 1.1 * (float(totalTime) * 10**-9)
    
    # %% Get the task list
    
#    taskList = tool_belt.get_task_list()
    
    # %% Optimize
    
    scanRange = 0.02
    depthRange = 5.0
    numOptiSteps = 30
        
    # Find the optimized position in x, y ,z
    optiCenters = find_nv_center.main(pulserIP, daqName,
						daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd0, 
                        daqDIPulserClock, daqDIPulserGate0,
						pulserDODaqClock, pulserDODaqGate0,
                        pulserDOAom, name, xCenter, yCenter, zCenter,
                        scanRange, scanRange/numOptiSteps,
                        depthRange, depthRange/numOptiSteps,
                        numpy.int64(10 * 10**6), True)
    
    xCenter = optiCenters[0]
    yCenter = optiCenters[1]
    zCenter = optiCenters[2]
        
                
    # Set the galvo and piezo position
    xyz.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, piezoSerial,
                  xCenter, yCenter, zCenter)
        
    # %%  Set up channels
    
                    
    # Set up the APD task for the signal
    streamReaderSignal, apdTaskSignal = apd.stream_read_load_daq(daqName, daqCIApd0,
                                                 daqDIPulserClock,
                                                 daqDIPulserGate0,
                                                 totalTime, 0)
    
    # Set up the APD task for the reference
    streamReaderReference, apdTaskReference = apd.stream_read_load_daq(daqName, daqCIApd1,
                                                 daqDIPulserClock,
                                                 daqDIPulserGate1,
                                                 totalTime, 1)
    
#    # Set up the APD task for the bakcground
#    streamReaderBackground, apdTaskBackground = apd.stream_read_load_daq(daqName, daqCIApd3,
#                                                 daqDIPulserClock,
#                                                 daqDIPulserGate3,
#                                                 totalTime, 3)
#    
    # %% 
    
    # we'll want to repeat each rf time nSample amount of times 
    
    for tauIndex in range(len(tauArray)):
        
        tau = tauArray[tauIndex].astype(numpy.int64)
        print(tau)
        
        # Define times based on tau
        
        preparationTime = polarizationTime + signalWaitTime + tau + signalWaitTime
        endRestTime = rfMaxTime - tau
        pulserSequence = Sequence()
        # Define the sequence
        # Ungate (high) the APD channel for the signal
        gateSignalTrain = [(AOMDelay + preparationTime, low), (gateTime, high), 
                           (polarizationTime - gateTime + referenceWaitTime + referenceTime + backgroundWaitTime + endRestTime, low)]
        pulserSequence.setDigital(pulserDODaqGate0, gateSignalTrain)
        
        # Ungate (high) the APD channel for the reference
        gateReferenceTrain = [(AOMDelay + preparationTime + polarizationTime + referenceWaitTime, low), (gateTime, high), 
                              (referenceTime - gateTime + backgroundWaitTime + endRestTime, low)]
        pulserSequence.setDigital(pulserDODaqGate1, gateReferenceTrain)
        
#        # Ungate (high) the APD channel for the background
#        gateBackgroundTrain = [( AOMDelay + preparationTime + polarizationTime + referenceWaitTime + referenceTime + backgroundWaitTime, low), 
#                               (gateTime, high), (endRestTime - gateTime, low)]
#        pulserSequence.setDigital(pulserDODaqGate0, gateBackgroundTrain)
        
        # The AOM on (high) polarizes the NV and allows readout of the state
        aomTrain = [(polarizationTime, high), (signalWaitTime + tau + signalWaitTime, low), 
                    (polarizationTime, high), (referenceWaitTime, low), 
                    (referenceTime, high), (backgroundWaitTime + endRestTime + AOMDelay, low)]
        pulserSequence.setDigital(pulserDOAom, aomTrain)
        
        # RF on (high) mixes the states, and duration changes with stepping through T
        rfTrain = [( AOMDelay + polarizationTime + signalWaitTime, low), (tau, high), 
                   (signalWaitTime + polarizationTime + referenceWaitTime + referenceTime + backgroundWaitTime + endRestTime, low)]
        pulserSequence.setDigital(pulserDORf, rfTrain)
        

        
        # Set the PulseStreamer to start on python's command. We can run the
        # loaded stream repeatedly using startNow().
        # Lastly, stream the sequence to the pulsestreamer
        pulser.setTrigger(start=TriggerStart.SOFTWARE)
        pulser.stream(pulserSequence, nSamples, 
                      OutputState([pulserDODaqClock], 0, 0))
        
        if tauIndex == 2:
            pulserSequence.plot()
        
        pulser.startNow()
        
            
        # Read the total counts over the nSamples from the three counters, then 
        # subtract the previous counts and add to the array of counts
        signal = streamReaderSignal.read_one_sample_uint32()
        countsSignal[tauIndex] = signal - countsSignalPrevious
        countsSignalPrevious = signal
        
        reference = streamReaderReference.read_one_sample_uint32()
        countsReference[tauIndex] = reference - countsReferencePrevious
        countsReferencePrevious = reference
        
#        background = streamReaderBackground.read_one_sample_uint32()
#        countsBackground[tau] = background - countsBackgroundPrevious
#        countsBackgroundPrevious = background
            
#        for task in taskList:
#            if task.name == apdTask.name:
#                taskList.remove(task)
#                
#        apdTask.close()
        if tool_belt.safe_stop():
            break
## %% Average the counts over the iterations        
#
#    countsSignalAveraged = numpy.average(countsSignal, axis = 0)
#    countsReferenceAveraged = numpy.average(countsReference, axis = 0)
#    countsBackgroundAveraged = numpy.average(countsBackground, axis = 0)
    
# %% Calculate the Rabi data, signal / reference over different Tau, subtracting out backgound
        
    countsRabi = (countsSignal) / (countsReference) 
    
# %% Plot the Rabi signal
    
    fig, axesPack = plt.subplots(1, 2, figsize=(17, 8.5))
    
    ax = axesPack[0]
    ax.plot(tauArray, countsSignal, 'r-')
    ax.plot(tauArray, countsReference, 'g-')
#    ax.plot(tauArray, countsBackground, 'o-')
    ax.set_xlabel('rf time (ns)')
    ax.set_ylabel('Counts')
    
    ax = axesPack[1]
    ax.plot(tauArray , countsRabi, 'b-')
    ax.set_title('Normalized Signal with varying rf time')
    ax.set_xlabel('rf time (ns)')
    ax.set_ylabel('Normalized signal')
    
    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

# %% Turn off the RF and save the data

    sigGen.write("ENBR 0")
    
    timeStamp = tool_belt.get_time_stamp()

    rawData = {"timeStamp": timeStamp,
               "name": name,
               "xyzCenters": [xCenter, yCenter, zCenter],
               "rfFrequency": rfFrequency,
               "rfPower": rfPower,
               "rfMinTime": int(rfMinTime),
               "rfMaxTime": int(rfMaxTime),
               "numTimeSteps": numTimeSteps,
               "nSamples": nSamples,
               "rabi": countsRabi.astype(float).tolist(),
               "rawCounts": [countsSignal.astype(int).tolist(),
                        countsReference.astype(int).tolist()]}


    filePath = tool_belt.get_file_path("rabi", timeStamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
    
    
    
    
    
    