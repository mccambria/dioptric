# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:00:03 2019

A sequence to measure the delay between the rf signal and the aom

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
#import time
import matplotlib.pyplot as plt

# %% Main


def main(pulserIP, daqName, rfAddress,
         daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd0, daqCIApd1,
         daqDIPulserClock, daqDIPulserGate0,daqDIPulserGate1,
         pulserDODaqClock, pulserDODaqGate0, pulserDODaqGate1,
         pulserDOAom, pulserDORf,
         name, xCenter, yCenter, zCenter,
         AOMDelayTmp,
         rfFrequency, rfPower, rfPiPulse,
         startTime, endTime, numTimeSteps,
         numSamples, numAverage):
    
    # %% Create an array of time to use in delaying the rf pi pulse
    
    timeArray = numpy.linspace(startTime, endTime, num=numTimeSteps)
    
    # %% Convert piPulse into int64
    
    rfPiPulse = numpy.int64(rfPiPulse)
    
    # %% Get the signal generator
    
    sigGen = tool_belt.get_VISA_instr(rfAddress)
     
    # Set the rf frequency
    sigGen.write("FREQ %fGHZ" % (rfFrequency))
    
    sigGen.write("AMPR %fDBM" % (rfPower))
    sigGen.write("ENBR 1")
    
    # %% "Press enter to stop..."
    tool_belt.init_safe_stop()
    
    # %% Get the task list
    
    taskList = tool_belt.get_task_list()
    
    # %% Get Pulser

    pulser = PulseStreamer(pulserIP)  
    
    low = 0
    high = 1
    
    # %% Define values
           
    polarizationTime = numpy.int64(1000)
    gateTime = numpy.int64(300)
    AOMDelay = numpy.int64(AOMDelayTmp)
    waitTime = numpy.int64(1000)
    rfDelay = numpy.int(40)
    
    # %% Create the arrays
    
    countsSignal = numpy.empty([numAverage, len(timeArray)])
    countsReference = numpy.empty([numAverage, len(timeArray)])    
    
    #%% Optimize
    
#    scanRange = 0.02
#    depthRange = 5.0
#    numOptiSteps = 30
#        
#    ## Find the optimized position in x, y ,z
#    optiCenters = find_nv_center.main(pulserIP, daqName,
#						daqAOGalvoX, daqAOGalvoY, piezoSerial, daqCIApd0, 
#                        daqDIPulserClock, daqDIPulserGate0,
#						pulserDODaqClock, pulserDODaqGate0,
#                        pulserDOAom, name, xCenter, yCenter, zCenter,
#                        scanRange, scanRange/numOptiSteps,
#                        depthRange, depthRange/numOptiSteps,
#                        numpy.int64(10 * 10**6), doPlot=True)
#    
#    xCenter = optiCenters[0]
#    yCenter = optiCenters[1]
#    zCenter = optiCenters[2]
    
    # Set the galvo and piezo position
    xyz.write_daq(daqName, daqAOGalvoX, daqAOGalvoY, piezoSerial,
                  xCenter, yCenter, zCenter)

    # %% Get the pulser and define the sequence
    
    for indAvg in range(numAverage):
    
        print('Averaging Index: ' + str(indAvg))
    
        tool_belt.init_safe_stop()
    
        # Set up the first APD task
        streamReaderSignal, apdTaskSignal = apd.stream_read_load_daq(daqName, daqCIApd0,
                                                     daqDIPulserClock,
                                                     daqDIPulserGate0,
                                                     5000 * numSamples, 0)
        
        streamReaderReference, apdTaskReference = apd.stream_read_load_daq(daqName, daqCIApd1,
                                                     daqDIPulserClock,
                                                     daqDIPulserGate1,
                                                     5000 * numSamples, 1)    
        
        countsSignalPrev=0
        countsReferencePrev=0
        
        for timeIndex in range(len(timeArray)):
            time = timeArray[timeIndex].astype(numpy.int64)
            
            print(time)
            
            pulserSequence = Sequence()
    
            # Define the sequences        
            signalTrain = [(rfDelay + AOMDelay + polarizationTime + waitTime, low), (gateTime, high), (polarizationTime - gateTime + waitTime + polarizationTime, low)]
            pulserSequence.setDigital(pulserDODaqGate0, signalTrain)
            
            referenceTrain = [(rfDelay + AOMDelay + polarizationTime + waitTime + polarizationTime + waitTime, low), (gateTime, high), (polarizationTime - gateTime, low)]
            pulserSequence.setDigital(pulserDODaqGate1, referenceTrain)       
            
            AOMTrain = [(rfDelay + polarizationTime, high), (waitTime, low), (polarizationTime, high), (waitTime, low), (polarizationTime, high)]
            pulserSequence.setDigital(pulserDOAom, AOMTrain)
            
            rfTrain = [(AOMDelay + time, low), (rfPiPulse, high), ( - time - rfPiPulse + polarizationTime + waitTime + polarizationTime + waitTime + polarizationTime, low)]
            pulserSequence.setDigital(pulserDORf, rfTrain)  
      

          
            # Set the PulseStreamer to start on python's command. We can run the
            # loaded stream repeatedly using startNow().        
#            pulserSequence = pulserSequence.getSequence()
    
            
            pulser.setTrigger(start=TriggerStart.SOFTWARE)
            pulser.stream(pulserSequence, numSamples, 
                          OutputState([pulserDOAom,pulserDODaqClock], 0, 0)) ##
            
            if (timeIndex == 2) and (indAvg == 0):
                pulserSequence.plot()
                
            # Start the timing stream        
            pulser.startNow()
                    

            
            # Read in the counts
            countsSignaltmp = streamReaderSignal.read_one_sample_uint32()
            countsReferencetmp = streamReaderReference.read_one_sample_uint32()
            
            countsSignal[indAvg][timeIndex] = countsSignaltmp - countsSignalPrev
            countsSignalPrev = countsSignaltmp
            
            countsReference[indAvg][timeIndex] = countsReferencetmp - countsReferencePrev
            countsReferencePrev = countsReferencetmp
            
            print(countsSignal[indAvg][timeIndex])
            print(countsReference[indAvg][timeIndex])
            
            if tool_belt.safe_stop():
                break
                
        for task in taskList:
            if task.name == apdTaskSignal.name:
                taskList.remove(task)
        apdTaskSignal.close() 
        
        for task in taskList:
            if task.name == apdTaskReference.name:
                taskList.remove(task)    
        apdTaskReference.close()       

    
    # %% Average the counts
    
    countsSignalAveraged = numpy.average(countsSignal, axis = 0)
    countsReferenceAveraged = numpy.average(countsReference, axis = 0)    
    
    countsNormalizedAveraged = countsSignalAveraged / countsReferenceAveraged
    
    # %% Plot the Rabi signal
    
    fig, axesPack = plt.subplots(1, 2, figsize=(17, 8.5))
    
    ax = axesPack[0]
    ax.plot(timeArray, countsSignalAveraged, 'r-', label = 'signal')
    ax.plot(timeArray, countsReferenceAveraged, 'g-', label = 'reference')
    ax.set_xlabel('delay of pi pulse (ns)')
    ax.set_ylabel('Counts')
    ax.legend()
    
    ax = axesPack[1]
    ax.plot(timeArray , countsNormalizedAveraged, 'b-')
    ax.set_xlabel('delay of pi pulse (ns)')
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
               "piPulse": rfPiPulse.astype(float),
               "startTime": int(startTime),
               "endTime": int(endTime),
               "numTimeSteps": numTimeSteps,
               "numSamples": numSamples,
               "numAverage": numAverage,
               "timeArray": timeArray.astype(int).tolist(),
               "normalizedCounts": countsNormalizedAveraged.astype(float).tolist(),
               "rawAveragedCounts": [countsSignalAveraged.astype(int).tolist(),
                        countsReferenceAveraged.astype(int).tolist()],
               "rawCounts": [countsSignal.astype(int).tolist(),
                        countsReference.astype(int).tolist()]}


    filePath = tool_belt.get_file_path("rf_delay", timeStamp, name)
    tool_belt.save_figure(fig, filePath)
    tool_belt.save_raw_data(rawData, filePath)
        
        
        
        
        
        
        
        
        
    