# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:17:21 2018

@author: cambria
"""

# Import the libraries we'll be using
import time
import nidaqmx
import nidaqmx.stream_writers as niStreamWriters
import nidaqmx.system._collections.physical_channel_collection as niPhysicalChannels
import numpy

# Declare any constant variables here
DAQ_NAME = "Dev1"

# Define functions here
def get_ao_chan_name(chanNum):
    """
    Gets the name of the physical analog output channel for the given channel number
    
    Params:
        chanNum: int
            The channel number
            
    Returns:
        int: The channel name
    """
    return DAQ_NAME + "/ao" + str(chanNum)

def get_ai_chan_name(chan_num):
    """
    Gets the name of the physical analog input channel for the given channel number
    
    Params:
        chanNum: int
            The channel number
            
    Returns:
        int: The channel name
    """
    return DAQ_NAME + "/ai" + str(chan_num)

def list_channels_and_terminals(daqName):
    """
    Prints the physical channels and terminals for the device. 
    The physical channels in our case are the channels on the PXIe-6363
    """
    aoChannels = niPhysicalChannels.AOPhysicalChannelCollection(daqName)
    print("\nAO Channels: ")
    for chan in aoChannels:
        print(chan.name, chan.ao_output_types, sep=", ")
        
    aiChannels = niPhysicalChannels.AIPhysicalChannelCollection(daqName)
    print("\nAI Channels: ")
    for chan in aiChannels:
        print(chan.name)
        
    doLines = niPhysicalChannels.DOLinesCollection(daqName)
    print("\nDO Lines: ")
    for chan in doLines:
        print(chan.name)
        
    diLines = niPhysicalChannels.DILinesCollection(daqName)
    print("\nDI Lines: ")
    for chan in diLines:
        print(chan.name)
        
    coChannels = niPhysicalChannels.COPhysicalChannelCollection(daqName)
    print("\nCO Channels: ")
    for chan in coChannels:
        print(chan.name)
        
    ciChannels = niPhysicalChannels.CIPhysicalChannelCollection(daqName)
    print("\nCI Channels: ")
    for chan in ciChannels:
        print(chan.name)
    
    device = nidaqmx.nidaqmx.system.device.Device(daqName)
    terms = device.terminals
    print("\nTerminals: ")
    for term in terms:
        print(terms.name)

############## JUST SOME STUFF I WAS MESSING AROUND WITH! ###################

# Just get the voltage from AI 0 (analog input 0)

#with nidaqmx.Task() as task:
#    task.ai_channels.add_ai_voltage_chan(DAQ_NAME + "/ai0")
#    test = task.read()
#    
#print("Running...")
#print("the voltage on AI 0 is: " + str(test))



#    # Create a task object and add an object representing the physical channel AO 0
#    writeTask = nidaqmx.Task("writeTask")
#    taskList.append(writeTask)
#    writeTask.ao_channels.add_ao_voltage_chan(get_ao_chan_name(0), min_val=-1.0, max_val=1.0)
#    
#    # Configure the timing - write one sample per second until we say stop
#    writeTask.timing.cfg_samp_clk_timing(1, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS) 
#    
#    # Loop on the array of voltages specified for 5 seconds
#    writeTask.write([.01, .02, .03, .04, .1])
#    writeTask.start()
#    time.sleep(5)
#    writeTask.stop()
#    writeTask.close() # Clears the memory associated with the task
#    taskList.pop() # Remove the task from our list

#############################################################################
    
def main(taskList):
    
    
    # Create an object representing the physical DAQ and reset it to its initialized state
    daq = nidaqmx.system.device.Device(DAQ_NAME)
    daq.reset_device()
    
    # Make a new task to stream a sine wave
    streamTask = nidaqmx.Task("streamTask")
    taskList.append(streamTask)
    streamTask.ao_channels.add_ao_voltage_chan(get_ao_chan_name(0), min_val=-5.0, max_val=5.0)
    streamTask.ao_channels.add_ao_voltage_chan(get_ao_chan_name(1), min_val=-5.0, max_val=5.0)
    
    # Set the clock rate so we can digitize a 1 kHZ wave with 100 samples per period
    samplesPerPeriod = 100
    frequency = 1
    samplesPerSecond = samplesPerPeriod*frequency
    continuous = nidaqmx.constants.AcquisitionType.CONTINUOUS
    streamTask.timing.cfg_samp_clk_timing(samplesPerSecond, samps_per_chan=samplesPerSecond*5, sample_mode=continuous) 
    
    # Expose a stream and set up a stream writer
    testStream = nidaqmx.task.OutStream(streamTask)
    streamWriter = niStreamWriters.AnalogMultiChannelWriter(testStream, auto_start=True)
    
    # Set up our digitized sine wave
    amplitude = 1.0
    offset = (0.0, 0.0) # This should be the center of the galvo
    # [0, 1, 2, ... 99]
    sampleTimes = numpy.arange(samplesPerPeriod) 
    # [sin(0), sin(2pi * 1/100), sin(2pi * 2/100), ... sin(2pi * 99/100)]
    ao0Samples = numpy.sin(2 * numpy.pi * (sampleTimes / samplesPerPeriod))
    ao0Samples = amplitude * ao0Samples # Set the amplitude to .1
    ao0Samples = ao0Samples + offset[0]
    # cosine for ao1
    ao1Samples = numpy.cos(2 * numpy.pi * (sampleTimes / samplesPerPeriod))
    ao1Samples = amplitude * ao1Samples # Set the amplitude to .1
    ao1Samples = ao1Samples + offset[1]
    samplesArray = numpy.stack((ao0Samples, ao1Samples))
    streamWriter.write_many_sample(samplesArray)
    
    print("Generating waves")
    input("Press enter to stop...")
    
    # For finite tasks call wait until done to hold execution until a task is completed
    streamTask.stop()
    #streamTask.wait_until_done() 
    streamTask.close()
    taskList.pop()
    print("Done!")
        
###################### The below runs when the script runs #######################

# Functions only run when called. Since this part of the script is not in a 
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == "__main__":
    try:
        tasks = [] # Let's keep a list of tasks so that we can clean them up if we crash out
        main(tasks)
    except Exception as e:
        print(e)
        print("We crashed out!")
    finally:
        # This will run no matter what happens in main(). Do whatever cleanup
        # is necessary here
        for task in tasks:
            task.close() 
