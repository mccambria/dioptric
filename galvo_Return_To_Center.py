# -*- coding: utf-8 -*-
"""
A program to set the galvo position to any position, as in (0,0)
Created on Tue Nov 20 13:32:40 2018
@author: agardill
"""


DAQ_NAME = "Dev1"

import time
import nidaqmx
import nidaqmx.stream_writers as niStreamWriters
import nidaqmx.system._collections.physical_channel_collection as niPhysicalChannels
import numpy

#name the DAQ device
DAQ_NAME = "Dev1"

################################## VARIABLES ######################################

#centering x position (in volts)
xPosition=3.0

#centering x position (in volts)
yPosition=2.0

################################# FUNCTIONS ##########################################
#obtains the physical analog outputs of the DAQ

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

#obtains the physical analog inputs of the DAQ
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

###############################################################################

def main(taskList):

            # Create an object representing the physical DAQ and reset it to its initialized state
    daq = nidaqmx.system.device.Device(DAQ_NAME)
    daq.reset_device()

        # Make a new task to stream voltage
    streamTask = nidaqmx.Task("streamTask")
    taskList.append(streamTask)
    streamTask.ao_channels.add_ao_voltage_chan(get_ao_chan_name(0), min_val=-1.0, max_val=1.0)
    streamTask.ao_channels.add_ao_voltage_chan(get_ao_chan_name(1), min_val=-1.0, max_val=1.0)

       # Expose a stream and set up a stream writer
    testStream = nidaqmx.task.OutStream(streamTask)
    streamWriter = niStreamWriters.AnalogMultiChannelWriter(testStream, auto_start=True)

       # sets voltage of galvo to preset positions
    ao0Samples=xPosition
    ao1Samples=yPosition

   # samplesArray = numpy.ndarray(ao0Samples)
    streamWriter.write_one_sample(ao0Sample)

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
