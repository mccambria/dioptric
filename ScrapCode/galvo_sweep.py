# -*- coding: utf-8 -*-
"""
This program sweeps the galvo over a square grid. The file can also be imported
as a library if you're looking to use the functions it contains. The program
parameters are defined below. The sweep is conducted from left to right (like
reading a book).

Created on Tue Nov 13 19:11:28 2018

@author: mccambria
"""

######################## Program parameters here ########################

# Modify the values below to change the program parameters
# Numbers should be floats unless otherwise specified

# The IP address adopted by the PulseStreamer is hardcoded. See the lab wiki
# for information on how to change it
PULSE_STREAMER_IP = "128.104.160.11"
DAQ_NAME = "dev1" # This is the automatically assigned name of the DAQ

# Galvo voltage range for a given direction is
# (number of samples in that direction) * (volts per step between samples)
SAMPLES_PER_DIM = 100 # number of samples in each direction - can be an int
RESOLUTION = .01 # volts per step between samples

# Offset voltages to align the top left of the grid
OFFSET_X = -0.5
OFFSET_Y = -0.5

# Initial values for the galvo to sit at before we start the PulseStreamer
# This doesn't really matter, it should just be different than the first value
INITIAL_X = 0.0
INITIAL_Y = 0.0

PERIOD = .250 # Period of a sample in seconds

########################### Import statements here ###########################

import sys
import os
# By default, python looks for modules in ...installDirectoy\Lib\site-packages
# We can tell it to additionally look elsewhere by appending a path to sys.path
# pulse_streamer_grpc does not live in the default directory so we need to add
# that path before we import the library
sys.path.append(os.getcwd() + '/PulseStreamerExamples/python/lib')
from pulse_streamer_grpc import PulseStreamer
import nidaqmx
import nidaqmx.stream_writers as niStreamWriters
import nidaqmx.stream_readers as niStreamReaders
import NV_utils
from NV_utils import DAQChannelTypes as DAQChannelTypes
import numpy

######################## Functions and other defs here ########################



########################### Define main ###########################

def main(pulser, daq, taskList):
    """
    This is the code that will run when you run this file as a program.
    For more info on how main works, see the very end of this file

    Params:
        pulser: PulseStreamer
            The PulseStreamer that we'll be using
        daq: nidaqmx.system.device.Device
            The DAQ that we'll be using
        taskList: list
            The list that we'll populate with active tasks so
            we can clean them up if main crashes
    """

    ##################### Set up the galvo values #####################

    # Calculate the total number of samples we'll collect
    totalSamples = SAMPLES_PER_DIM^2

    # Set up vectors for the number of samples in each direction
    # [0, 1, 2, ... samplesPerDim]
    stepsX = numpy.arange(SAMPLES_PER_DIM)
    stepsY = numpy.arange(SAMPLES_PER_DIM)

    # Apply scale and offsets to get the voltages we'll apply to the galvo
    # Note that the x/y angles, not the actual x/y positions are linear
    # in these voltages. We don't care about this since our angular range is
    # is small enough to warrant sin(theta) ~= theta.
    voltagesX = (RESOLUTION * stepsX) + OFFSET_X
    voltagesY = (RESOLUTION * stepsY) + OFFSET_Y

    # Get the 2 x totalSamples array of voltages to apply to the galvo
    galvoVoltages = NV_utils.cartesian_product(voltagesX, voltagesY)

    # Add a dummy value that the galvo will sit at until
    # we start the PulseStreamer
    galvoInitials = numpy.array([INITIAL_X, INITIAL_Y])
    numpy.insert(galvoVoltages, galvoInitials, 0, axis=0)

    ##################### Set up the DAQ #####################

    # Create the task
    sweepTask = nidaqmx.Task("sweepTask")
    taskList.append(sweepTask)

    # Set up the output channels
    ao0Name = NV_utils.get_DAQ_chan_name(DAQ_NAME, DAQChannelTypes.AO, 0)
    sweepTask.ao_channels.add_ao_voltage_chan(ao0Name,
                                              min_val=-1.0,
                                              max_val=1.0)
    ao1Name = NV_utils.get_DAQ_chan_name(DAQ_NAME, DAQChannelTypes.AO, 1)
    sweepTask.ao_channels.add_ao_voltage_chan(ao1Name,
                                              min_val=-1.0,
                                              max_val=1.0)

    # Set up the output stream to the galvo
    outputStream = nidaqmx.task.OutStream(sweepTask)
    streamWriter = niStreamWriters.AnalogMultiChannelWriter(outputStream,
                                                            auto_start=True)

    # Configure the sample to advance on the rising edge of the PFI0 input
    # Assume a max sample rate of 4 samples per channel per second
    # We'll stop once we've run all the samples
    freq = 1/PERIOD
    sweepTask.timing.cfg_samp_clk_timing(freq,
                                         source="PFI0",
                                         samps_per_chan=totalSamples)

    # Write the galvo voltages to the stream
    streamWriter.write_many_sample(galvoVoltages)

    ##################### Run the PulseStreamer #####################

    periodNano = int(PERIOD * (10^9)) # integer period in ns
    NV_utils.run_pulse_streamer_square_wave(pulser, 
                                            periodNano, 
                                            [0], 
                                            totalSamples)


#################### The below runs when the script runs #####################

# Functions only run when called. Since this part of the script is not in a
# function, it will run when the script is run.
# __name__ will only be __main__ if we're running the file as a program.
# The below pattern enables us to import this file as a module without
# running it as a program.
if __name__ == "__main__":

    # Attempt to get the PulseStreamer from the given IP
    # Exit the program if we can't get it
    try:
        pulser = PulseStreamer(PULSE_STREAMER_IP)
        print(pulser.isRunning())
    except Exception as e:
        print(e)
        print("No PulseStreamer found at IP-address: " + PULSE_STREAMER_IP)
        sys.exit()

    # Same for the DAQ
    try:
        daq = nidaqmx.system.device.Device(DAQ_NAME)
        daq.reset_device()
    except Exception as e:
        print(e)
        print("No DAQ named: " + DAQ_NAME)
        sys.exit()

    # Run the main program
    try:
        # Let's keep a list of tasks so that
        # we can clean them up if we crash out
        taskList = []
        main(pulser, daq, taskList)
    except Exception as e:
        print(e)
        print("We crashed out!")
    finally:
        # This will run no matter what happens in main()
        # Do whatever cleanup is necessary here
        for task in taskList:
            task.close()