# -*- coding: utf-8 -*-
"""
APD control functions

Created on Mon Mar  4 11:07:58 2019

@author: mccambria
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt

# Library modules
import time
import numpy
import nidaqmx
import nidaqmx.stream_readers as stream_readers
from nidaqmx.constants import TriggerType
from nidaqmx.constants import Level
from nidaqmx.constants import AcquisitionType


# %% Load


def stream_read_load_daq(daqName, daqCIReadFrom, daqDIPulserClock,
                         daqDIPulserGate, period, daqCtrChan=None):
    """
    Load the task. The task starts immediately, so you'll want to
    discard the first value you read. Note that the task returns cumulative
    counts over the entire task.

    Params:
        daqName: string
            The name of the DAQ that we'll be using
        daqCIReadFrom: int
            PFI to use as the counter source (connected to APD)
        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        daqDIPulserGate: int
            DAQ DI for gate signal from pulser
        period: numpy.int64
            Period of a sample in ns
        daqCtrChan: int
            Index for the counter channel id, eg 1 for dev1/ctr1,
            defaults to 0

    Returns:
        nidaqmx.stream_reader
        nidaqmx.task
    """

    # Create the task
    # Append it to the task list so we can clean it up later
    if daqCtrChan == None:
        daqCtrChan = 0
    task = nidaqmx.Task("apd-stream_read_load_daq_" + str(daqCtrChan))
    taskList = tool_belt.get_task_list()
    taskList.append(task)

    chanName = daqName + "/ctr" + str(daqCtrChan)
    chan = task.ci_channels.add_ci_count_edges_chan(chanName)
    chan.ci_count_edges_term = "PFI" + str(daqCIReadFrom)

    # Set up the input stream
    inputStream = nidaqmx.task.InStream(task)
    streamReader = stream_readers.CounterReader(inputStream)
    # Just collect whatever data is available when we read
    streamReader.verify_array_shape = False

    # Set up the gate ("pause trigger")
    if daqDIPulserGate is not None:
        daqDIPulserGateName = "PFI" + str(daqDIPulserGate)
        # Pause when low - i.e. read only when high
        task.triggers.pause_trigger.trig_type = TriggerType.DIGITAL_LEVEL
        task.triggers.pause_trigger.dig_lvl_when = Level.LOW
        task.triggers.pause_trigger.dig_lvl_src = daqDIPulserGateName

    # Configure the sample to advance on the rising edge of the PFI input.
    # The frequency specified is just the max expected rate in this case.
    # We'll stop once we've run all the samples.
    daqDIPulserClockName = "PFI" + str(daqDIPulserClock)
    freq = float(1/(period*(10**-9)))  # freq in seconds as a float
    task.timing.cfg_samp_clk_timing(freq, source=daqDIPulserClockName,
                                    sample_mode=AcquisitionType.CONTINUOUS)

    # Start the task. It will start counting immediately so we'll have to
    # discard the first sample.
    task.start()

    return streamReader, task


# %% Read


def stream_read_daq(counterStreamReader, totalNumSamples, timeout,
                    callback=None, *callbackArgs):
    """
    Read the counter stream and fire the callback after each read

    Params:
        counterStreamReader: nidaqmx.stream_readers.CounterReader
            The counter stream reader we'll use to collect counts
        totalNumSamples: int
            The total number of samples we'll be collecting
        timeout: int
            Seconds to wait before quitting whether we collected all the
            samples or not
        callback: function
            The function to run at the end of each loop
        callbackArgs: tuples
            Packed arguments to pass to the callback
            
    Returns:
        numpy.ndarray: The counts
    """

    # Initialize the necessary values
    curNumSamples = 0
    startTime = time.time()

    # The DAQ only fills values [0: number of samples read] of whatever you
    # pass it so let's manually tack the new values onto a running list of
    # all the samples we've collected
    allSamplesCum = numpy.zeros(totalNumSamples, dtype=numpy.uint32)
    allSamplesDiff = numpy.zeros(totalNumSamples, dtype=numpy.uint32)

    # Something funny is happening if we read more than 1000 samples in a loop
    bufferSize = min(totalNumSamples, 1000)

    # The counter task begins counting as soon as the task starts.
    # The AO channel writes its first samples only on the first clock
    # signal after the task starts. This means there's one
    # sample from the counter stream that we don't want to record.
    # We do need it for a calculation below, however.
    firstValue = counterStreamReader.read_one_sample_uint32()

    tool_belt.init_safe_stop()

    while curNumSamples < totalNumSamples:

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        # Break out of the while if we've timed out
        elapsedTime = time.time() - startTime
        if elapsedTime > timeout:
            print("Timed out after " + str(timeout) + " seconds.")
            break

        # Initialize/reset the read sample array to its maximum possible size.
        newSamplesCum = numpy.zeros(bufferSize, dtype=numpy.uint32)

        # Read the samples currently in the DAQ memory.
        numNewSamples = counterStreamReader.read_many_sample_uint32(newSamplesCum)
        if numNewSamples == bufferSize:
            raise Warning("The DAQ buffer contained more samples than "
                          "expected. Validate your parameters and "
                          "increase bufferSize if necessary.")

        # Check if we collected more samples than we need, which may happen
        # if the pulser runs longer than necessary. If so, just to throw out
        # excess samples.
        if curNumSamples + numNewSamples > totalNumSamples:
            numNewSamples = totalNumSamples - curNumSamples

        # Tack the new samples onto all samples
        newSamplesCum = newSamplesCum[0: numNewSamples]
        allSamplesCum[curNumSamples: curNumSamples + numNewSamples] = newSamplesCum

        # The DAQ counter reader returns cumulative counts, which is not what
        # we want. So we have to calculate the difference between samples
        # n and n-1 in order to get the actual count for the nth sample.
        newSamplesDiff = numpy.zeros(numNewSamples)
        for index in range(numNewSamples):

            currentIndex = curNumSamples + index
            currentValue = allSamplesCum[currentIndex]

            previousIndex = currentIndex - 1
            if previousIndex >= 0:
                previousValue = allSamplesCum[previousIndex]
            else:
                previousValue = firstValue

            newSamplesDiff[index] = currentValue - previousValue

        allSamplesDiff[curNumSamples: curNumSamples + numNewSamples] = newSamplesDiff

        # Update the current count
        curNumSamples = curNumSamples + numNewSamples

        # Run the callback
        if callback is not None:
            callback(newSamplesDiff, curNumSamples, *callbackArgs)

    return allSamplesDiff
