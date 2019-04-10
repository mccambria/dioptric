# -*- coding: utf-8 -*-
"""
Piezo control functions

Created on Fri Mar  1 16:39:27 2019

@author: mccambria
"""


# %% Imports


# User modules
import Utils.tool_belt as tool_belt
import nidaqmx.stream_writers as stream_writers
from nidaqmx.constants import AcquisitionType

# Library modules
import nidaqmx
import time


# %% Miscellaneous


def check_hysteresis(rate, stepSize):
    """
    Checks whether your write rate/step size is too fast
    and will cause hystersis.
    
    Params:
        rate: float
            Rate of the write in V/s
        stepSize: float
            Size of voltage step in V/s
    """
    
    return (rate > 20.0) or (stepSize > 0.5)
        

def sleep_until_writeTime(writeTime):
    """
    Returns once/if it's past writeTime
    
    Params:
        writeTime: float
            The time to sleep until
    """
    
    currentTime = time.time()
    timeDiff = currentTime - writeTime
    if timeDiff < 0:
        # Sleep to make up the difference
        time.sleep(abs(timeDiff))


# %% Writes
        
        
def slow_write_daq(daqName, daqAOPiezo, voltage):
    """
    Write to the piezo slowly to prevent hysteresis. Internal function,
    consuming code should use write_daq or stream_write_daq
    """
    
    task = nidaqmx.Task()

    try:
        
        # Set up the output channel
        chanName = daqName + "/AO" + str(daqAOPiezo)
        task.ao_channels.add_ao_voltage_chan(chanName, min_val=0.0, max_val=10.0)
    
        # Get the current voltage of the piezo AO
        startingVoltage = read_daq(daqName, daqAOPiezo)
    
        # Define a step size and time for the voltage to increase incrementally
        # Recall the scaling is 465 nm/V
        stepSize = 0.02
        stepTime = 0.0005
    
        # Define the steps of voltage as starting with the current voltage
        nextVoltage = startingVoltage
    
        increasing = (startingVoltage < voltage)
    
        # Start the task before the loop to avoid repeatedly starting
        # and stopping it
        task.start()
    
        # Set up a loop to add the step size to the previous voltage, slowly
        # stepping up or down to the desired voltage
        nextWriteTime = time.time()
        while (abs(nextVoltage - voltage) >= stepSize):
            
            sleep_until_writeTime(nextWriteTime)
                
            if increasing:
                nextVoltage += stepSize
            else:
                nextVoltage -= stepSize
            # The next write should happen stepTime after the current time
            nextWriteTime = time.time() + stepTime
            task.write(nextVoltage)
    
        sleep_until_writeTime(nextWriteTime)
        task.write(voltage)
    
    finally:
        task.close()


def write_daq(daqName, daqAOPiezo, voltage):
    """
    Set the piezo AO to the specified voltage. Handles hysteresis by
    writing progressively and always increasing to the final value.

    Params:
        daqName: string
            The name of the DAQ
        daqAOPiezo: int
            DAQ AO carrying piezo signal
        voltage: float
            Piezo voltage to set
    """

    if (voltage < 0.0) or (voltage > 10.0):
        raise ValueError("Piezo voltage is out of range.")

    # Set the voltage below the passed value initially so that we're always
    # increasing to the final value
    voltageDiff = read_daq(daqName, daqAOPiezo) - voltage
    if voltageDiff > 0.1:
        overshootVoltage = voltage - 0.5
        if overshootVoltage >= 0.0:
            slow_write_daq(daqName, daqAOPiezo, overshootVoltage)
        else:
            slow_write_daq(daqName, daqAOPiezo, 0.0)
        
    slow_write_daq(daqName, daqAOPiezo, voltage)


def stream_write_daq(daqName, daqAOPiezo, daqDIPulserClock,
                     voltageArr, period):
    """
    Stream the voltages in voltageArr to the piezo

    Params:
        daqName: string
            The name of the DAQ
        daqAOPiezo: int
            DAQ AO carrying piezo signal
        daqDIPulserClock: int
            DAQ DI for clock signal from pulser
        voltageArr: numpy.ndarray(float)
            Piezo voltage to set, can also be a list
        period: numpy.int64
            Period of a sample in ns

    Returns:
        nidaqmx.stream_writer
        nidaqmx.task
    """

    # Write the first value to the piezo. write_daq moves slowly to prevent
    # hysteresis
    write_daq(daqName, daqAOPiezo, voltageArr[0])

    # Create the task
    # Append it to the task list so we can clean it up later
    task = nidaqmx.Task("piezo-stream_write_daq")
    taskList = tool_belt.get_task_list()
    taskList.append(task)

    chanName = daqName + "/AO" + str(daqAOPiezo)
    task.ao_channels.add_ao_voltage_chan(chanName, min_val=0, max_val=10.0)

    # Set up the output stream
    outputStream = nidaqmx.task.OutStream(task)
    streamWriter = stream_writers.AnalogSingleChannelWriter(outputStream)

    # Configure the sample to advance on the rising edge of the PFI input.
    # The frequency specified is just the max expected rate in this case.
    # We'll stop once we've run all the samples.
    daqDIPulserClockName = "PFI" + str(daqDIPulserClock)
    freq = float(1/(period*(10**-9)))  # freq in seconds as a float
    task.timing.cfg_samp_clk_timing(freq, source=daqDIPulserClockName,
                                    sample_mode=AcquisitionType.CONTINUOUS)

    # Start the task before writing so that the channel will sit on the last 
    # value when the task stops. The first sample won't actually be written
    # until the first clock signal.
    task.start()  
    
    # Write the voltages to the stream
    streamWriter.write_many_sample(voltageArr)
    
    return streamWriter, task


# %% Reads


def read_daq(daqName, daqAOPiezo):
    """
    Get the current voltage of the piezo AO.

    Params:
        daqName: string
            The name of the DAQ
        daqAOPiezo: int
            DAQ AO carrying piezo signal
    
    Returns:
        float: the current piezo voltage
    """

    task = nidaqmx.Task()

    # Set up the internal channel
    chanName = daqName + "/_ao" + str(daqAOPiezo) + "_vs_aognd"
    task.ai_channels.add_ai_voltage_chan(chanName, min_val=0, max_val=10.0)

    voltage = task.read()
    task.close()

    return voltage
