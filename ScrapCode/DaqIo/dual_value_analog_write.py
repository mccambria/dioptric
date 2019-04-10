# -*- coding: utf-8 -*-
"""
DAQ task to write a pair of voltages. Writes immediately in start.

Created on Mon Jan 14 17:30:03 2019

@author: mccambria
"""

# User modules
import Utils.tool_belt as tool_belt

# Library modules
import nidaqmx

def start(daqName, daqAOWriteToA, daqAOWriteToB, voltages):
    """
    Write the passed voltages

    Params:
        daqName: string
            The name of the DAQ that we'll be using
        daqAOWriteToA: list(int)
            DAQ AO channels to write the A values to
        daqAOWriteToB: tuple(int)
            DAQ AO channels to write the B values to
        voltages: tuple(float)
            pair of values to write to the AOs

    Returns:
        nidaqmx.task
    """
    
    # Create the task
    # Append it to the task list so we can clean it up later
    task = nidaqmx.Task("dual_value_analog_write")
    taskList = tool_belt.get_task_list()
    taskList.append(task)
    
    # Set up the output channels
    for chan in daqAOWriteToA:
        chanName = daqName + "/AO" + str(chan)
        task.ao_channels.add_ao_voltage_chan(chanName,
                                             min_val=-10.0, max_val=10.0)
    for chan in daqAOWriteToB:
        chanName = daqName + "/AO" + str(chan)
        task.ao_channels.add_ao_voltage_chan(chanName,
                                             min_val=-10.0, max_val=10.0)
    
    task.write(voltages)
    
    return task
    