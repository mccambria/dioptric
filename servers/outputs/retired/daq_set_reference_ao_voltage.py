# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:25:30 2021

@author: kolkowitz
"""

from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
import nidaqmx
from nidaqmx.constants import AcquisitionType
import nidaqmx.stream_writers as stream_writers
import numpy
import logging
import socket



def test_define_reference():
    task_name = 'test_define_reference'
    # Create a new task
    # task = nidaqmx.Task(task_name)
    # self.task = task

    with nidaqmx.Task() as task:
        # Set up the output channels
        channel = task.ao_channels.add_ao_voltage_chan(
            "dev1/AO0", min_val=-10.0, max_val=10.0
        )
        
        #By default, the channel reference value is set to 10.0 V. 
        print(channel.ao_dac_ref_val)
        
        #We can change the reference value below to 5.0 V. For sensitive measurements, this will reduce the noise by about half!
        channel.ao_dac_ref_val = 5.0
        print(channel.ao_dac_ref_val)
    

test_define_reference()