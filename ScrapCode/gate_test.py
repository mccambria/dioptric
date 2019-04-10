# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:27:21 2019

@author: mccambria
"""

import sys
import os
sys.path.append(os.getcwd() + "\..")
import nidaqmx
import nidaqmx.stream_readers as niStreamReaders
import cfm_utils
from nidaqmx.constants import TerminalConfiguration
from nidaqmx.constants import Level

try:
    apdTask = nidaqmx.Task("apdTask")
    taskList = cfm_utils.get_task_list()
    taskList.append(apdTask)
    
    chanName = "dev1/ctr2"
    chan = apdTask.ci_channels.add_ci_count_edges_chan(chanName)
    apdTask.ci_ctr_timebase_rate = 100000000
    apdTask.ci_count_edges_gate_term = "PFI1"

finally:
    cfm_utils.task_list_close_all()