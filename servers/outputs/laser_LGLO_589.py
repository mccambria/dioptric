# -*- coding: utf-8 -*-
"""
Interface server for Laserglow 589 to control analog voltage from DAQ and gated
by Pulse Streamer

Created on January 24th, 2023

@author: gardill

### BEGIN NODE INFO
[info]
name = laser_LGLO_589
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
import nidaqmx
from nidaqmx.constants import AcquisitionType
import nidaqmx.stream_writers as stream_writers
import numpy as np
import logging
import socket
from laser_LGLO_base import LaserLgloBase


class LaserLglo589(LaserLgloBase):
    wavelength = 589
    name = f"laser_LGLO_{wavelength}"
        
        
__server__ = LaserLglo589()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
    
    
    