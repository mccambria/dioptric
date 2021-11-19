# -*- coding: utf-8 -*-
"""
Output server for the Cobolt 515 nm laser. 

Created on Mon Apr  8 19:50:12 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = cobolt_638
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
import nidaqmx.stream_writers as stream_writers
import numpy
import logging
import socket
from cobolt_base import CoboltBase


class Cobolt638(CoboltBase):
    wavelength = 638
    name = 'cobolt_{}'.format(wavelength)
        
        
__server__ = Cobolt638()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
