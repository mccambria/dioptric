# -*- coding: utf-8 -*-
"""
Output server for the Integrated Optics 520 nm laser. Controlled by the DAQ.

Created on Mon Apr  8 19:50:12 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = laser_INTE_520
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

from laser_COBO_base import LaserCoboBase


class LaserInte520(LaserCoboBase):
    name = 'laser_INTE_520'
        
__server__ = LaserInte520()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
