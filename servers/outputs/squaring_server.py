# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:35:18 2019

@author: Matt

### BEGIN NODE INFO
[info]
name = Squaring Server
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


import time
from labrad.server import LabradServer, setting


class SquaringServer(LabradServer):
    name = "Squaring Server"

    @setting(10, data='v[]', returns='v[]')
    def square(self, c, data):
        time.sleep(2)
        return data**2


__server__ = SquaringServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
