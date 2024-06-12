# -*- coding: utf-8 -*-
"""
Output server for the Berkeley Nucleonics 835 microwave signal generator.

Created on Wed Apr 10 12:53:38 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = test_server
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
import pyvisa as visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
import logging
import socket
from utils import common


class TestServer(LabradServer):
    name = "test_server"
    pc_name = socket.gethostname()

    def initServer(self):
        logging.info("init complete")


__server__ = TestServer()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
