"""
Virtual LabRAD server to monitor enclosure temps continuously.

Created on June 25, 2025

@author: Eric Gediman

### BEGIN NODE INFO
[info]
name = temp_monitor_SRS_ptc10
version = 1.0
description = Virtual LabRAD server to monitor enclosure temps.

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import datetime
import os
import socket
import time

import serial
from labrad.server import LabradServer, setting


class TempMonitorSrsPtc10(LabradServer):
    # writes command to get output of 4A -- where the temp controller is connected to
    # See the ptc10 manual for more info

    name = "temp_monitor_SRS_ptc10"
    port = "COM9"
    baudrate = 9600
    pc_name = socket.gethostname()

    # We will be running this constantly
    def initServer(self):
        """
        Initialize the server and Variables.

        """
        self.ser = serial.Serial(self.port, baudrate=self.baudrate)
        # Edit these as needed
        # On linux, you may have to give perms for this, use chmod 666

    @setting(0, cmd="y", val="i")
    def set_param(self, c, cmd, val):
        self.ser.write(cmd + b"=" + bytes(str(val), 'ascii') + b'\n')
        time.sleep(1)
        while not data:
            data = self.ser.readline()
            if len(data) > 0:
                return data

    @setting(0, cmd="y")
    def get_temp(self, c, cmd):
        self.ser.write(cmd)
        time.sleep(1)
        data = b""
        while not data:
            data = self.ser.readline()
            if len(data) > 0:
                # grabs the int value of temp
                result = float(data.split(b"\r")[0])
                return result

    def stopServer(self):
        """Ensure all everything is closed."""
        self.ser.close()


__server__ = TempMonitorSrsPtc10()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
