import datetime
import os
import socket
import time

import serial
from labrad.server import LabradServer, setting

"""
Virtual LabRAD server to monitor enclosure temps continuously.

Created on June 25, 2025

@author: Eric Gediman

### BEGIN NODE INFO
[info]
name = SRS_PTC_10
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


class EnclosureTemp(LabradServer):
    # writes command to get output of 4A -- where the temp controller is connected to
    # See the ptc10 manual for more info

    port = "COM9"
    baudrate = 9600
    output_file = ""
    # Os specific, edit as needed

    # edit this as needed
    nv_folder_path = "G:\\Enclosure_Temp"
    ser = serial.Serial(port, baudrate=baudrate)
    name = "enclosure_continuous_temp"
    pc_name = socket.gethostname()

    # We will be running this constantly
    def initServer(self):
        """
        Initialize the server and Variables.

        """
        self.run_templog()
        # Edit these as needed
        # On linux, you may have to give perms for this, use chmod 666

    def run_templog(self):
        if (
            datetime.datetime.now().strftime("%m%Y")
            != os.path.split(os.path.split(self.output_file)[0])[1]
        ):
            self.output_file = os.path.join(
                self.nv_folder_path,
                datetime.datetime.now().strftime("%m%Y"),
                "temp_data",
            )
        if not os.path.isdir(
            os.path.join(
                self.nv_folder_path, datetime.datetime.now().strftime("%m%Y")
            )
        ):
            os.mkdir(
                    os.path.join(
                    self.nv_folder_path, datetime.datetime.now().strftime("%m%Y")
                    )
            )

        file = open(self.output_file, "a")
        self.ser.write(b"4A?\n")
        time.sleep(1)
        data = b""
        while not data:
            data = self.ser.readline()
            if len(data) > 0:
                # grabs the int value of temp
                result = float(data.split(b"\r")[0])
                print(
                    str(result)
                    + ","
                    + datetime.datetime.now().strftime("%d:%H:%M:%S"),
                    file=file,
                )
            file.close()

    def stopServer(self):
        """Ensure all everything is closed."""
        self.ser.close()


__server__ = EnclosureTemp()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
