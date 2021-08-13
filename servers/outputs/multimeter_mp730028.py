# -*- coding: utf-8 -*-
"""
Output server for Multicomp Pro's 5.5 digit benchtop multimeter.
Programming manual here: https://www.farnell.com/datasheets/3205713.pdf

Created on August 10th, 2021

@author: mccambria

### BEGIN NODE INFO
[info]
name = multimeter_mp730028
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 
### END NODE INFO
"""


from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
import logging
import socket
import visa


class MultimeterMp730028(LabradServer):
    name = "multimeter_mp730028"
    pc_name = socket.gethostname()
    reset_cfm_opt_out = True

    def initServer(self):
        filename = (
            "E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log"
        )
        filename = filename.format(self.pc_name, self.name)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%y-%m-%d_%H-%M-%S",
            filename=filename,
        )
        config = ensureDeferred(self.get_config())
        config.addCallback(self.on_get_config)

    async def get_config(self):
        p = self.client.registry.packet()
        p.cd(["", "Config", "DeviceIDs"])
        p.get("{}_visa_address".format(self.name))
        result = await p.send()
        return result["get"]

    def on_get_config(self, config):
        resource_manager = visa.ResourceManager()
        visa_address = config
        self.multimeter = resource_manager.open_resource(visa_address)
        self.multimeter.baud_rate = 115200
        self.power_supply.read_termination = '\n'
        self.power_supply.write_termination = '\n'
        self.multimeter.write("*RST")
        self.multimeter.write("*IDN?")
        test = self.multimeter.read()
        logging.info(test)
        logging.info("Init complete")
    
    @setting(1, res_range='s')
    def config_res_measurement(self, c, res_range):
        """This probably doesn't work yet."""
        self.multimeter.write("SENS:FUNC RES")
        res_range_options = ["500", "5E3", "50E3", "500E3", "5E6", "50E6", "500E6"]
        cmd = "CONF:SCAL:RES {}".format(res_range)
        self.multimeter.write(cmd)
        # Set the update rate to fast (maximum speed)
        self.multimeter.write("RATE F")
    
    @setting(2, detector_type='s', units='s')
    def config_temp_measurement(self, c, detector_type, units):
        
        # Set the measurement mode
        self.multimeter.write('SENS:FUNC "TEMP"')
        
        # Reset the measurement parameters and supposedly set the detector
        # type, but as far as I can tell this doesn't actually do... anything
        cmd = 'CONF:SCAL:TEMP:RTD {}'.format(detector_type)
        self.multimeter.write(cmd)
        
        # Set the detector type
        cmd = "SENS:TEMP:RTD:TYP {}".format(detector_type)
        self.multimeter.write(cmd)
        
        # Set the display type - just show the temperature
        self.multimeter.write("SENS:TEMP:RTD:SHOW TEMP")
        
        # Set the units
        cmd = "SENS:TEMP:RTD:UNIT {}".format(units)
        self.multimeter.write(cmd)
        
        # Set the update rate to fast (maximum speed)
        self.multimeter.write("RATE F")

    @setting(5, returns='v[]')
    def measure(self, c):
        """Return the value from the main display."""
        value = self.multimeter.query("MEAS1?")
        return float(value)

    @setting(6)
    def reset(self, c):
        """Fully reset to factory defaults"""
        self.multimeter.write("*RST")


__server__ = MultimeterMp730028()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
