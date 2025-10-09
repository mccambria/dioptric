# -*- coding: utf-8 -*-
"""
Output server for the R&S NGC103 desktop power supply.

Created on Wed Feb 26 09:19:48 2025

@author: rcantuv, sean, quinn, 
LabRad wrapper of '/dioptric/nv-field-control/actual-field-control/fieldcontrol.py'

### BEGIN NODE INFO
[info]
name = power_supply_RNS_ngc103
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
import sys

sys.path.append(r"C:\Users\kolko\Downloads\rsinstrument-1.102.0\rsinstrument-1.102.0")
import logging
import socket
import time

import pyvisa as visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
from labrad.server import LabradServer, setting
from RsInstrument import RsInstrument
import RsInstrument as Rs
from twisted.internet.defer import ensureDeferred

from utils import common
from utils import tool_belt as tb

import numpy as np
from datetime import datetime
from typing import Union
class PowerSupplyRnsNgc103(LabradServer, RsInstrument):
    name = "power_supply_RNS_ngc103"
    pc_name = socket.gethostname()

    def initServer(self, startOpen=False):
        tb.configure_logging(self)
        config = common.get_config_dict()
        self.device_id = config["DeviceIDs"][f"{self.name}_visa"]
        # di_clock = config["Wiring"]["Daq"]["di_clock"]
        # resource_manager = visa.ResourceManager()
        # self.pwr_sup = resource_manager.open_resource(device_id)
        # Set the VISA read and write termination. This is specific to the
        # instrument - you can find it in the instrument's programming manual
        # self.pwr_sup.read_termination = "\n"
        # self.pwr_sup.write_termination = "\n"
        # Set our channels for FM
        # self.daq_di_pulser_clock = di_clock
        # self.daq_ao_sig_gen_mod = config[2]
        # self.task = None  # Initialize state variable
        # self.reset(None)
        # self._set_freq(2.87)
        logging.info("Init complete")
        
        self.open = False
        self.instr = None
        if startOpen:
            self.open_connection()
            
    def _write_command(self, cmd : str) -> None:
        """
        Write buffered command to the instrument.

        :return None:
        """
        # try:
        self.instr.write_str(cmd + "; *WAI")
        # except
        
    def _query_command(self, cmd : str) -> any:
        """
        Query buffered command.

        :return any:
        """
        return self.instr.query_str(cmd + "; *WAI")

    @setting(0)
    # def open_connection(self, c, IP : str = None, direction_channels : dict[str, int] = {"x": 1, "y": 2, "z": 3}) -> None:
    def open_connection(self) -> None:
        """
        Open connection with the device. Ensure to call this before attempting to use the device.

        :return None:
        """
        IP = self.device_id
        direction_channels = {"x": 1, "y": 2, "z": 3}

        self.IP = IP
        self.direction_channels = direction_channels
            
        self.open = True
        Rs.RsInstrument.assert_minimum_version('1.50.0')
        IP=None
        if IP == None:
            self.instr = Rs.RsInstrument('TCPIP::192.168.56.101::hislip0', True, False, "Simulate=True")
            print("the power supply " + self._query_command('*IDN?') + " was connected at " + str(datetime.now()))
        else:
            self.instr = Rs.RsInstrument(f'{IP}', True, False, "Simulate=False")
            print("the power supply " + self._query_command('*IDN?') + " was connected at " + str(datetime.now()))

    @setting(1)
    def close_connection(self, c) -> None:
        """
        Close connection with the device. Ensure to call this when you're done using the device.

        :return None:
        """
        print("the power supply " + self._query_command('*IDN?') + " was disconnected at " + str(datetime.now()))
        
        # Close connection to device
        self.open = False
        self.instr.close()
        
    @setting(2, which="?", voltage="v")
    def set_voltage(self, c, which : Union[int, str], voltage : float) -> None:
        """
        Set voltage of a specific channel. 
        
        :param int | str which: The channel to activate, either the channel number or axis name.
        :param float voltage: The amount of voltage to set, in volts.
        :return None:
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self._write_command(f"INST OUT{which}")
        self._write_command(f"VOLT {voltage}")
        
    @setting(3, which="?", returns="v")
    def get_voltage(self, c, which : Union[int, str]) -> None:
        """
        Get voltage of a specific channel. 
        
        Note: This is not the voltage being currently pushed into the circuit, this is just the value which the voltage is set at. If you want a measurement, use RS_NGC103.measure_voltage().

        :param int | str which: The channel to read, either the channel number or axis name.
        :return voltage: The voltage active on the selected channel
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self._write_command(f"INST OUT{which}")

        str_out = self.instr.query("VOLT?")
        return float(str_out)
        
    @setting(4, which="?", returns="v")
    def measure_voltage(self, c, which : Union[int, str]) -> None:
        """
        Measure the voltage currently being applied on the circuit
        
        Note: This is the voltage being currently pushed into the circuit. If you want this channel's set value, use RS_NGC103.get_voltage().

        :param int | str which: The channel to read, either the channel number or axis name.
        :return voltage: The voltage active on the selected channel
        """

        if isinstance(which, str):
            which = self.direction_channels[which]

        self._write_command(f"INST OUT{which}")

        str_out = self.instr.query("MEAS:VOLT?")
        return float(str_out)
        
    @setting(5, which="?", current="v")
    def set_current(self, c, which : Union[int, str], current : float) -> None:
        """
        Set current of a specific channel. 
        
        :param int | str which: The channel to activate, either the channel number or axis name.
        :param float current: The amount of current to set, in amps.
        :return None:
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self._write_command(f"INST OUT{which}")
        self._write_command(f"CURR {current}")
        
    @setting(6, which="?", returns="v")
    def get_current(self, c, which : Union[int, str]) -> float:
        """
        Get current of a specific channel. Note: This is not the current being currently pushed into the circuit, this is just the value which the current is set at. If you want a measurement, use RS_NGC103.measure_current().

        :param int | str which: The channel to read, either the channel number or axis name.
        :return current: The current active on the selected channel
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self._write_command(f"INST OUT{which}")

        str_out = self.instr.query("CURR?")
        return float(str_out)
    
    @setting(7, which="?", returns="v")
    def measure_current(self, c, which : Union[int, str]) -> None:
        """
        Measure the current currently being applied on the circuit
        
        Note: This is the current being currently pushed into the circuit. If you want this channel's set value, use RS_NGC103.get_current().

        :param int | str which: The channel to read, either the channel number or axis name.
        :return current: The current active on the selected channel
        """

        if isinstance(which, str):
            which = self.direction_channels[which]

        self._write_command(f"INST OUT{which}")

        str_out = self.instr.query("MEAS:CURR?")
        return float(str_out)

    @setting(8, which="?")
    def activateChannel(self, c, which : Union[int, str]) -> None:
        """
        Activate a specific channel.

        :param int | str which: The channel to activate, either the channel number or axis name. 
        :return None:
        """
        if isinstance(which, str):
            which = self.direction_channels[which]

        self._write_command(f"INST OUT{which}")
        self._write_command("OUTP:CHAN ON")
    
    @setting(9, which="?")
    def deactivateChannel(self, c, which : Union[int, str]) -> None:
        """
        Deactivate a specific channel.

        :param int | str which: The channel to activate, either the channel number or axis name. 
        :return None:
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        if self.maintaining[which]:
            self.release_current(which)
            time.sleep(1)

        self._write_command(f"INST OUT{which}")
        self._write_command("OUTP:CHAN OFF")
    
    @setting(10)
    def activateAll(self, c) -> None:
        """
        Activates all channels.

        :return None:
        """
        for n in self.direction_channels.values():
            self.activateChannel(n)
    
    @setting(11)
    def deactivateAll(self, c) -> None:
        """
        Deactivates all channels.
        
        :return None:
        """
        for n in self.direction_channels.values():
            self.deactivateChannel(n)

    @setting(12)
    def activateMaster(self, c) -> None:
        """
        Activates master control of the device..
        
        :return None:
        """
        self._write_command("OUTP:MAST ON")

    @setting(13)
    def deactivateMaster(self, c) -> None:
        """
        Deactivates master control of the device.
        
        :return None:
        """
        self._write_command("OUTP:MAST OFF")

__server__ = PowerSupplyRnsNgc103()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)

    # rm = visa.ResourceManager()
    # addr = "TCPIP::192.168.0.130::INSTR"
    # pwr_sup = rm.open_resource(addr)

    PowerSupplyRnsNgc103.initServer(__server__, startOpen=True)
