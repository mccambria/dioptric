#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:42:36 2024

@author: sean, quinn
"""
import numpy as np
import RsInstrument as Rs

from datetime import datetime
from typing import Union

class RS_NGC103:
    def __init__(self, IP : str = None, direction_channels : dict[str, int] = {"x": 1, "y": 2, "z": 3}, start_open=False) -> None:
        self.IP = IP
        self.direction_channels = direction_channels

        self.instr = None

        self.open = False
        if start_open:
            self.open_connection(IP)

    def open_connection(self, IP : str = None) -> None:
        """
        Open connection with the device. Ensure to call this before attempting to use the device.

        :return None:
        """
        self.open = True
        Rs.RsInstrument.assert_minimum_version('1.50.0')
    
        if IP == None:
            self.instr = Rs.RsInstrument('TCPIP::192.168.56.101::hislip0', True, False, "Simulate=True")
            print("the power supply " + self.instr.query_str('*IDN?') + " was connected at " + str(datetime.now()))
        else:
            self.instr = Rs.RsInstrument(f'TCPIP::{IP}::INSTR', True, False, "Simulate=False")
            print("the power supply " + self.instr.query_str('*IDN?') + " was connected at " + str(datetime.now()))
        
    def close_connection(self) -> None:
        """
        Close connection with the device. Ensure to call this when you're done using the device.

        :return None:
        """
        print("the power supply " + self.instr.query_str('*IDN?') + " was disconnected at " + str(datetime.now()))
        self.open = False
        self.instr.close()
    
    def current_for_field(self, x : float, y : float, z : float) -> list[float]:
        """
        coordinate conventions (cartesian)
        x: out of the stage (controlled by PCB)
        y: "side to side" (controlled by small coils)
        z: out of the table "up and down" (controlled by big coils)
        
        first task: define what the measured field is given a 1 A current in each direction
        in an ideal world:
        x = [20 , 0  , 0 ]
        y = [0  , 20 , 0 ]
        z = [0  , 0  , 20]
        but this is not an ideal world. The diamond will not be centered, the coils and PCB produce stray fields, etc.
        
        second task: 89 and 54
        the system should behave linearly i.e. the field produced is a linear function of the input current
        we can thus use the machinery of linear algebra
        
        we can line up the vectors x,y,z as defined and shove them into a matrix 
        this matrix converts the input current to the magnetic field
        B = MI
        where M = [x y z]
        
        third task: we want to know the current we need given a desired field so the output is the matrix inverse
        I = M^-1 B
        """
        x = [19 , 3  , 2 ]
        y = [1  , 19 , 2 ]
        z = [0  , 1  , 19]
        
        M = [x, y, z] 
        M = np.transpose(M)
        
        M = np.linalg.inv(M)
        np.array()
        current = M.dot([x, y, z])

        return current
    
    def set_current(self, which : Union[int, str], current : float) -> None:
        """
        Set current of a specific channel. 
        
        :param int | str which: The channel to activate, either the channel number or axis name.
        :param float current: The amount of current to set, in amps.
        :return None:
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self.instr.write_str(f"INST OUT{which}")
        self.instr.write_str(f"CURR {current}")
    
    def get_current(self, which : Union[int, str]) -> float:
        """
        Get current of a specific channel.

        :param int | str which: The channel to read, either the channel number or axis name.
        :return current: The current active on the selected channel
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self.instr.write_str(f"INST OUT{which}")

        str_out = self.instr.query("CURR?")
        return float(str_out)
    
    def activateChannel(self, which : Union[int, str]) -> None:
        """
        Activate a specific channel.

        :param int | str which: The channel to activate, either the channel number or axis name. 
        :return None:
        """
        if isinstance(which, str):
            which = self.direction_channels[which]

        self.instr.write_str(f"INST OUT{which}")
        self.instr.write_str("OUTP:CHAN ON")
    
    def deactivateChannel(self, which : Union[int, str]) -> None:
        """
        Deactivate a specific channel.

        :param int | str which: The channel to activate, either the channel number or axis name. 
        :return None:
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self.instr.write_str(f"INST OUT{which}")
        self.instr.write_str("OUTP:CHAN OFF")
    
    def activateAll(self) -> None:
        """
        Activates all channels.

        :return None:
        """
        for n in self.direction_channels.values():
            self.activateChannel(n)
    
    def deactivateAll(self) -> None:
        """
        Deactivates all channels.
        
        :return None:
        """
        for n in self.direction_channels.values():
            self.deactivateChannel(n)

    def activateMaster(self) -> None:
        """
        Activates master control of the device..
        
        :return None:
        """
        self.instr.write_str("OUTP ON")

    def deactivateMaster(self) -> None:
        """
        Deactivates master control of the device.
        
        :return None:
        """
        self.instr.write_str("OUTP OFF")
