#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:42:36 2024

@author: sean, quinn
"""
from datetime import datetime
import threading
import time
from typing import Union

import numpy as np
import RsInstrument as Rs

class RS_NGC103:
    def __init__(self, IP : str = None, direction_channels : dict[str, int] = {"x": 1, "y": 2, "z": 3}, start_open=False) -> None:
        self.IP = IP
        self.direction_channels = direction_channels

        self.instr = None

        self.open = False
        if start_open:
            self.open_connection(IP)

        self.maintaining = False
        self.maintain_thread = None

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
        
        # Close connection to device
        self.open = False
        self.instr.close()

        # Call the stop function, and then join thread to safely deactivate
        if self.maintaining:
            self.maintaining = False
            self.maintain_thread.join()
        
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
    
    def maintain_current(self, which : Union[int, str], current : float, voltage_start : float = 0.1) -> None:
        """
        Hack to maintain a specific current, despite the CC mode not working on the NGC103, we can still adjust the voltage to keep it at the correct level.

        :param int | str which: The channel to activate, either the channel number or axis name.
        :param float voltage_start: The initial value to be set for volts, defaults to 0.1 
        :param float current: The desired current (amps) which you wish to maintain.
        """

        def maintain_loop(voltage_guess):

            # Update the current accordingly every second
            while True:
                if self.maintaining == False or self.open == False:
                    break

                # R = V / I
                actual_current = self.measure_current(which)
                actual_resistance = voltage_guess / actual_current

                voltage_guess = current * actual_resistance
                
                self.set_voltage(which, voltage_guess)
                
                time.sleep(1)
                #print(f"voltage set to {voltage_guess}, intended current {current}")

        self.maintaining = True
        
        # V = IR
        voltage_guess = voltage_start
        self.set_voltage(which, voltage_guess)

        self.maintain_thread = threading.Thread(name="bg", target=maintain_loop, args=[voltage_guess], daemon=False)
        self.maintain_thread.start()

    def set_voltage(self, which : Union[int, str], voltage : float) -> None:
        """
        Set voltage of a specific channel. 
        
        :param int | str which: The channel to activate, either the channel number or axis name.
        :param float voltage: The amount of voltage to set, in volts.
        :return None:
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self.instr.write_str(f"INST OUT{which}")
        self.instr.write_str(f"VOLT {voltage}")
    
    def get_voltage(self, which : Union[int, str]) -> None:
        """
        Get voltage of a specific channel. 
        
        Note: This is not the voltage being currently pushed into the circuit, this is just the value which the voltage is set at. If you want a measurement, use RS_NGC103.measure_voltage().

        :param int | str which: The channel to read, either the channel number or axis name.
        :return voltage: The voltage active on the selected channel
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self.instr.write_str(f"INST OUT{which}")

        str_out = self.instr.query("VOLT?")
        return float(str_out)

    def measure_voltage(self, which : Union[int, str]) -> None:
        """
        Measure the voltage currently being applied on the circuit
        
        Note: This is the voltage being currently pushed into the circuit. If you want this channel's set value, use RS_NGC103.get_voltage().

        :param int | str which: The channel to read, either the channel number or axis name.
        :return voltage: The voltage active on the selected channel
        """

        if isinstance(which, str):
            which = self.direction_channels[which]

        self.instr.write_str(f"INST OUT{which}")

        str_out = self.instr.query("MEAS:VOLT?")
        return float(str_out)

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
        Get current of a specific channel. Note: This is not the current being currently pushed into the circuit, this is just the value which the current is set at. If you want a measurement, use RS_NGC103.measure_current().

        :param int | str which: The channel to read, either the channel number or axis name.
        :return current: The current active on the selected channel
        """
        if isinstance(which, str):
            which = self.direction_channels[which]
        
        self.instr.write_str(f"INST OUT{which}")

        str_out = self.instr.query("CURR?")
        return float(str_out)
    
    def measure_current(self, which : Union[int, str]) -> None:
        """
        Measure the current currently being applied on the circuit
        
        Note: This is the current being currently pushed into the circuit. If you want this channel's set value, use RS_NGC103.get_current().

        :param int | str which: The channel to read, either the channel number or axis name.
        :return current: The current active on the selected channel
        """

        if isinstance(which, str):
            which = self.direction_channels[which]

        self.instr.write_str(f"INST OUT{which}")

        str_out = self.instr.query("MEAS:CURR?")
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
        self.instr.write_str("OUTP:MAST ON")

    def deactivateMaster(self) -> None:
        """
        Deactivates master control of the device.
        
        :return None:
        """
        self.instr.write_str("OUTP:MAST OFF")
