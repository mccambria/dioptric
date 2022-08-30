# -*- coding: utf-8 -*-
"""
Interface for TTL pulse generators

Created on August 29th, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod


class PulseGen(ABC):
    
    @abstractmethod
    def constant(self, c, digital_channels=[], analog_channels=[], analog_voltages=[]):
        """
        Set the outputs to constant values; defaulta are everything off

        Parameters
        ----------
        digital_channels : list(int), optional
            Digital channels to set high
        analog_channels : list(int), optional
            Analog channels to set nonzero
        analog_voltages : list(float), optional
            Voltages to assign to analog_channels
        """
        pass
