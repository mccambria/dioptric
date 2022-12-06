# -*- coding: utf-8 -*-
"""
Interface for signal generators

Created on August 29th, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod


class SigGen(ABC):
    
    @abstractmethod
    def uwave_on(self, c):
        """
        Turn on the signal. This is like opening an internal gate on
        the signal generator.
        """
        pass

    @abstractmethod
    def uwave_off(self, c):
        """
        Turn off the signal. This is like closing an internal gate on
        the signal generator.
        """
        pass

    @abstractmethod
    def set_freq(self, c, freq):
        """
        Set the frequency of the signal

        Params
            freq: float
                The frequency of the signal in GHz
        """
        pass

    @abstractmethod
    def set_amp(self, c, amp):
        """
        Set the amplitude of the signal

        Params
            amp: float
                The amplitude of the signal in dBm
        """
        pass


    @abstractmethod
    def mod_off(self, c):
        """Turn off the analog modulation."""
        pass

    @abstractmethod
    def load_iq(self, c):
        """
        Set up IQ modulation controlled via the external IQ ports
        """
        pass


    @abstractmethod
    def reset(self, c):
        """
        Make sure the device is in a neutral state for the next experiment
        """
        
        pass

