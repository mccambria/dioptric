# -*- coding: utf-8 -*-
"""
Interface for TTL pulse generators

Created on August 29th, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod
from labrad.server import LabradServer
from labrad.server import setting


class PulseGen(LabradServer,ABC):
    
    
    @setting(1, seq_file="s", num_repeat="i", seq_args_string="s", returns="*?")
    def stream_immediate(self, c, seq_file, num_repeat=1, seq_args_string=""): # from pulse streamer. For the OPX this will get the full sequence we want and run it. DONE
        """Load the sequence from seq_file and immediately run it for
        the specified number of repitions. End in the specified
        final output state. return the desired ret vals. It's the same func for every pulse generatng device so it lives on the interface

        Params
            seq_file: str
                A sequence file from the sequence library
            num_repeat: int
                Number of times to repeat the sequence. Default is 1
            args: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None. All values in list must have same type.

        Returns
            list(any)
                Arbitrary list returned by the sequence file
        """
        
        ret_vals = self.stream_load(c, seq_file, seq_args_string)
        self.stream_start(c, num_repeat)
        
        return ret_vals

    @abstractmethod
    def stream_load(self, c, seq_file, seq_args_string=""):
        """
        Load the sequence from seq_file. End in the specified final output state.
        The sequence will not run until you call stream_start.

        Params
            seq_file: str
                A sequence file from the sequence library
            args: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None

        Returns
            list(any)
                Arbitrary list returned by the sequence file
        """
        pass

    @abstractmethod
    def stream_start(self, c, num_repeat=1):
        """
        Run the currently loaded stream for the specified number of repitions.

        Params
            num_repeat: int
                Number of times to repeat the sequence. Default is 1
        """
        pass
    
    # @abstractmethod
    # def constant(self, c, digital_channels=[], analog_0_voltage=0.0, analog_1_voltage=0.0):
    #     """
    #     Set the outputs to constant values; default are everything off

    #     Parameters
    #     ----------
    #     digital_channels : list(int), optional
    #         Digital channels to set high
    #     analog_channels : list(int), optional
    #         Analog channels to set nonzero
    #     analog_voltages : list(float), optional
    #         Voltages to assign to analog_channels
    #     """
    #     pass
    
    # @abstractmethod
    # def force_final(self, c):
    #     """
    #     don't think we need this'
    #     """
    #     pass

    @abstractmethod
    def reset(self, c):
        """
        Reset the pulse generating device so that it doesn't have any sequences loaded or running
        """
        pass
