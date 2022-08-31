# -*- coding: utf-8 -*-
"""
server for the Quantum Machines OPX

Created on August 29th, 2022

@author: carter fox

### BEGIN NODE INFO
[info]
name = opx
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

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *

from labrad.server import LabradServer
from labrad.server import setting
from twisted.internet.defer import ensureDeferred
from numpy import count_nonzero, nonzero, concatenate
import numpy as np
import logging
import re
import time
import socket
from inputs.interfaces.tagger import Tagger
from timing.interfaces.pulse_gen import PulseGen


class OPX(LabradServer, Tagger, PulseGen):
    name = "opx"
    pc_name = socket.gethostname()

    def initServer(self):
        pass
    
    
    ### probably need some on_config and get_config stuff...
   
    #%% sequence loading and executing functions ###
        
    def get_seq(self, seq_file, seq_args_string): # from pulse streamer. DONE
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tool_belt.decode_seq_args(seq_args_string)
            
            seq, final, ret_vals = seq_module.get_seq(        # here seq is the qua program. refer to opx sequence template      
                self, self.config_dict, args
            )
            
        return seq, final, ret_vals
    
    @setting(1, seq_file="s", seq_args_string="s", returns="*?")
    def stream_load(self, c, seq_file, seq_args_string=""): #from pulse streamer. for the opx all this function does is return back some property(s) of the sequence you want
        """For the OPX, this will just initiate the communication with the QOP
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
        seq, final, ret_vals = self.get_seq(seq_file, seq_args_string)
        
        return ret_vals

    @setting(2, num_repeat="i")
    def stream_start(self, c, num_repeat=1): # from pulse streamer. The OPX doesn't use this because it needs the program to be inputted, so it only uses stream_immediate(). 
                                             # If someone tries to use this, it will spit out an error telling them to change stream_start() to stream_immediate()
        print("Error: Must use stream_immediate() instead. You are using the OPX, which requires that. Update the routine so it uses stream_immediate(), which should have no effect on the routine.")
        return
    
    @setting(0, seq_file="s", num_repeat="i", seq_args_string="s", returns="*?")
    def stream_immediate(self, c, seq_file, num_repeat=1, seq_args_string=""): # from pulse streamer. For the OPX this will get the sequence we want and run it. 
        """Load the sequence from seq_file and immediately run it for
        the specified number of repitions. End in the specified
        final output state.

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
        
        qua_seq, final, ret_vals = self.get_seq(self, seq_file, seq_args_string)
        
        qmm = QuantumMachinesManager(qop_ip) #open communication with the QOP
        qm = qmm.open_qm(config)
        # execute QUA program
        job = qm.execute(qua_seq)
        
        return job ## In the pulse streamer server, this function just returns the ret vals but its not even used usually. We will want something like read_counter_simple to have the job as an input. 
    
    
    #%% counting and time tagging functions ### 
        
    
    # @jit(nopython=True)
    @setting(5, num_to_read="i", returns="*w")
    def read_counter_simple(self, c, num_to_read=None): #from apd tagger

        return return_counts

    # @jit(nopython=True)
    @setting(6, num_to_read="i", returns="*2w")
    def read_counter_separate_gates(self, c, num_to_read=None): #from apd tagger

        return return_counts

    # @jit(nopython=True)
    @setting(11, modulus="i", num_to_read="i", returns="*2w")
    def read_counter_modulo_gates(self, c, modulus, num_to_read=None): #from apd tagger

        return return_counts
    
    @setting(3, num_to_read="i", returns="*s*i")
    def read_tag_stream(self, c, num_to_read=None): #from apd tagger 
        
        return timestamps, channels


    @setting(9)
    def clear_buffer(self, c): # from apd tagger
        """Clear the hardware's internal buffer. Should be called before
        starting a pulse sequence."""
        
        return


    #%%


__server__ = OPX()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
