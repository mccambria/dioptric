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
        
    def get_seq(self, seq_file, seq_args_string): # this returns the qua sequence without repeats. from pulse streamer. DONE
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tool_belt.decode_seq_args(seq_args_string)
            
            seq, final, ret_vals = seq_module.get_seq(        # here seq is the qua program. refer to opx sequence template      
                self, self.config_dict, args
            )
        
        self.qua_program = seq
        
        return seq, final, ret_vals

    def get_full_seq(self, seq_file, seq_args_string, num_repeat): # this one returns the full qua sequence with repeats, such as rabi with nreps=1e5. DONE
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tool_belt.decode_seq_args(seq_args_string)
            
            seq, final, ret_vals = seq_module.get_full_seq(        # here seq is the qua program. refer to opx sequence template      
                self, self.config_dict, args, num_repeat
            )
        
        self.qua_program = seq
        
        return seq, final, ret_vals
    
    @setting(1, seq_file="s", seq_args_string="s", returns="*?")
    def stream_load(self, c, seq_file, seq_args_string=""): #from pulse streamer. for the opx all this function does is return back some property(s) of the sequence you want. DONE
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
    def stream_start(self, c, num_repeat=1): # from pulse streamer. It will start the full sequence. DONE
        seq, final, ret_vals = self.get_full_seq(seq_file, seq_args_string, num_repeat)
        self.qmm = QuantumMachinesManager(qop_ip) #open communication with the QOP
        self.qm = qmm.open_qm(config)          
        # execute QUA program
        self.job = qm.execute(self.qua_program)#maybe I handle num_repeats by having the program have an input for number of iterations and do self.qua_program(num_repeats)                
        return 
    
    @setting(0, seq_file="s", num_repeat="i", seq_args_string="s", returns="*?")
    def stream_immediate(self, c, seq_file, num_repeat=1, seq_args_string=""): # from pulse streamer. For the OPX this will get the full sequence we want and run it. DONE
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
        
        seq, final, ret_vals = self.get_full_seq(seq_file, seq_args_string, num_repeat)
        
        self.qmm = QuantumMachinesManager(qop_ip) #open communication with the QOP
        self.qm = qmm.open_qm(config)          
        # execute QUA program
        self.job = qm.execute(self.qua_program)     
        return ret_vals
    
    #%% counting and time tagging functions ### 
        
    
    # @jit(nopython=True)
    @setting(5, num_to_read="i", returns="*w")
    def read_counter_simple(self, c, num_to_read=None): #from apd tagger. for the opx it fetches the results from the job. Don't think num_to_read has to do anything
    
        results = fetching_tool(self.job, data_list=["sig_counts"], mode="wait_for_all")
        
        while results.is_processing():
            # Fetch results
            return_counts = results.fetch_all()
        
        return return_counts

    # @jit(nopython=True)
    @setting(6, num_to_read="i", returns="*2w")
    def read_counter_separate_gates(self, c, num_to_read=None): #from apd tagger
    
        results = fetching_tool(self.job, data_list=["sig_counts","ref_counts"], mode="wait_for_all")
        
        while results.is_processing():
            # Fetch results
            sig_counts, ref_counts = results.fetch_all()
            return_counts = [sig_counts, ref_counts]
        
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
