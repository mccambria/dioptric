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
        """For the OPX, this will grab a single iteration of the desired sequence
            seq_file: str
                A qua sequence file from the sequence library
            seq_args_string: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None
                
        Returns
            seq: Qua program 
                A sequence written in Qua for the OPX
            final: unsure. 
            list(any)
                Arbitrary list returned by the sequence file
        """    
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tool_belt.decode_seq_args(seq_args_string)
            
            seq, final, ret_vals = seq_module.get_seq(        # here seq is the qua program. refer to opx sequence template      
                self, self.config_dict, args
            )
        
        self.qua_program = seq
        self.seq_file = seq_file
        self.seq_args_string = seq_args_string
        
        return seq, final, ret_vals

    def get_full_seq(self, seq_file, seq_args_string, num_repeat): # this one returns the full qua sequence with repeats, such as rabi with nreps=1e5. DONE
        """For the OPX, this will grab the desired sequence with the desired number of repeats
            seq_file: str
                A qua sequence file from the sequence library
            seq_args_string: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None
            num_repeat: int
                The number of times to repeat the sequence, such as the number of reps in a rabi routine
                
        Returns
            seq: Qua program 
                A sequence written in Qua for the OPX
            final: unsure. 
            list(any)
                Arbitrary list returned by the sequence file
        """    
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tool_belt.decode_seq_args(seq_args_string)
            
            seq, final, ret_vals = seq_module.get_full_seq(        # here seq is the qua program. refer to opx sequence template      
                self, self.config_dict, args, num_repeat
            )
        
        self.qua_program = seq
        self.seq_file = seq_file
        self.seq_args_string = seq_args_string
        
        return seq, final, ret_vals
    
    @setting(1, seq_file="s", seq_args_string="s", returns="*?")
    def stream_load(self, c, seq_file, seq_args_string=""): #from pulse streamer. for the opx all this function does is return back some property(s) of the sequence you want. DONE
        """For the OPX, this will just grab a single iteration of the desired sequence and return the parameter of interest
        Params
            seq_file: str
                A qua sequence file from the sequence library
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
        """For the OPX, this will run the qua program already grabbed by stream_load(). It opens communication with the QOP and executes the program, creating the job 
        Params
            num_repeat: int
                The number of times the sequence is repeated, such as the number of reps in a rabi routine
        """
        
        seq, final, ret_vals = self.get_full_seq(self.seq_file, self.seq_args_string, num_repeat) #gets the full sequence
        self.qmm = QuantumMachinesManager(qop_ip) # opens communication with the QOP so we can make a quantum machine
        self.qm = qmm.open_qm(config)             # makes a quantum machine with the specific configuration
        self.job = qm.execute(self.qua_program)   # executes the qua program on the quantum machine             
        return 
    
    @setting(3, seq_file="s", num_repeat="i", seq_args_string="s", returns="*?")
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
        
        seq, final, ret_vals = self.get_full_seq(seq_file, seq_args_string, num_repeat) #gets the full sequence
        
        self.qmm = QuantumMachinesManager(qop_ip) # opens communication with the QOP so we can make a quantum machine
        self.qm = qmm.open_qm(config)             # makes a quantum machine with the specific configuration
        self.job = qm.execute(self.qua_program)   # executes the qua program on the quantum machine        
        return ret_vals
    
    #%% counting and time tagging functions ### 
        
    @setting(47, num_to_read="i", returns="*3w")
    def read_counter_complete(self, c, num_to_read=None):
        return self.read_counter_setting_internal(num_to_read)
    
    def read_counter_setting_internal(self, c, num_to_read=None): #from apd tagger. for the opx it fetches the results from the job. Don't think num_to_read has to do anything
        """This is the core function that any tagger we have needs. 
        For the OPX this fetches the data from the job that was created when the program was executed. 
        Assumes "counts_st" is one of the data streams
        The count stream should be a three level list. First level is the sample, second is the gates, third is the different apds. 
        first index gives the sample. next level gives the gate. next level gives which apd
        [  [ [],[] ] , [ [],[] ], [ [],[] ]  ]
        ##### This may be slightly wrong. It may be apds then gate, in which I need to slightly change the sequence code
        
        Params
            num_to_read: int
                This is not needed for the OPX
        Returns
            return_counts: array
                This is an array of the counts 
        """
        results = fetching_tool(self.job, data_list=["counts_st"], mode="wait_for_all")
        
        while results.is_processing():
            # Fetch results
            return_counts = results.fetch_all() #just not sure if its gonna put it into the list structure we want
        
        return return_counts

    @setting(5, num_to_read="i", returns="*w")
    def read_counter_simple(self, c, num_to_read=None):

        complete_counts = self.read_counter_setting_internal(num_to_read)

        # To combine APDs we assume all the APDs have the same gate
        gate_channels = list(self.tagger_di_gate.values())
        first_gate_channel = gate_channels[0]
        if not all(val == first_gate_channel for val in gate_channels):
            logging.critical("Combined counts from APDs with different gates.")

        # Just find the sum of each sample in complete_counts
        return_counts = [
            np.sum(sample, dtype=int) for sample in complete_counts ]

        return return_counts

    @setting(6, num_to_read="i", returns="*2w")
    def read_counter_separate_gates(self, c, num_to_read=None):

        complete_counts = self.read_counter_setting_internal(num_to_read)
        # logging.info(complete_counts)

        # To combine APDs we assume all the APDs have the same gate
        gate_channels = list(self.tagger_di_gate.values())
        first_gate_channel = gate_channels[0]
        if not all(val == first_gate_channel for val in gate_channels):
            logging.critical("Combined counts from APDs with different gates.")

        # Add the APD counts as vectors for each sample in complete_counts
        return_counts = [
            np.sum(sample, 0, dtype=int).tolist() for sample in complete_counts ]

        return return_counts
    
    @setting(11, modulus="i", num_to_read="i", returns="*2w")
    def read_counter_modulo_gates(self, c, modulus, num_to_read=None):

        complete_counts = self.read_counter_setting_internal(num_to_read)
        # logging.info(complete_counts)

        # To combine APDs we assume all the APDs have the same gate
        gate_channels = list(self.tagger_di_gate.values())
        first_gate_channel = gate_channels[0]
        if not all(val == first_gate_channel for val in gate_channels):
            logging.critical("Combined counts from APDs with different gates.")

        # Add the APD counts as vectors for each sample in complete_counts
        # sum_lambda = lambda arg: np.sum(arg, 0, dtype=int).tolist()
        # with Pool() as p:
        #     separate_gate_counts = p.map(sum_lambda, complete_counts)
        separate_gate_counts = [np.sum(el, 0, dtype=int).tolist() for el in complete_counts]

        # Run the modulus
        return_counts = []
        for sample in separate_gate_counts:
            sample_list = []
            for ind in range(modulus):
                sample_list.append(np.sum(sample[ind::modulus]))
            return_counts.append(sample_list)

        return return_counts

    
    @setting(7, num_to_read="i", returns="*s*i")
    def read_tag_stream(self, c, num_to_read=None): #from apd tagger 
    
        results = fetching_tool(self.job, data_list=["times_st", "apd_indices"], mode="wait_for_all")
        
        channels_list = []
        
        while results.is_processing():
            # Fetch results
            times, apds = results.fetch_all()
            times_list = []
            for i in range(len(times)):
                times_list = times_list + times[i]
                channels_list = channels_list + np.full(len(times[i]),apd_indices[i]).tolist()
            times_array = np.array(times_list)
            channels_array = np.array(channels_list)
            ind_order = np.argsort(times_array)
            time_tags = times_array[ind_order]
            channels = channels_array[ind_order]
            time_tags = time_tags.tolist()
            channels = channels.tolist()
            
        return time_tags, channels 

    
    @setting(8, apd_indices="*i", gate_indices="*i", clock="b") # from apd tagger. 
    def start_tag_stream(self, c, apd_indices, gate_indices=None, clock=True):
        """opx doesn't need it"""
        pass
    
    @setting(9, apd_indices="*i", gate_indices="*i", clock="b") # from apd tagger. 
    def stop_tag_stream(self, c, apd_indices, gate_indices=None, clock=True):
        """OPX doesn't need it"""
        pass
    
    @setting(10)
    def clear_buffer(self, c): # from apd tagger
        """OPX doesn't need it"""
        pass

    #%% functions specific to the OPX
    
    def close_qm_machines_and_manager(self, c):
        self.qmm.close_all_quantum_machines()
        self.qmm.close()
    
    
    
    #%%


__server__ = OPX()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
