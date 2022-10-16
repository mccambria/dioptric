# -*- coding: utf-8 -*-
"""
Interface for TTL pulse counters

Created on August 29th, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod
import logging
from labrad.server import LabradServer
from labrad.server import setting
import numpy as np

class Counter(LabradServer, ABC):
    
    
    # def read_counter_setting_internal(self, num_to_read):
    #     if self.stream is None:
    #         logging.error("read_counter attempted while stream is None.")
    #         return
    #     if num_to_read is None:
    #         # Poll once and return the result
    #         counts = self.read_counter_internal()
    #     else:
    #         # Poll until we've read the requested number of samples
    #         counts = []
    #         while len(counts) < num_to_read:
    #             counts.extend(self.read_counter_internal())
                
    #         if len(counts) > num_to_read:
    #             msg = "Read {} samples, only requested {}".format(
    #                 len(counts), num_to_read
    #             )
    #             logging.error(msg)

    #     return counts
    
    
    @setting(7, num_to_read="i", returns="*3w")
    def read_counter_complete(self, c, num_to_read=None):
        return self.read_counter_setting_internal(num_to_read)
    
    
    @setting(8, num_to_read="i", returns="*w")
    def read_counter_simple(self, c, num_to_read=None):
        
        complete_counts = self.read_counter_setting_internal(num_to_read)

        # To combine APDs we assume all the APDs have the same gate
        # gate_channels = list(self.tagger_di_gate.values())
        # first_gate_channel = gate_channels[0]
        # if not all(val == first_gate_channel for val in gate_channels):
        #     logging.critical("Combined counts from APDs with different gates.")

        # Just find the sum of each sample in complete_counts
        return_counts = [
            np.sum(sample, dtype=int) for sample in complete_counts ]

        return return_counts

    @setting(9, num_to_read="i", returns="*2w")
    def read_counter_separate_gates(self, c, num_to_read=None):

        complete_counts = self.read_counter_setting_internal(num_to_read)
        # logging.info(complete_counts)

        # To combine APDs we assume all the APDs have the same gate
        # gate_channels = list(self.tagger_di_gate.values())
        # first_gate_channel = gate_channels[0]
        # if not all(val == first_gate_channel for val in gate_channels):
        #     logging.critical("Combined counts from APDs with different gates.")

        # Add the APD counts as vectors for each sample in complete_counts
        return_counts = [
            np.sum(sample, 0, dtype=int).tolist() for sample in complete_counts ]

        return return_counts
    
    @setting(10, modulus="i", num_to_read="i", returns="*2w")
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
    
    @setting(11, num_to_read="i", returns="*2w")
    def read_counter_separate_apds(self, c, num_to_read=None):

        complete_counts = self.read_counter_setting_internal(num_to_read)

        # Just find the sum of the counts for each APD for each
        # sample in complete_counts
        return_counts = [
            [np.sum(apd_counts, dtype=int) for apd_counts in sample]
            for sample in complete_counts
        ]

        return return_counts
    
    
    @abstractmethod
    def reset(self, c):
        """
        Reset the tagger
        """
        pass
    
    # @abstractmethod
    # def get_channel_mapping(self, c):
    #     """
    #     do we need this???
    #     """
    #     pass


    @abstractmethod
    def clear_buffer(self, c):
        """
        Clear the buffer of the time tagger if necessary
        """
        pass
    
    # @abstractmethod
    # def read_counter_modulo_gates(self, c, modulus, num_to_read=None):
    #     """
    #     Read the counts in the form of the three level list structure then sum the counts from different apds, keeping different gates distinct.
    #     Then assemble the data structure according to the modulus. If the modulus is the same as the number of gates, then it is the same
    #     as read_counter_separate_gates. If modulus is 1, it is the same as read_counter_simple. It essentially grabs up to the gate of interest
    #     The output is therefore a 2 level list, where the first level is samples and the second level is gates. 

    #     Parameters
    #     ----------
    
    #     num_to_read : int, optional
    #         Tells the hardware how many samples we want to read
            
    #     modulus : int
    #         The number of gates of interest to include in the returned data structure of counts

    #     Returns
    #     -------
    #     return_counts : list
    #         Two level list. First level is samples, second is gates.
    #     """
    #     pass


    # @abstractmethod
    # def read_counter_separate_apds(self, c, num_to_read=None):
    #     """
    #     Read the counts in the form of the three level list structure then sum the gates for each apd. 
    #     The output is therefore a two level list. The first level is samples, the second is apds. 

    #     Parameters
    #     ----------
    
    #     num_to_read : int, optional
    #         Tells the hardware how many samples we want to read
            
    #     Returns
    #     -------
    #     return_counts : list
    #         Two level list. First level is samples, second is apd.
    #     """
    #     pass
    
    # @abstractmethod
    # def read_counter_complete(self, c, num_to_read=None):
    #     """
    #     Read the counts and return them in the form of the three level list described below

    #     Parameters
    #     ----------
        
    #     num_to_read : int, optional
    #         Tells the hardware how many samples we want to read

    #     Returns 
    #     -------
        
    #     list 
    #         Three level list. First level is samples), second is apds, third is gates, as shown below for 2 samples (s), 2 apds (a), 3 gates (g)
    #         [  [ [s1_a1_g1, s1_a1_g2, s1_a1_g3] , [s1_a2_g1, s1_a2_g2, s1_a2_g3] ],  [ [s2_a1_g1, s2_a1_g2, s2_a1_g3] , [s2_a2_g1, s2_a2_g2, s2_a2_g3] ]  ]

    #     """
    #     pass
    
    # @abstractmethod
    # def read_counter_simple(self, c, num_to_read=None):
    #     """
    #     Read the counts in the form of the three level list structure then sum the counts from different apds and different gates within each sample
    #     The output is therefore a one level list of counts for each sample

    #     Parameters
    #     ----------
    
    #     num_to_read : int, optional
    #         Tells the hardware how many samples we want to read

    #     Returns
    #     -------
    #     return_counts : list
    #         List of counts for each sample

    #     """
    #     pass

    # @abstractmethod
    # def read_counter_separate_gates(self, c, num_to_read=None):
    #     """
    #     Read the counts in the form of the three level list structure then sum the counts from different apds, keeping different gates distinct.
    #     The output is therefore a 2 level list, where the first level is samples and the second level is gates. 

    #     Parameters
    #     ----------
    
    #     num_to_read : int, optional
    #         Tells the hardware how many samples we want to read

    #     Returns
    #     -------
    #     return_counts : list
    #         Two level list. First level is samples, second is gates.

    #     """
        
    #     pass
