# -*- coding: utf-8 -*-
"""
Interface for TTL pulse counters

Created on August 29th, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod
import logging

class Counter(ABC):
    
    
    def read_counter_setting_internal(self, num_to_read):
        if self.stream is None:
            logging.error("read_counter attempted while stream is None.")
            return
        if num_to_read is None:
            # Poll once and return the result
            counts = self.read_counter_internal()
        else:
            # Poll until we've read the requested number of samples
            counts = []
            while len(counts) < num_to_read:
                counts.extend(self.read_counter_internal())
            if len(counts) > num_to_read:
                msg = "Read {} samples, only requested {}".format(
                    len(counts), num_to_read
                )
                logging.error(msg)

        return counts
    
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
    
    @abstractmethod
    def read_counter_modulo_gates(self, c, modulus, num_to_read=None):
        """
        Read the counts in the form of the three level list structure then sum the counts from different apds, keeping different gates distinct.
        Then assemble the data structure according to the modulus. If the modulus is the same as the number of gates, then it is the same
        as read_counter_separate_gates. If modulus is 1, it is the same as read_counter_simple. It essentially grabs up to the gate of interest
        The output is therefore a 2 level list, where the first level is samples and the second level is gates. 

        Parameters
        ----------
    
        num_to_read : int, optional
            Tells the hardware how many samples we want to read
            
        modulus : int
            The number of gates of interest to include in the returned data structure of counts

        Returns
        -------
        return_counts : list
            Two level list. First level is samples, second is gates.
        """
        pass


    @abstractmethod
    def read_counter_separate_apds(self, c, num_to_read=None):
        """
        Read the counts in the form of the three level list structure then sum the gates for each apd. 
        The output is therefore a two level list. The first level is samples, the second is apds. 

        Parameters
        ----------
    
        num_to_read : int, optional
            Tells the hardware how many samples we want to read
            
        Returns
        -------
        return_counts : list
            Two level list. First level is samples, second is apd.
        """
        pass
    
    @abstractmethod
    def read_counter_complete(self, c, num_to_read=None):
        """
        Read the counts and return them in the form of the three level list described below

        Parameters
        ----------
        
        num_to_read : int, optional
            Tells the hardware how many samples we want to read

        Returns 
        -------
        
        list 
            Three level list. First level is samples), second is apds, third is gates, as shown below for 2 samples (s), 2 apds (a), 3 gates (g)
            [  [ [s1_a1_g1, s1_a1_g2, s1_a1_g3] , [s1_a2_g1, s1_a2_g2, s1_a2_g3] ],  [ [s2_a1_g1, s2_a1_g2, s2_a1_g3] , [s2_a2_g1, s2_a2_g2, s2_a2_g3] ]  ]

        """
        pass
    
    @abstractmethod
    def read_counter_simple(self, c, num_to_read=None):
        """
        Read the counts in the form of the three level list structure then sum the counts from different apds and different gates within each sample
        The output is therefore a one level list of counts for each sample

        Parameters
        ----------
    
        num_to_read : int, optional
            Tells the hardware how many samples we want to read

        Returns
        -------
        return_counts : list
            List of counts for each sample

        """
        pass

    @abstractmethod
    def read_counter_separate_gates(self, c, num_to_read=None):
        """
        Read the counts in the form of the three level list structure then sum the counts from different apds, keeping different gates distinct.
        The output is therefore a 2 level list, where the first level is samples and the second level is gates. 

        Parameters
        ----------
    
        num_to_read : int, optional
            Tells the hardware how many samples we want to read

        Returns
        -------
        return_counts : list
            Two level list. First level is samples, second is gates.

        """
        
        pass
