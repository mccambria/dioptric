# -*- coding: utf-8 -*-
"""
Interface for TTL pulse time taggers

Created on August 29th, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod
from servers.inputs.interfaces.counter import Counter
import logging
import numpy as np

class Tagger(Counter):
    
    @abstractmethod
    def start_tag_stream(self, c, apd_indices, gate_indices=None, clock=True):
        """
        Start a tag stream 

        Parameters
        ----------
        
        apd_indices : list
            Indicates the channels for which apds we are using
            
        gate_indices : list, optional
            Indicates the channels for the gates corresponding to the apds
            
        clock : boolean, optional
            Indicates if using a clock with the tagger
            
        These inputs are necessary for the swabian time taggers. 
        The OPX just needs the apd_indices to know which apds to play measure() statements on, but that can live in the config and be pulled from there in the sequence

        """
        
        pass
    
    @abstractmethod
    def stop_tag_stream(self, c):
        """
        Stop a tag stream
        """
        pass
    
    @abstractmethod
    def read_tag_stream(self, c, num_to_read=None):
        """
        Read the tag stream. Returns the tags and channels for all
        the counts that have occurred since the buffer was cleared. 

        Parameters
        ----------
        num_to_read : int, optional
            Tells the hardware how many samples we want to read
            
        Returns
        ----------
        list
            Time tags in ps
        list
            Channel names
        """
        pass
    
    
    