# -*- coding: utf-8 -*-
"""
Interface for TTL pulse time taggers

Created on August 29th, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod
from servers.inputs.interfaces.counter import Counter


class Tagger(Counter):
    
    @abstractmethod
    def read_tag_stream(self, c, num_to_read=None):
        """
        Read the tag stream. Returns the tags and channels for all
        the counts that have occurred since the buffer was cleared. 

        Parameters
        ----------
        num_to_read : int, optional
            Polls the hardware until we get num_to_read samples. If
            None, just poll once and return
            
        Returns
        ----------
        list
            Time tags in ps
        list
            Channel names
        """
        pass
