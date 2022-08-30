# -*- coding: utf-8 -*-
"""
Interface for TTL pulse counters

Created on August 29th, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod


class Counter(ABC):
    @abstractmethod
    def read_counter_simple(self, c, num_to_read=None):
        """Return the number of TTL pulses that during each sampling period

        Parameters
        ----------
        num_to_read : int, optional
            Polls the hardware until we get num_to_read samples. If
            None, just poll once and return
        
        Returns
        ----------
        list
            list of counts for each sample
        """
        pass
