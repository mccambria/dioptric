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
from labrad.server import LabradServer
from labrad.server import setting

class Tagger(Counter, ABC):
    
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
    

    @setting(16, num_to_read="i", returns="*s*i")
    def read_tag_stream(self, c, num_to_read=None):
        """Read the stream started with start_tag_stream. Returns two lists,
        each as long as the number of counts that have occurred since the
        buffer was refreshed. First list is timestamps in ps, second is
        channel names
        """
        if self.stream is None:
            logging.error("read_tag_stream attempted while stream is None.")
            return
        if num_to_read is None:
            timestamps, channels = self.read_raw_stream()
        else:
            timestamps = np.array([], dtype=np.int64)
            channels = np.array([], dtype=int)
            num_read = 0
            while True:
                # logging.info('in the while loop')
                # logging.info(num_read)
                timestamps_chunk, channels_chunk = self.read_raw_stream()
                timestamps = np.append(timestamps, timestamps_chunk)
                channels = np.append(channels, channels_chunk)
                # Check if we've read enough samples
                new_num_read = np.count_nonzero(channels_chunk == self.tagger_di_clock)
                num_read += new_num_read
                if num_read >= num_to_read:
                    break
        # Convert timestamps to strings since labrad does not support int64s
        # It must be converted to int64s back on the client
        timestamps = timestamps.astype(str).tolist()
        return timestamps, channels
    
    
    