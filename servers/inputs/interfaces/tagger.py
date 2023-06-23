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
from labrad.server import setting
from numba import jit, njit


class Tagger(Counter, ABC):
    @abstractmethod
    def start_tag_stream(self, c, apd_indices=None, gate_indices=None, clock=True):
        """
        Start a tag stream
        Note: These inputs are necessary for the swabian time taggers. The OPX just needs
        the apd_indices to know which apds to play measure() statements on, but that can live in the config and be pulled from there in the sequence.

        Parameters
        ----------
        apd_indices : list
            Indicates the channels for which apds we are using
        gate_indices : list, optional
            Indicates the channels for the gates corresponding to the apds
        clock : boolean, optional
            Indicates if using a clock with the tagger
        """

        pass

    @abstractmethod
    def stop_tag_stream(self, c):
        """
        Stop a tag stream
        """
        pass

    @setting(301, num_to_read="i", returns="*s*i")
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


@njit
def tags_to_counts(
    buffer_channels,
    clock_channel,
    apd_gate_channel,
    apd_channels,
    leftover_channels,
):
    """This is the core counter function for the converting time tags to counts.
    It needs to be fast - if it's not fast enough, we may encounter unexpected
    behavior, like certain samples returning 0 counts when clearly they should
    return something > 0. For that reason, this function lives outside the class
    so that it can be compiled by numba. It's written in very basic (and slow,
    natively) python so that the compiler has no trouble understanding what
    to do.

    Parameters
    ----------
    buffer_channels : _type_
        List of channels returned by the read call on the tagger device
    clock_channel : _type_
        Tagger device's clock channel
    apd_gate_channel : _type_
        Tagger device's APD virtual gate channel
    apd_channels : _type_
        Tagger device's channels hooked up to the APDs
    leftover_channels : _type_
        List containing current leftover channels (i.e. any tags that didn't
        have a clock pulse come after them the last rad request)

    Returns
    -------
    3D array(int)
        Main data structure (return_counts) - the first dimension is for samples,
        the second is for APDs, and the third is for reps/gates.
    array(int)
        Updated leftover_channels
    """

    # Assume a single gate for both APDs: get all the gates once and then
    # count for each APD individually
    open_channel = apd_gate_channel
    close_channel = -open_channel

    # Find clock clicks (sample breaks)
    clock_click_inds = np.flatnonzero(buffer_channels == clock_channel)

    previous_sample_end_ind = None
    sample_end_ind = None

    # Figure out the number of samples and APDs
    num_samples = len(clock_click_inds)
    num_apds = len(apd_channels)

    # Allocate the data structure inside the loop so we know we have a sample
    data_structure_allocated = False

    for dim1 in range(num_samples):
        clock_click_ind = clock_click_inds[dim1]

        # Clock clicks end samples, so they should be included with the
        # sample itself
        sample_end_ind = clock_click_ind + 1

        # Get leftovers and make sure we've got an array for comparison
        # to find click indices
        if previous_sample_end_ind is None:
            join_tuple = (leftover_channels, buffer_channels[0:sample_end_ind])
            sample_channels = np.concatenate(join_tuple)
        else:
            sample_channels = buffer_channels[previous_sample_end_ind:sample_end_ind]

        # Find gate open and close clicks (gate close channel is negative of
        # gate open channel, signifying the falling edge)
        open_inds = np.flatnonzero(sample_channels == open_channel)
        close_inds = np.flatnonzero(sample_channels == close_channel)
        gates = np.column_stack((open_inds, close_inds))

        if not data_structure_allocated:
            num_reps = len(open_inds)
            return_counts = np.empty((num_samples, num_apds, num_reps), dtype=np.int32)
            data_structure_allocated = True

        for dim2 in range(num_apds):
            apd_channel = apd_channels[dim2]
            for dim3 in range(num_reps):
                gate = gates[dim3]
                num_counts = np.count_nonzero(
                    sample_channels[gate[0] : gate[1]] == apd_channel
                )
                return_counts[dim1, dim2, dim3] = num_counts

        previous_sample_end_ind = sample_end_ind

    # No samples were clocked - make a dummy return_counts and add everything to leftovers
    if sample_end_ind is None:
        return_counts = np.empty((0, 0, 0), dtype=np.int32)
        leftover_channels = np.append(leftover_channels, buffer_channels)
    # Reset leftovers from the last sample clock
    else:
        leftover_channels = buffer_channels[sample_end_ind:]

    return return_counts, leftover_channels
