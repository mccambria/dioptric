# -*- coding: utf-8 -*-
"""
Interface for servers that control xy positioning and support streaming (advancing
between passed coordinates on a TTL trigger)

Created on December 1st, 2022

@author: mccambria
"""

from abc import abstractmethod
from servers.outputs.interfaces.pos_xy import PosXy


class PosXyStream(PosXy):
    @abstractmethod
    def load_stream_xy(self, c, x_coords, y_coords, continuous=False):
        """Loads a stream of x and y coordinates that will be stepped through on clock
        pulses. The first coordinate (i.e. (x_coords[0], y_coords[0])) is written immediately

        Parameters
        ----------
        x_coords : list(numeric)
            Could be int or float depending on exact hardware
        y_coords : list(numeric)
            Could be int or float depending on exact hardware
        continuous : bool
            If True, loop through the voltages continuously, going back to the first
            pair after the last pair was written
        """

        pass

    @abstractmethod
    def load_stream_xy(self, c, x_coords, continuous=False):
        """Loads a stream of x coordinates that will be stepped through on clock
        pulses. The first coordinate (i.e. x_coords[0]) is written immediately

        Parameters
        ----------
        x_coords : list(numeric)
            Could be int or float depending on exact hardware
        continuous : bool
            If True, loop through the voltages continuously, going back to the first
            pair after the last pair was written
        """

        pass

    @abstractmethod
    def load_stream_xy(self, c, y_coords, continuous=False):
        """Loads a stream of y coordinates that will be stepped through on clock
        pulses. The first coordinate (i.e. y_coords[0]) is written immediately

        Parameters
        ----------
        y_coords : list(numeric)
            Could be int or float depending on exact hardware
        continuous : bool
            If True, loop through the voltages continuously, going back to the first
            pair after the last pair was written
        """

        pass
