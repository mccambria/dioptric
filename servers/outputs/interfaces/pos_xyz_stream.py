# -*- coding: utf-8 -*-
"""
Interface for servers that control xy positioning and support streaming (advancing
between passed coordinates on a TTL trigger)

Created on December 1st, 2022

@author: mccambria
"""

from abc import abstractmethod
from servers.outputs.interfaces.pos_xyz import PosXyz


class PosXyzStream(PosXyz):
    @abstractmethod
    def load_stream_xyz(self, c, x_coords, y_coords,  z_coords, continuous=False):
        """Loads a stream of x and y coordinates that will be stepped through on clock
        pulses. The first coordinate (i.e. (x_coords[0], y_coords[0])) is written immediately

        Parameters
        ----------
        x_coords : list(numeric)
            Could be int or float depending on exact hardware
        y_coords : list(numeric)
            Could be int or float depending on exact hardware
        z_coords : list(numeric)
            Could be int or float depending on exact hardware
        continuous : bool
            If True, loop through the voltages continuously, going back to the first
            pair after the last pair was written
        """

        pass
