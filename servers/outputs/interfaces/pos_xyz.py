# -*- coding: utf-8 -*-
"""
Interface for servers that control xy positioning

Created on December 1st, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod


class PosXyz(ABC):
    @abstractmethod
    def write_xyz(self, c, x_coord, y_coord, z_coord):
        """Set the positioner to the passed coordinates

        Parameters
        ----------
        x_coord : numeric
            Could be int or float depending on exact hardware
        y_coord : numeric
            Could be int or float depending on exact hardware
        z_coord : numeric
            Could be int or float depending on exact hardware
        """
        pass

    @abstractmethod
    def reset(self, c):
        """Make sure the device is in a neutral state for the next experiment"""

        pass

    def reset_cfm_opt_out(self, c):
        """Do not reset positioning devices by default with tool_belt.reset_cfm"""

        pass
