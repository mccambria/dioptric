# -*- coding: utf-8 -*-
"""
Interface for servers that control xy positioning

Created on December 1st, 2022

@author: mccambria
"""

from abc import ABC, abstractmethod


class PosXy(ABC):
    @abstractmethod
    def write_xy(self, c, x_coord, y_coord):
        """Set the positioner to the passed coordinates

        Parameters
        ----------
        x_coord : numeric
            Could be int or float depending on exact hardware
        y_coord : numeric
            Could be int or float depending on exact hardware
        """
        pass

    @abstractmethod
    def write_x(self, c, x_coord):
        """Set the positioner to the passed coordinates

        Parameters
        ----------
        x_coord : numeric
            Could be int or float depending on exact hardware
        """
        pass

    @abstractmethod
    def write_y(self, c, y_coord):
        """Set the positioner to the passed coordinates

        Parameters
        ----------
        y_coord : numeric
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
