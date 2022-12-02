# -*- coding: utf-8 -*-
"""
Interface for servers that control xy positioning and support streaming (advancing
between passed coordinates on a TTL trigger)

Created on December 1st, 2022

@author: mccambria
"""

from abc import abstractmethod
from servers.outputs.interfaces.pos_xy import PosXy
import numpy as np


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


# load_sweep_scan_xy
def get_grid_coords_xy(x_center, y_center, x_range, y_range, num_steps):
    """Load a scan that will wind through the grid defined by the passed
    parameters. Currently we require x_range = y_range.

    Normal scan performed, starts in bottom right corner, and starts
    heading left

    Params
        x_center: float
            Center x voltage of the scan
        y_center: float
            Center y voltage of the scan
        x_range: float
            Full scan range in x
        y_range: float
            Full scan range in y
        num_steps: int
            Number of steps the break the ranges into
        period: int
            Expected period between clock signals in ns

    Returns
        list(float)
            The x voltages that make up the scan
        list(float)
            The y voltages that make up the scan
    """

    ######### Assumes x_range == y_range #########

    if x_range != y_range:
        raise ValueError("x_range must equal y_range for now")

    x_num_steps = num_steps
    y_num_steps = num_steps

    # Force the scan to have square pixels by only applying num_steps
    # to the shorter axis
    half_x_range = x_range / 2
    half_y_range = y_range / 2

    x_low = x_center - half_x_range
    x_high = x_center + half_x_range
    y_low = y_center - half_y_range
    y_high = y_center + half_y_range

    # Apply scale and offset to get the voltages we'll apply to the galvo
    # Note that the polar/azimuthal angles, not the actual x/y positions
    # are linear in these voltages. For a small range, however, we don't
    # really care.
    x_voltages_1d = np.linspace(x_low, x_high, num_steps)
    y_voltages_1d = np.linspace(y_low, y_high, num_steps)

    ######### Works for any x_range, y_range #########

    # Winding cartesian product
    # The x values are repeated and the y values are mirrored and tiled
    # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

    # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
    x_inter = np.concatenate((x_voltages_1d, np.flipud(x_voltages_1d)))
    # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
    if y_num_steps % 2 == 0:  # Even x size
        x_voltages = np.tile(x_inter, int(y_num_steps / 2))
    else:  # Odd x size
        x_voltages = np.tile(x_inter, int(np.floor(y_num_steps / 2)))
        x_voltages = np.concatenate((x_voltages, x_voltages_1d))

    # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
    y_voltages = np.repeat(y_voltages_1d, x_num_steps)

    voltages = np.vstack((x_voltages, y_voltages))

    self.load_stream_writer_xy(c, "Galvo-load_sweep_scan_xy", voltages, period)

    return x_voltages_1d, y_voltages_1d


def load_cross_scan_xy(
    self, c, x_center, y_center, xy_range, num_steps, period
):
    """Load a scan that will first step through xy_range in x keeping y
    constant at its center, then step through xy_range in y keeping x
    constant at its center.

    Params
        x_center: float
            Center x voltage of the scan
        y_center: float
            Center y voltage of the scan
        xy_range: float
            Full scan range in x/y
        num_steps: int
            Number of steps the break the x/y range into
        period: int
            Expected period between clock signals in ns

    Returns
        list(float)
            The x voltages that make up the scan
        list(float)
            The y voltages that make up the scan
    """

    half_xy_range = xy_range / 2

    x_low = x_center - half_xy_range
    x_high = x_center + half_xy_range
    y_low = y_center - half_xy_range
    y_high = y_center + half_xy_range

    x_voltages_1d = np.linspace(x_low, x_high, num_steps)
    y_voltages_1d = np.linspace(y_low, y_high, num_steps)

    x_voltages = np.concatenate([x_voltages_1d, np.full(num_steps, x_center)])
    y_voltages = np.concatenate([np.full(num_steps, y_center), y_voltages_1d])

    voltages = np.vstack((x_voltages, y_voltages))

    self.load_stream_writer_xy(c, "Galvo-load_cross_scan_xy", voltages, period)

    return x_voltages_1d, y_voltages_1d


def load_scan_x(self, c, x_center, y_center, scan_range, num_steps, period):
    """Load a scan that will step through scan_range in x keeping y
    constant at its center.

    Params
        x_center: float
            Center x voltage of the scan
        y_center: float
            Center y voltage of the scan
        scan_range: float
            Full scan range in x/y
        num_steps: int
            Number of steps the break the x/y range into
        period: int
            Expected period between clock signals in ns

    Returns
        list(float)
            The x voltages that make up the scan
    """

    half_scan_range = scan_range / 2

    x_low = x_center - half_scan_range
    x_high = x_center + half_scan_range

    x_voltages = np.linspace(x_low, x_high, num_steps)
    y_voltages = np.full(num_steps, y_center)

    voltages = np.vstack((x_voltages, y_voltages))

    self.load_stream_writer_xy(c, "Galvo-load_scan_x", voltages, period)

    return x_voltages


def load_scan_y(self, c, x_center, y_center, scan_range, num_steps, period):
    """Load a scan that will step through scan_range in y keeping x
    constant at its center.

    Params
        x_center: float
            Center x voltage of the scan
        y_center: float
            Center y voltage of the scan
        scan_range: float
            Full scan range in x/y
        num_steps: int
            Number of steps the break the x/y range into
        period: int
            Expected period between clock signals in ns

    Returns
        list(float)
            The y voltages that make up the scan
    """

    half_scan_range = scan_range / 2

    y_low = y_center - half_scan_range
    y_high = y_center + half_scan_range

    x_voltages = np.full(num_steps, x_center)
    y_voltages = np.linspace(y_low, y_high, num_steps)

    voltages = np.vstack((x_voltages, y_voltages))

    self.load_stream_writer_xy(c, "Galvo-load_scan_y", voltages, period)

    return y_voltages


def load_arb_scan_xy(self, c, x_points, y_points, period):
    """Load a scan that goes between points. E.i., starts at [1,1] and
    then on a clock pulse, moves to [2,1]. Can work for arbitrarily large
    number of points
    (previously load_two_point_xy_scan)

    Params
        x_points: list(float)
            X values correspnding to positions in x
            y_points: list(float)
            Y values correspnding to positions in y
        period: int
            Expected period between clock signals in ns

    """

    voltages = np.vstack((x_points, y_points))

    self.load_stream_writer_xy(c, "Galvo-load_arb_scan_xy", voltages, period)

    return


def load_circle_scan_xy(self, c, radius, num_steps, period):
    """Load a circle scan centered about 0,0. Useful for testing cat's eye
    stationary point. For this reason, the scan runs continuously, not
    just until it makes it through all the samples once.

    Params
        radius: float
            Radius of the circle in V
        num_steps: int
            Number of steps the break the x/y range into
        period: int
            Expected period between clock signals in ns

    Returns
        list(float)
            The x voltages that make up the scan
        list(float)
            The y voltages that make up the scan
    """

    angles = np.linspace(0, 2 * np.pi, num_steps)
    x_voltages = radius * np.sin(angles)
    y_voltages = radius * np.cos(angles)
    voltages = np.vstack((x_voltages, y_voltages))
    self.load_stream_writer_xy(
        c, "Galvo-load_circle_scan_xy", voltages, period, True
    )
    return x_voltages, y_voltages


def load_two_point_xy(self, c, x1, y1, x2, y2, period):
    """Flip back an forth continuously between two points"""

    x_voltages = [x1, x2] * 32
    y_voltages = [y1, y2] * 32
    voltages = np.vstack((x_voltages, y_voltages))
    self.load_stream_writer_xy(
        c, "Galvo-load_two_point_xy", voltages, period, True
    )
    return x_voltages, y_voltages
