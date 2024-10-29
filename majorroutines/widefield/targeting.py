# -*- coding: utf-8 -*-
"""
Widefield extension of the standard optimize in majorroutines

Created 29 Oct, 2024

@author: mccambria
@author: sbchand
"""

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy import inf
from scipy.optimize import minimize
from scipy.signal import correlate

from majorroutines.targeting import optimize
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey


def optimize_pixel(nv_sig):
    """Optimize pixel-based coordinates."""
    try:
        return optimize(nv_sig, coords_key=CoordsKey.PIXEL)
    except Exception as e:
        print(f"Error during pixel optimization: {e}")
        return None


def optimize_sample(nv_sig):
    """Optimize sample-based coordinates."""
    try:
        return optimize(nv_sig, coords_key=CoordsKey.SAMPLE)
    except Exception as e:
        print(f"Error during sample optimization: {e}")
        return None


def optimize_sample_xy(nv_sig):
    """Optimize sample-based coordinates along XY axes."""
    try:
        return optimize(nv_sig, coords_key=CoordsKey.SAMPLE, axes=Axes.XY)
    except Exception as e:
        print(f"Error during XY sample optimization: {e}")
        return None


def optimize_sample_z(nv_sig):
    """Optimize sample-based coordinates along the Z-axis."""
    try:
        return optimize(nv_sig, coords_key=CoordsKey.SAMPLE, axes=Axes.Z)
    except Exception as e:
        print(f"Error during Z-axis sample optimization: {e}")
        return None


if __name__ == "__main__":
    try:
        # Initialize plotting
        kpl.init_kplotlib()

        # Load data from the data manager
        data = dm.get_raw_data(file_id=1521874556597, load_npz=True)
        img_array = np.array(data["img_array"])

        # Perform some widefield operations for testing
        counts1 = widefield.integrate_counts_from_adus(img_array, (126.687, 128.27))
        counts2 = widefield.integrate_counts_from_adus(img_array, (126.487, 128.27))

        print(f"Counts at (126.687, 128.27): {counts1}")
        print(f"Counts at (126.487, 128.27): {counts2}")

        # Show the plots
        plt.show(block=True)

    except Exception as e:
        print(f"An error occurred in the main block: {e}")
