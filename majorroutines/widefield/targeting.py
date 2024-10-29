# -*- coding: utf-8 -*-
"""
Widefield extension of the standard optimize in majorroutines

Created Fall 2023

@author: mccambria
"""

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy import inf
from scipy.optimize import minimize
from scipy.signal import correlate

from majorroutines import optimize_xyz, targeting
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey


def optimize_slm_calibration(nv_sig, do_plot=False):
    num_attempts = 5
    attempt_ind = 0
    while True:
        # xy
        img_array = stationary_count_lite(nv_sig, ret_img_array=True)
        opti_pixel_coords, pixel_drift = optimize_pixel_with_img_array(
            img_array, nv_sig, None, do_plot, return_drift=True
        )
        counts = widefield.integrate_counts_from_adus(img_array, opti_pixel_coords)

        if nv_sig.expected_counts is not None and check_expected_counts(nv_sig, counts):
            return pixel_drift

        # z
        try:
            _, counts = main(
                nv_sig,
                axes_to_optimize=[2],
                opti_necessary=True,
                do_plot=do_plot,
                num_attempts=2,
            )
        except Exception:
            pass

        if nv_sig.expected_counts is None or check_expected_counts(nv_sig, counts):
            return pixel_drift

        attempt_ind += 1
        if attempt_ind == num_attempts:
            raise RuntimeError("Optimization failed.")


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1521874556597, load_npz=True)
    img_array = np.array(data["img_array"])

    print(widefield.integrate_counts_from_adus(img_array, (126.687, 128.27)))
    print(widefield.integrate_counts_from_adus(img_array, (126.687 - 0.2, 128.27)))
    # data = dm.get_raw_data(file_id=1522533978767, load_npz=True)
    # ref_img_array = np.array(data["img_array"])

    # optimize_pixel_by_ref_img_array(img_array, ref_img_array)

    plt.show(block=True)
