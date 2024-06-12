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

from majorroutines.optimize import expected_counts_check, main, stationary_count_lite
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import LaserKey

# region Internal


@njit(cache=True)
def _2d_gaussian_exp(x0, y0, sigma, x_crop_mesh, y_crop_mesh):
    return np.exp(
        -(((x_crop_mesh - x0) ** 2) + ((y_crop_mesh - y0) ** 2)) / (2 * sigma**2)
    )


@njit(cache=True)
def _optimize_pixel_cost(fit_params, x_crop_mesh, y_crop_mesh, img_array_crop):
    amp, x0, y0, sigma, offset = fit_params
    gaussian_array = offset + amp * _2d_gaussian_exp(
        x0, y0, sigma, x_crop_mesh, y_crop_mesh
    )
    diff_array = gaussian_array - img_array_crop
    return np.sum(diff_array**2)


@njit(cache=True)
def _optimize_pixel_cost_jac(fit_params, x_crop_mesh, y_crop_mesh, img_array_crop):
    amp, x0, y0, sigma, offset = fit_params
    inv_twice_var = 1 / (2 * sigma**2)
    gaussian_exp = _2d_gaussian_exp(x0, y0, sigma, x_crop_mesh, y_crop_mesh)
    x_diff = x_crop_mesh - x0
    y_diff = y_crop_mesh - y0
    spatial_der_coeff = 2 * amp * gaussian_exp * inv_twice_var
    gaussian_jac_0 = gaussian_exp
    gaussian_jac_1 = spatial_der_coeff * x_diff
    gaussian_jac_2 = spatial_der_coeff * y_diff
    gaussian_jac_3 = amp * gaussian_exp * (x_diff**2 + y_diff**2) / (sigma**3)
    gaussian_jac_4 = 1
    coeff = 2 * ((offset + amp * gaussian_exp) - img_array_crop)
    cost_jac = [
        np.sum(coeff * gaussian_jac_0),
        np.sum(coeff * gaussian_jac_1),
        np.sum(coeff * gaussian_jac_2),
        np.sum(coeff * gaussian_jac_3),
        np.sum(coeff * gaussian_jac_4),
    ]
    return np.array(cost_jac)


# endregion


def optimize_pixel_by_ref_img_array(img_array, ref_img_array=None, margin=10):
    """Correlate images to track drift. Not fully implemented yet"""
    if ref_img_array is None:
        config = common.get_config_module()
        ref_img_array = config.ref_img_array

    current_drift = np.array([-2, -3])
    # current_drift = widefield.get_pixel_drift()

    shape = ref_img_array.shape
    ref_center = np.array([shape[1] / 2, shape[0] / 2])

    expected_center = ref_center + current_drift
    crop_img_array = img_array[
        current_drift[1] + margin : current_drift[1] + shape[0] - margin,
        current_drift[0] + margin : current_drift[0] + shape[1] - margin,
    ]

    corr = correlate(ref_img_array, crop_img_array, mode="valid")
    new_drift = np.unravel_index(np.argmax(corr), corr.shape)

    fig, ax = plt.subplots()
    kpl.imshow(ax, ref_img_array)
    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array)
    fig, ax = plt.subplots()
    kpl.imshow(ax, corr)

    print(new_drift)


def optimize_pixel_and_z(nv_sig, do_plot=False):
    num_attempts = 5
    attempt_ind = 0
    while True:
        # xy
        img_array = stationary_count_lite(nv_sig, ret_img_array=True)
        opti_pixel_coords = optimize_pixel_with_img_array(
            img_array, nv_sig, None, do_plot
        )
        counts = widefield.integrate_counts_from_adus(img_array, opti_pixel_coords)

        if nv_sig.expected_counts is not None and expected_counts_check(nv_sig, counts):
            return

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

        if nv_sig.expected_counts is None or expected_counts_check(nv_sig, counts):
            return

        attempt_ind += 1
        if attempt_ind == num_attempts:
            raise RuntimeError("Optimization failed.")
    return opti_pixel_coords


def optimize_pixel(nv_sig, do_plot=False):
    img_array = stationary_count_lite(nv_sig, ret_img_array=True)
    return optimize_pixel_with_img_array(img_array, nv_sig, None, do_plot)


def optimize_pixel_with_img_array(
    img_array,
    nv_sig=None,
    pixel_coords=None,
    do_plot=False,
    return_popt=False,
):
    if do_plot:
        fig, ax = plt.subplots()
        kpl.imshow(ax, img_array, cbar_label="ADUs")

    # Default operations of the routine
    set_pixel_drift = nv_sig is not None
    set_scanning_drift = set_pixel_drift
    pixel_drift_adjust = True
    pixel_drift = None
    radius = None
    do_print = True

    if nv_sig is not None and pixel_coords is not None:
        raise RuntimeError(
            "nv_sig and pixel_coords cannot both be passed to optimize_pixel_with_img_array"
        )

    # Get coordinates
    if nv_sig is not None:
        original_pixel_coords = widefield.get_nv_pixel_coords(nv_sig, False)
        pixel_coords = widefield.get_nv_pixel_coords(
            nv_sig, pixel_drift_adjust, pixel_drift
        )
    if radius is None:
        radius = widefield._get_camera_spot_radius()
    initial_x = pixel_coords[0]
    initial_y = pixel_coords[1]

    # Limit the range to the NV we're looking at
    half_range = radius
    left = round(initial_x - half_range)
    right = round(initial_x + half_range)
    top = round(initial_y - half_range)
    bottom = round(initial_y + half_range)
    x_crop = np.linspace(left, right, right - left + 1)
    y_crop = np.linspace(top, bottom, bottom - top + 1)
    x_crop_mesh, y_crop_mesh = np.meshgrid(x_crop, y_crop)
    img_array_crop = img_array[top : bottom + 1, left : right + 1]

    # Bounds and guesses
    min_img_array_crop = np.min(img_array_crop)
    max_img_array_crop = np.max(img_array_crop)
    bg_guess = min_img_array_crop
    amp_guess = int(img_array[round(initial_y), round(initial_x)] - bg_guess)
    radius_guess = radius / 2
    guess = (amp_guess, initial_x, initial_y, radius_guess, bg_guess)
    diam = radius * 2

    bounds = (
        (0, max_img_array_crop - min_img_array_crop),
        (left, right),
        (top, bottom),
        (1, diam),
        (0, max_img_array_crop),
    )

    args = (x_crop_mesh, y_crop_mesh, img_array_crop)
    res = minimize(
        _optimize_pixel_cost,
        guess,
        bounds=bounds,
        args=args,
        jac=_optimize_pixel_cost_jac,
    )
    popt = res.x

    # Testing
    # opti_pixel_coords = popt[1:3]
    # print(_optimize_pixel_cost(guess, *args))
    # print(_optimize_pixel_cost(popt, *args))
    # print(guess)
    # print(popt)
    # fig, ax = plt.subplots()
    # # gaussian_array = _circle_gaussian(x, y, *popt)
    # # ax.plot(popt[2], popt[1], color="white", zorder=100, marker="o", ms=6)
    # ax.plot(*opti_pixel_coords, color="white", zorder=100, marker="o", ms=6)
    # if type(radius) is list:
    #     for ind in range(len(radius)):
    #         for sub_radius in radius[ind]:
    #             circle = plt.Circle(opti_pixel_coords, sub_radius, fill=False, color="white")
    #             ax.add_patch(circle)
    # else:
    #     circle = plt.Circle(opti_pixel_coords, single_radius, fill=False, color="white")
    #     ax.add_patch(circle)
    # kpl.imshow(ax, img_array)
    # ax.set_xlim([pixel_coords[0] - 15, pixel_coords[0] + 15])
    # ax.set_ylim([pixel_coords[1] + 15, pixel_coords[1] - 15])
    # plt.show(block=True)

    opti_pixel_coords = popt[1:3]
    if set_pixel_drift:
        drift = (np.array(opti_pixel_coords) - np.array(original_pixel_coords)).tolist()
        widefield.set_pixel_drift(drift)
    if set_scanning_drift:
        # widefield.set_scanning_drift_from_pixel_drift()
        widefield.set_all_scanning_drift_from_pixel_drift()
    opti_pixel_coords = opti_pixel_coords.tolist()

    if do_print:
        r_opti_pixel_coords = [round(el, 3) for el in opti_pixel_coords]
        print(f"Optimized pixel coordinates: {r_opti_pixel_coords}")
        counts = widefield.integrate_counts_from_adus(img_array, opti_pixel_coords)
        r_counts = round(counts, 3)
        print(f"Counts at optimized coordinates: {r_counts}")
        print()

    if return_popt:
        return popt
    else:
        return opti_pixel_coords


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
