# -*- coding: utf-8 -*-
"""
Optimize on an NV

Largely rewritten August 16th, 2023

@author: mccambria
"""


# region Imports and constant

import copy
import dataclasses
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numpy import inf
from scipy.optimize import curve_fit, minimize

from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import (
    Axes,
    CollectionMode,
    CoordsKey,
    NVSig,
    PosControlMode,
    VirtualLaserKey,
)

# endregion
# region Plotting functions


def _create_figure(positioner):
    kpl.init_kplotlib(kpl.Size.SMALL)
    fig, axes_pack = plt.subplots(1, 3, figsize=kpl.double_figsize)
    axis_titles = ["X axis", "Y axis", "Z axis"]
    for ind in range(3):
        ax = axes_pack[ind]
        ax.set_title(axis_titles[ind])
        xlabel = pos.get_positioner_units(positioner)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
    return fig


def _update_figure(fig, axis_ind, scan_vals, count_vals, text=None):
    axes = fig.get_axes()
    ax = axes[axis_ind]
    ax.plot(scan_vals, count_vals)
    if text is not None:
        kpl.anchored_text(ax, text, kpl.Loc.LOWER_RIGHT)
    kpl.show()


def _fit_gaussian(scan_vals, count_vals, axis_ind, positive_amplitude=True, fig=None):
    # Param order: amplitude, center, sd width, offset
    fit_func = tb.gaussian
    bg_guess = min(count_vals) if positive_amplitude else max(count_vals)
    low = np.min(scan_vals)
    high = np.max(scan_vals)
    scan_range = high - low
    center_guess = (high + low) / 2
    amplitude_guess = (
        max(count_vals) - bg_guess if positive_amplitude else min(count_vals) - bg_guess
    )
    guess = (amplitude_guess, center_guess, scan_range / 3, bg_guess)
    popt = None
    try:
        amplitude_lower = 0 if positive_amplitude else -inf
        amplitude_upper = inf if positive_amplitude else 0
        low_bounds = [amplitude_lower, low, 0, 0]
        high_bounds = [amplitude_upper, high, inf, inf]
        bounds = (low_bounds, high_bounds)
        popt, pcov = curve_fit(fit_func, scan_vals, count_vals, p0=guess, bounds=bounds)
        # Consider it a failure if we railed or somehow got out of bounds
        for ind in range(len(popt)):
            param = popt[ind]
            if not (low_bounds[ind] < param < high_bounds[ind]):
                popt = None
    except Exception as ex:
        print(ex)
        # pass

    if popt is None:
        print("Optimization failed for axis {}".format(axis_ind))

    # Plot
    if (fig is not None) and (popt is not None):
        # Plot the fit
        linspace_scan_vals = np.linspace(low, high, num=1000)
        fit_count_vals = fit_func(linspace_scan_vals, *popt)
        # Add popt to the axes
        r_popt = [round(el, 3) for el in popt]
        text = rf"a={r_popt[0]}\n $\mu$={r_popt[1]}\n $\sigma$={r_popt[2]}\n offset={r_popt[3]}"
        _update_figure(fig, axis_ind, linspace_scan_vals, fit_count_vals, text)

    center = None
    if popt is not None:
        center = popt[1]

    return center


# endregion
# region Private axis optimization functions


def _read_counts_counter_stream(axis_ind=None, scan_vals=None):
    if axis_ind is not None:
        axis_stream_fn = pos.get_positioner_stream_fn(axis_ind)
        axis_stream_fn(scan_vals)
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_gen()
    counts = []
    num_read_so_far = 0
    counter.start_tag_stream()
    if axis_ind is not None:
        num_steps = len(scan_vals)
    else:
        num_steps = 1
    pulse_gen.stream_start(num_steps)
    while num_read_so_far < num_steps:
        if tb.safe_stop():
            break
        new_samples = counter.read_counter_simple()
        num_new_samples = len(new_samples)
        if num_new_samples > 0:
            counts.extend(new_samples)
            num_read_so_far += num_new_samples
    counter.stop_tag_stream()
    return [np.array(counts, dtype=int)]


def _read_counts_counter_step(axis_ind=None, scan_vals=None):
    if axis_ind is not None:
        axis_write_fn = pos.get_positioner_write_fn(positioner, axis_ind)
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_gen()
    counter.start_tag_stream()
    counts = []
    for ind in range(len(scan_vals)):
        if tb.safe_stop():
            break
        if axis_ind is not None:
            axis_write_fn(scan_vals[ind])
        pulse_gen.stream_start(1)
        new_samples = counter.read_counter_simple(1)
        counts.append(np.average(new_samples))
    counter.stop_tag_stream()
    return [np.array(counts, dtype=int)]


def _read_counts_camera_step(nv_sig, axis_ind=None, scan_vals=None):
    if axis_ind is not None:
        axis_write_fn = pos.get_positioner_write_fn(positioner, axis_ind)
    pixel_coords = pos.get_nv_coords(nv_sig, CoordsKey.PIXEL)
    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()
    counts = []
    camera.arm()
    for ind in range(len(scan_vals)):
        if tb.safe_stop():
            break
        if axis_ind is not None:
            axis_write_fn(scan_vals[ind])
        pulse_gen.stream_start()
        img_str = camera.read()
        img_array_adus, baseline = widefield.img_str_to_array(img_str)
        img_array = widefield.adus_to_photons(img_array_adus, baseline=baseline)
        sample = widefield.integrate_counts(img_array, pixel_coords)
        counts.append(sample)
    camera.disarm()
    return [np.array(counts, dtype=int), img_array]


def _get_opti_virtual_laser_key(positioner):
    if positioner is CoordsKey.SAMPLE:
        laser_key = VirtualLaserKey.IMAGING
    else:
        config = common.get_config_dict()
        positioner_dict = config["Positioning"]["Positioners"][positioner]
        laser_key = positioner_dict["opti_virtual_laser_key"]
    return laser_key


def _read_counts_camera_sequence(
    nv_sig: NVSig,
    positioner=None,
    axis_ind=None,
    scan_vals=None,
    virtual_laser_key=None,
):
    """
    Specific function for widefield setup - XY control from AODs,
    Z control from objective piezo, imaged onto a camera
    """
    # Basic setup
    pixel_coords = pos.get_nv_coords(nv_sig, CoordsKey.PIXEL)
    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()
    if axis_ind is not None:
        num_steps = len(scan_vals)
    else:
        num_steps = 1

    if positioner is not None:
        virtual_laser_key = _get_opti_virtual_laser_key(positioner)
    elif virtual_laser_key is None:
        virtual_laser_key = VirtualLaserKey.IMAGING

    # Sequence setup

    if virtual_laser_key == VirtualLaserKey.IMAGING:
        imaging_laser_dict = tb.get_virtual_laser_dict(virtual_laser_key)
        imaging_laser_name = tb.get_physical_laser_name(virtual_laser_key)
        imaging_laser_positioner = pos.get_laser_positioner(virtual_laser_key)
        imaging_readout = imaging_laser_dict["duration"]
        laser_coords = pos.get_nv_coords(nv_sig, imaging_laser_positioner)
        seq_args = [
            imaging_readout,
            imaging_laser_name,
            [laser_coords[0]],
            [laser_coords[1]],
        ]
        seq_file_name = "simple_readout-scanning.py"
        num_reps = 1
    elif virtual_laser_key == VirtualLaserKey.ION:
        # Get the coordinates for the charge polarization laser
        pol_laser_positioner = pos.get_laser_positioner(VirtualLaserKey.CHARGE_POL)
        pol_coords = pos.get_nv_coords(nv_sig, pol_laser_positioner)

        # Get the coordinates for the ionization laser
        ion_laser_positioner = pos.get_laser_positioner(VirtualLaserKey.ION)
        ion_coords = pos.get_nv_coords(nv_sig, ion_laser_positioner)

        seq_args = [pol_coords, ion_coords]
        seq_file_name = "optimize_ionization_laser_coords.py"
        num_reps = 50
    if axis_ind is None or axis_ind == 2:
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file_name, seq_args_string, num_reps)
    # # For z the sequence is the same every time and z is moved manually
    if axis_ind == 2:
        axis_write_fn = pos.get_positioner_write_fn(positioner, axis_ind)

    # print(seq_args)
    # return

    # Collect the counts
    counts = []
    camera.arm()
    for ind in range(num_steps):
        if tb.safe_stop():
            break

        # Modify the sequence as necessary and start the pulse generator
        if axis_ind is not None:
            val = scan_vals[ind]
            if axis_ind in [0, 1]:
                if virtual_laser_key == VirtualLaserKey.IMAGING:
                    seq_args[-2 + axis_ind] = [val]
                elif virtual_laser_key == VirtualLaserKey.ION:
                    seq_args[1][axis_ind] = val
                seq_args_string = tb.encode_seq_args(seq_args)

                pulse_gen.stream_load(seq_file_name, seq_args_string, num_reps)
            elif axis_ind == 2:
                axis_write_fn(val)

        # Read the camera images
        img_array_list = []

        def rep_fn(rep_ind):
            img_str = camera.read()
            sub_img_array, _ = widefield.img_str_to_array(img_str)
            img_array_list.append(sub_img_array)

        widefield.rep_loop(num_reps, rep_fn)

        # Process the result
        img_array = np.mean(img_array_list, axis=0)
        sample = widefield.integrate_counts_from_adus(img_array, pixel_coords)
        counts.append(sample)

    camera.disarm()

    return [np.array(counts, dtype=int), img_array]


def _find_center_coords(nv_sig: NVSig, positioner, axis_ind, fig=None):
    """Optimize on just one axis (0, 1, 2) for (x, y, z)."""

    num_steps = 20
    scan_range = pos.get_positioner_optimize_range(positioner)
    if positioner is CoordsKey.Z:
        axis_center = pos.get_nv_coords(nv_sig, positioner)
    else:
        coords = pos.get_nv_coords(nv_sig, positioner)
        axis_center = coords[axis_ind]
    scan_vals = pos.get_scan_1d(axis_center, scan_range, num_steps)
    # Read counts for the scanned values
    ret_vals = _read_counts(nv_sig, positioner, axis_ind, scan_vals)
    counts = ret_vals[0]

    # Plot the data if figure is provided
    if fig is not None:
        _update_figure(fig, axis_ind, scan_vals, counts)

    # Perform Gaussian fit on the counts
    laser_key = _get_opti_virtual_laser_key(positioner)
    positive_amplitude = laser_key != VirtualLaserKey.ION
    opti_coord = _fit_gaussian(scan_vals, counts, axis_ind, positive_amplitude, fig)

    return opti_coord, scan_vals, counts


def _find_center_pixel_coords(nv_sig, do_plot=False):
    img_array = stationary_count_lite(nv_sig, ret_img_array=True)
    return _find_center_pixel_coords_with_img_array(img_array, nv_sig, None, do_plot)


def _find_center_pixel_coords_with_img_array(
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
    radius = None
    do_print = True

    if nv_sig is not None and pixel_coords is not None:
        raise RuntimeError(
            "nv_sig and pixel_coords cannot both be passed to optimize_pixel_with_img_array"
        )

    # Get coordinates
    if pixel_coords is None:
        pixel_coords = pos.get_nv_coords(nv_sig, CoordsKey.PIXEL)
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

    opti_pixel_coords = popt[1:3].tolist()

    if do_print:
        r_opti_pixel_coords = [round(el, 3) for el in opti_pixel_coords]
        counts = widefield.integrate_counts_from_adus(img_array, opti_pixel_coords)
        r_counts = round(counts, 3)
        print(f"Optimized pixel coordinates: {r_opti_pixel_coords}")
        print(f"Counts at optimized pixel coordinates: {r_counts}")
        print()

    if return_popt:
        return popt
    else:
        return opti_pixel_coords


def _read_counts(
    nv_sig, positioner=None, axis_ind=None, scan_vals=None, virtual_laser_key=None
):
    # Position us at the starting point
    pos.set_xyz_on_nv(nv_sig)
    # How we conduct the scan depends on the config
    config = common.get_config_dict()
    collection_mode = config["collection_mode"]
    pulse_gen = tb.get_server_pulse_gen()

    # Assume the lasers are sequence controlled if using camera
    if collection_mode == CollectionMode.CAMERA:
        ret_vals = _read_counts_camera_sequence(
            nv_sig, positioner, axis_ind, scan_vals, virtual_laser_key
        )

    elif collection_mode == CollectionMode.COUNTER:
        laser_key = _get_opti_virtual_laser_key(positioner)
        laser_dict = tb.get_virtual_laser_dict(laser_key)
        laser_name = tb.get_physical_laser_name(laser_key)

        if positioner != CoordsKey.SAMPLE:
            msg = "Optimization with a counter is only implemented for global coordinates."
            raise NotImplementedError(msg)
        if laser_key != VirtualLaserKey.IMAGING:
            msg = "Optimization with a counter is only implemented for imaging lasers."
            raise NotImplementedError(msg)

        delay = pos.get_positioner_delay(positioner=positioner)
        control_mode = pos.get_positioner_control_mode(positioner)

        seq_file_name = "simple_readout.py"
        readout = laser_dict["duration"]
        seq_args = [delay, readout, laser_name]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file_name, seq_args_string)

        if control_mode == PosControlMode.STEP:
            ret_vals = _read_counts_counter_step(axis_ind, scan_vals)
        elif control_mode == PosControlMode.STREAM:
            ret_vals = _read_counts_counter_stream(axis_ind, scan_vals)

    return ret_vals


def _create_opti_nv_sig(nv_sig, opti_coords, coords_key):
    """Make a copy of nv_sig with coords updated so that positioner
    is set to opti_coords after adjusting for drift

    Parameters
    ----------
    nv_sig : _type_
        _description_
    opti_coords : _type_
        _description_
    positioner : _type_
        _description_
    """

    opti_nv_sig = copy.deepcopy(nv_sig)
    drift = pos.get_drift(coords_key)
    if coords_key is CoordsKey.Z:
        drift *= -1
    else:
        drift = [-1 * el for el in drift]
    adj_opti_coords = pos.adjust_coords_for_drift(opti_coords, drift)
    pos.set_nv_coords(opti_nv_sig, adj_opti_coords, coords_key)
    return opti_nv_sig


# endregion
# region Misc private functions


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
# region General public functions


def stationary_count_lite(
    nv_sig, virtual_laser_key=VirtualLaserKey.IMAGING, ret_img_array=False
):
    ret_vals = _read_counts(nv_sig, virtual_laser_key=virtual_laser_key)
    counts = ret_vals[0]
    avg_counts = np.average(counts)
    if ret_img_array:
        return ret_vals[1]
    return avg_counts


def check_expected_counts(nv_sig, counts):
    expected_counts = nv_sig.expected_counts
    lower_bound = 0.95 * expected_counts
    upper_bound = 1.1 * expected_counts
    return lower_bound < counts < upper_bound


def compensate_for_drift(nv_sig: NVSig, no_crash=False):
    """Compensate for drift either by adjusting the sample position to recenter the sample
    or by adjusting the laser positioners to account for the drift

    Parameters
    ----------
    nv_sig : NVSig
        NV to optimize on
    no_crash : bool, optional
        flag to disable RuntimeError raised if drift compensation fails, by default False

    Raises
    ------
    RuntimeError
        Crashes out if drift compensation fails - disable by setting the no_crash flag
    """

    ### Check if drift compensation is globally disabled

    config = common.get_config_dict()
    disable_drift_compensation = config.get("disable_drift_compensation", False)
    if disable_drift_compensation:
        return

    ### Check if drift compensation is necessary by reading counts at current coordinates

    expected_counts = nv_sig.expected_counts
    print(f"Expected counts: {expected_counts}")
    current_counts = stationary_count_lite(nv_sig)
    compensation_necessary = expected_counts is None or not check_expected_counts(
        nv_sig, current_counts
    )
    print(f"Counts at initial coordinates: {current_counts}")
    if not compensation_necessary:
        print("Drift compensation unnecessary.")
        return

    ### Compensation is necessary - find the center coordinates along each available axis

    # Determine what axes are available and what positioner to use
    xy_coords_key = pos.get_drift_xy_coords_key()
    passed_coords = pos.get_nv_coords(nv_sig, xy_coords_key, drift_adjust=False)
    disable_z_drift_compensation = config.get("disable_z_drift_compensation", False)
    if disable_z_drift_compensation or not pos.has_z_positioner():
        axes = Axes.XY.value
    else:
        axes = Axes.XYZ.value
        passed_z_coord = pos.get_nv_coords(nv_sig, CoordsKey.Z, drift_adjust=False)
        passed_coords.append(passed_z_coord)

    opti_succeeded = False
    num_attempts = 5

    # Loop through attempts until we succeed or give up
    for ind in range(num_attempts):
        # Setup
        if opti_succeeded or tb.safe_stop():
            break
        print(f"Attempt number {ind+1}")
        axis_failed = False

        # Main loop
        for axis_ind in axes:
            # Check if z optimization is necessary after xy optimization
            if axis_ind == 2 and axes == [0, 1, 2]:
                current_counts = stationary_count_lite(nv_sig)
                if expected_counts is not None and check_expected_counts(
                    nv_sig, current_counts
                ):
                    print("Z drift compensation unnecessary.")
                    opti_succeeded = True
                    break

            # Perform the optimization
            if axis_ind <= 1 and xy_coords_key is CoordsKey.PIXEL:
                if axis_ind == 0:
                    opti_xy_coords = _find_center_pixel_coords(nv_sig)
                opti_coord = opti_xy_coords[axis_ind]
            else:
                positioner = xy_coords_key if axis_ind <= 1 else CoordsKey.Z
                ret_vals = _find_center_coords(nv_sig, positioner, axis_ind)
                opti_coord = ret_vals[0]

            # Set drift if we succeeded
            if opti_coord is None:
                axis_failed = True
            else:
                # Determine if the drift should be added on to the existing drift
                # or calculated from scratch. It should be added if the drift is
                # compensated for by moving the sample but we measured the drift
                # in a different coordinate space
                cumulative = (
                    pos.has_sample_positioner()
                    and xy_coords_key is not CoordsKey.SAMPLE
                    and axis_ind in [0, 1]
                )

                drift_val = opti_coord - passed_coords[axis_ind]
                pos.set_drift_val(drift_val, axis_ind, cumulative)

        # Try again if any individual axis failed
        if axis_failed:
            continue

        # Check the counts - if the threshold is not set, we just do one pass and succeed
        current_counts = stationary_count_lite(nv_sig)
        print(f"Counts after drift compensation: {round(current_counts, 1)}")
        if expected_counts is None:
            opti_succeeded = True
            break
        elif check_expected_counts(nv_sig, current_counts):
            opti_succeeded = True
        else:
            print("Counts after drift compensation out of bounds.")

    ### Cleanup and return

    # Make sure we're sitting back on the NV regardless of what happened
    pos.set_xyz_on_nv(nv_sig)

    if opti_succeeded:
        print("Drift compensation succeeded!")
    elif not no_crash:
        raise RuntimeError("Maxed out number of attempts. Drift compensation failed.")

    print()


def optimize_pixel(nv_sig, img_array=None):
    do_plot = True
    if img_array is not None:
        opti_coords = _find_center_pixel_coords_with_img_array(
            img_array, nv_sig, do_plot=do_plot
        )
    else:
        opti_coords = _find_center_pixel_coords(nv_sig, do_plot=do_plot)

    # Make a copy of the passed NV, but with coords updated so that the
    # positioner is set to the opti_coords after adjusting for drift
    opti_nv_sig = _create_opti_nv_sig(nv_sig, opti_coords, CoordsKey.PIXEL)

    virtual_laser_key = VirtualLaserKey.IMAGING
    final_counts = stationary_count_lite(opti_nv_sig, virtual_laser_key)

    print(f"Optimized coordinates: {opti_coords}")
    print(f"Counts at optimized coordinates: {final_counts}")

    return opti_coords, final_counts


def optimize(
    nv_sig: NVSig, coords_key: str = CoordsKey.SAMPLE, axes: Axes | None = None
):
    """Optimize coords for the passed NV and coords_key, leaving other positioners fixed.
    Returns actual optimal coordinates without drift compensation. Use this when first
    characterizing an NV

    Parameters
    ----------
    nv_sig : NVSig
        _description_
    coords_key : str, optional
        _description_, by default CoordsKey.SAMPLE
    axes : Axes, optional
        _description_, by default None

    Returns
    -------
    (list, int)
        (opti_coords, final_counts)
    """

    # Divert for pixel coords since we can just take a picture
    if coords_key == CoordsKey.PIXEL:
        return optimize_pixel(nv_sig)

    ### Setup

    start_time = time.time()

    if axes is None:
        axes = Axes.Z if coords_key == CoordsKey.Z else Axes.XY
    axes = axes.value

    fig = _create_figure(coords_key)
    scan_vals = [None] * 3
    scan_counts = [None] * 3
    opti_coords = [None] * 3

    ### Perform the optimizations

    for axis_ind in axes:
        ret_vals = _find_center_coords(nv_sig, coords_key, axis_ind, fig)
        opti_coords[axis_ind] = ret_vals[0]
        scan_vals[axis_ind] = ret_vals[1]
        scan_counts[axis_ind] = ret_vals[2]

    ### Check the counts at the optimized coordinates and report the results

    # Make a copy of the passed NV, but with coords updated so that the
    # positioner is set to the opti_coords after adjusting for drift
    opti_nv_sig = _create_opti_nv_sig(nv_sig, opti_coords, coords_key)
    virtual_laser_key = _get_opti_virtual_laser_key(coords_key)
    final_counts = stationary_count_lite(opti_nv_sig, virtual_laser_key)

    opti_coords = [round(coord, 3) for coord in opti_coords]
    print(f"Optimized coordinates: {opti_coords}")
    print(f"Counts at optimized coordinates: {final_counts}")

    ### Clean up and save the data

    tb.reset_cfm()
    end_time = time.time()
    time_elapsed = end_time - start_time

    timestamp = dm.get_time_stamp()

    raw_data = {
        "timestamp": timestamp,
        "time_elapsed": time_elapsed,
        "nv_sig": nv_sig,
        "opti_coords": opti_coords,
        "axes": axes,
        "scan_vals": scan_vals,
        "scan_counts": scan_counts,
        "final_counts": final_counts,
    }

    nv_name = nv_sig.name
    filePath = dm.get_file_path(__file__, timestamp, nv_name)
    if fig is not None:
        dm.save_figure(fig, filePath)
    dm.save_raw_data(raw_data, filePath)

    # Return the optimized coordinates we found and the final counts
    return opti_coords, final_counts


# endregion

if __name__ == "__main__":
    file_name = "2023_09_21-21_07_51-widefield_calibration_nv1"
    data = dm.get_raw_data(file_name)

    fig = _create_figure()
    nv_sig = data["nv_sig"]
    keys = ["x", "y", "z"]
    for axis_ind in range(3):
        scan_vals = data[f"{keys[axis_ind]}_scan_vals"]
        count_vals = data[f"{keys[axis_ind]}_counts"]
        _update_figure(fig, axis_ind, scan_vals, count_vals)
        _fit_gaussian(scan_vals, count_vals, axis_ind, True, fig)

    plt.show(block=True)
