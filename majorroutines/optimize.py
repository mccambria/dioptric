# -*- coding: utf-8 -*-
"""
Optimize on an NV

Largely rewritten August 16th, 2023

@author: mccambria
"""


# region Imports and constant


import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import copy
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import common
from utils import widefield
from utils.constants import ControlMode, CountFormat, CollectionMode
from utils.constants import LaserKey, LaserPosMode

# endregion
# region Plotting functions


def _create_figure():
    kpl.init_kplotlib(kpl.Size.SMALL)
    config = common.get_config_dict()
    fig, axes_pack = plt.subplots(1, 3, figsize=kpl.double_figsize)
    axis_titles = ["X axis", "Y axis", "Z axis"]
    for ind in range(3):
        ax = axes_pack[ind]
        ax.set_title(axis_titles[ind])
        xlabel_key = "xy_units" if ind in [0, 1] else "z_units"
        xlabel = config["Positioning"][xlabel_key]
        ax.set_xlabel(xlabel)
        count_format = config["count_format"]
        if count_format == CountFormat.RAW:
            ylabel = "Counts"
        elif count_format == CountFormat.KCPS:
            ylabel = "Count rate (kcps)"
        ax.set_ylabel(ylabel)
    return fig


def _update_figure(fig, axis_ind, scan_vals, count_vals, text=None):
    axes = fig.get_axes()
    ax = axes[axis_ind]
    ax.plot(scan_vals, count_vals)
    if text is not None:
        kpl.anchored_text(ax, text, kpl.Loc.UPPER_RIGHT)
    kpl.flush_update(fig=fig)


def _fit_gaussian(nv_sig, scan_vals, count_vals, axis_ind, fig=None):
    # Param order: amplitude, center, sd width, offset
    fit_func = tb.gaussian
    bg_guess = 0.0  # Guess 0
    low = np.min(scan_vals)
    high = np.max(scan_vals)
    scan_range = high - low
    coords = nv_sig["coords"]
    guess = (max(count_vals) - bg_guess, coords[axis_ind], scan_range / 3, bg_guess)
    popt = None
    try:
        low_bounds = [0, low, 0, 0]
        high_bounds = [inf, high, inf, inf]
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
        text = "a={:.3f}\n $\mu$={:.3f}\n $\sigma$={:.3f}\n offset={:.3f}".format(*popt)
        _update_figure(fig, axis_ind, linspace_scan_vals, fit_count_vals, text)

    center = None
    if popt is not None:
        center = popt[1]

    return center


# endregion
# region Private axis optimization functions


def _read_counts_counter_stream(cxn, axis_ind, scan_vals):
    axis_stream_fn = pos.get_axis_stream_fn(axis_ind)
    axis_stream_fn(scan_vals)
    counter = tb.get_server_counter(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    counts = []
    num_read_so_far = 0
    counter.start_tag_stream()
    num_steps = len(scan_vals)
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
    return np.array(counts, dtype=int)


def _read_counts_counter_step(cxn, axis_ind, scan_vals):
    axis_write_fn = pos.get_axis_write_fn(axis_ind)
    counter = tb.get_server_counter(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    counter.start_tag_stream()
    counts = []
    for ind in range(len(scan_vals)):
        if tb.safe_stop():
            break
        axis_write_fn(scan_vals[ind])
        pulse_gen.stream_start(1)
        new_samples = counter.read_counter_simple(1)
        counts.append(np.average(new_samples))
    counter.stop_tag_stream()
    return np.array(counts, dtype=int)


def _read_counts_camera_step(cxn, axis_ind, scan_vals, nv_sig, laser_key):
    axis_write_fn = pos.get_axis_write_fn(axis_ind)
    pixel_coords = nv_sig["pixel_coords"]
    num_reps = nv_sig[laser_key]["num_reps"]
    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    counts = []
    camera.arm()
    for ind in range(len(scan_vals)):
        if tb.safe_stop():
            break
        axis_write_fn(scan_vals[ind])
        pulse_gen.stream_start(num_reps)
        img_array = camera.read()
        sample = widefield.counts_from_img_array(img_array, pixel_coords)
        counts.append(sample)
    camera.disarm()
    return np.array(counts, dtype=int)


def _read_counts_camera_sequence(cxn, axis_ind, scan_vals, nv_sig, laser_key):
    """
    Specific function for widefield setup - XY control from AODs,
    Z control from objective piezo, imaged onto a camera
    """
    # Basic setup
    control_mode = pos.get_axis_control_mode(axis_ind)
    pixel_coords = nv_sig["pixel_coords"]
    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)

    # Sequence setup
    coords = nv_sig["coords"]
    laser_dict = nv_sig[laser_key]
    laser_name = laser_dict["name"]
    readout = laser_dict["readout_dur"]
    num_reps = laser_dict["num_reps"]
    seq_args = [readout, laser_name, [coords[0]], [coords[1]]]
    seq_file_name = "simple_readout-scanning.py"
    # For z the sequence is the sam every time and z is moved manually
    if axis_ind == 2:
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file_name, seq_args_string)
        axis_write_fn = pos.get_axis_write_fn(axis_ind)

    # Collect the counts
    counts = []
    camera.arm()
    for val in scan_vals:
        if tb.safe_stop():
            break
        if axis_ind in [0, 1]:
            seq_args[axis_ind + 2] = [val]
            seq_args_string = tb.encode_seq_args(seq_args)
            pulse_gen.stream_load(seq_file_name, seq_args_string)
        else:
            axis_write_fn(val)
        pulse_gen.stream_start(num_reps)
        img_array = camera.read()
        sample = widefield.counts_from_img_array(img_array, pixel_coords)
        counts.append(sample)
    camera.disarm()
    return np.array(counts, dtype=int)


def _optimize_on_axis(cxn, nv_sig, axis_ind, laser_key, fig=None):
    """Optimize on just one axis (0, 1, 2) for (x, y, z)"""

    ### Basic setup and definitions

    num_steps = 20
    config = common.get_config_dict()
    collection_mode = config["collection_mode"]
    config_positioning = config["Positioning"]
    pulse_gen = tb.get_server_pulse_gen(cxn)
    laser_dict = nv_sig[laser_key]
    laser_name = laser_dict["name"]
    readout = laser_dict["readout_dur"]
    laser_power = tb.set_laser_power(cxn, nv_sig, laser_key)

    # Axis-specific definitions
    config_axis_labels = {0: "xy", 1: "xy", 2: "z"}
    label = config_axis_labels[axis_ind]
    scan_range = config_positioning[f"{label}_optimize_range"]
    delay = config_positioning[f"{label}_delay"]
    control_mode = pos.get_axis_control_mode(axis_ind)

    # Get the center coordinates to optimize about
    coords = nv_sig["coords"]
    # The opti_offset flag allows a different NV at a specified offset to be used as a proxy for
    # optiimizing on the actual target NV. Useful if the real target NV is poorly isolated
    opti_offset = "opti_offset" in nv_sig and nv_sig["opti_offset"] is not None
    if opti_offset:
        coords += np.array(nv_sig["opti_offset"])
    axis_center = coords[axis_ind]
    scan_vals = pos.get_scan_1d(axis_center, scan_range, num_steps)

    # Position us at the starting point
    start_coords = np.copy(coords)
    start_coords[axis_ind] = np.min(scan_vals)
    pos.set_xyz(cxn, start_coords)

    ### Record the counts
    # How we conduct the scan depends on the config

    # Check if the xy positions are controlled by the sequence itself.
    # If so, then we need a specific sequence even for stepped/streamed z optimization.
    xy_control_mode = config_positioning["xy_control_mode"]
    xy_sequence_control = xy_control_mode == ControlMode.SEQUENCE
    if xy_sequence_control:
        if collection_mode == CollectionMode.CAMERA:
            counts = _read_counts_camera_sequence(
                cxn, axis_ind, scan_vals, nv_sig, laser_key
            )

    else:
        seq_file_name = "simple_readout.py"
        seq_args = [delay, readout, laser_name, laser_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file_name, seq_args_string)

        if collection_mode == CollectionMode.COUNTER:
            if control_mode == ControlMode.STEP:
                counts = _read_counts_counter_step(cxn, axis_ind, scan_vals)
            elif control_mode == ControlMode.STREAM:
                counts = _read_counts_counter_stream(cxn, axis_ind, scan_vals)

        elif collection_mode == CollectionMode.CAMERA:
            counts = _read_counts_camera_step(
                cxn, axis_ind, scan_vals, nv_sig, laser_key
            )

    ### Plot, fit, return

    count_format = config["count_format"]
    if count_format == CountFormat.RAW:
        f_counts = counts
    elif count_format == CountFormat.KCPS:
        f_counts = (counts / 1000) / (readout / 10**9)
    if fig is not None:
        _update_figure(fig, axis_ind, scan_vals, f_counts)
    opti_coord = _fit_gaussian(nv_sig, scan_vals, f_counts, axis_ind, fig)

    return opti_coord, scan_vals, f_counts


# endregion
# region General public functions


def stationary_count_lite(
    cxn,
    nv_sig,
    coords=None,
    laser_key=LaserKey.IMAGING,
    ret_img_array=False,
    scanning_drift_adjust=True,
    pixel_drift_adjust=True,
):
    # Set up
    config = common.get_config_dict()
    collection_mode = config["collection_mode"]
    pulse_gen = tb.get_server_pulse_gen(cxn)
    laser_dict = nv_sig[laser_key]
    laser_name = laser_dict["name"]
    readout = laser_dict["readout_dur"]
    num_reps = laser_dict["num_reps"] if "num_reps" in laser_dict else 1
    tb.set_filter(cxn, nv_sig, laser_key)
    laser_power = tb.set_laser_power(cxn, nv_sig, laser_key)
    if coords is None:
        # coords = nv_sig["coords"]
        coords = pos.get_nv_coords(nv_sig, laser_key, drift_adjust=False)
    # print(coords)
    # print(scanning_drift_adjust)
    if scanning_drift_adjust:
        coords = pos.adjust_coords_for_drift(coords)
    pos.set_xyz_on_nv(cxn, nv_sig)

    # Load the sequence
    xy_control_mode = pos.get_xy_control_mode(laser_name=laser_name)
    xy_sequence_control = xy_control_mode == ControlMode.SEQUENCE
    if xy_sequence_control:
        laser_pos_mode = config["Optics"][laser_name]["pos_mode"]
        if laser_pos_mode == LaserPosMode.SCANNING:
            seq_args = [readout, laser_name, [coords[0]], [coords[1]]]
            seq_file_name = "simple_readout-scanning.py"
        if laser_pos_mode == LaserPosMode.WIDEFIELD:
            seq_args = [readout, laser_name]
            seq_file_name = "simple_readout-widefield.py"
    else:
        delay = 0
        seq_args = [delay, readout, laser_name, laser_power]
        seq_file_name = "simple_readout.py"
    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file_name, seq_args_string)

    # Collect the data
    if collection_mode == CollectionMode.COUNTER:
        counter_server = tb.get_server_counter(cxn)
        counter_server.start_tag_stream()
        pulse_gen.stream_start(num_reps)
        new_samples = counter_server.read_counter_simple(num_reps)
        counter_server.stop_tag_stream()
    elif collection_mode == CollectionMode.CAMERA:
        pixel_coords = nv_sig["pixel_coords"]
        if pixel_drift_adjust:
            pixel_coords = widefield.adjust_pixel_coords_for_drift(pixel_coords)
        camera = tb.get_server_camera(cxn)
        camera.arm()
        new_samples = []
        pulse_gen.stream_start(num_reps)
        img_array = camera.read()
        camera.disarm()
        sample = widefield.counts_from_img_array(
            img_array, pixel_coords, drift_adjust=False
        )
        new_samples.append(sample)

    # Return
    avg_counts = np.average(new_samples)
    config = common.get_config_dict()
    count_format = config["count_format"]
    if ret_img_array:
        return img_array
    if count_format == CountFormat.RAW:
        return avg_counts
    elif count_format == CountFormat.KCPS:
        count_rate = (avg_counts / 1000) / (readout / 10**9)
        return count_rate


def prepare_microscope(cxn, nv_sig):
    """
    Prepares the microscope for a measurement. In particular,
    sets up the optics (positioning, collection filter, etc) and magnet,
    and sets the global coordinates. The laser set up must be handled by each routine

    If coords are not passed, the nv_sig coords (plus drift) will be used
    """

    pos.set_xyz_on_nv(cxn, nv_sig)

    if "collection_filter" in nv_sig:
        filter_name = nv_sig["collection_filter"]
        if filter_name is not None:
            tb.set_filter(cxn, optics_name="collection", filter_name=filter_name)

    magnet_angle = None if "magnet_angle" not in nv_sig else nv_sig["magnet_angle"]
    if magnet_angle is not None:
        rotation_stage_server = tb.get_server_magnet_rotation(cxn)
        rotation_stage_server.set_angle(magnet_angle)

    time.sleep(0.01)


def optimize_list(nv_sig_list):
    with common.labrad_connect() as cxn:
        optimize_list_with_cxn(cxn, nv_sig_list)


def optimize_list_with_cxn(cxn, nv_sig_list):
    tb.init_safe_stop()

    opti_coords_list = []
    current_counts_list = []
    for ind in range(len(nv_sig_list)):
        print("Optimizing on NV {}...".format(ind))

        if tb.safe_stop():
            break

        nv_sig = nv_sig_list[ind]
        opti_coords, current_counts = main_with_cxn(
            cxn, nv_sig, set_to_opti_coords=False, set_drift=False
        )

        if opti_coords is not None:
            opti_coords_list.append("[{:.3f}, {:.3f}, {:.2f}],".format(*opti_coords))
            current_counts_list.append("{},".format(current_counts))
        else:
            opti_coords_list.append("Optimization failed for NV {}.".format(ind))

    for coords in opti_coords_list:
        print(coords)


def main(
    nv_sig,
    set_to_opti_coords=True,
    save_data=False,
    plot_data=False,
    set_scanning_drift=True,
    set_pixel_drift=False,
    laser_key=LaserKey.IMAGING,
    drift_adjust=True,
    only_z_opt=None,
):
    with common.labrad_connect() as cxn:
        return main_with_cxn(
            cxn,
            nv_sig,
            set_to_opti_coords,
            save_data,
            plot_data,
            set_scanning_drift,
            set_pixel_drift,
            laser_key,
            drift_adjust,
            only_z_opt,
        )


def main_with_cxn(
    cxn,
    nv_sig,
    set_to_opti_coords=True,
    save_data=False,
    plot_data=False,
    set_scanning_drift=True,
    set_pixel_drift=True,
    laser_key=LaserKey.IMAGING,
    drift_adjust=True,
    only_z_opt=None,
):
    # If optimize is disabled, just do prep and return
    if nv_sig["disable_opt"]:
        prepare_microscope(cxn, nv_sig, adjusted_coords)
        return [], None

    ### Setup

    tb.reset_cfm(cxn)

    # Adjust the sig we use for drift

    passed_coords = nv_sig["coords"]
    if drift_adjust:
        adjusted_coords = pos.adjust_coords_for_drift(passed_coords)
    else:
        adjusted_coords = list(passed_coords)
    adjusted_nv_sig = copy.deepcopy(nv_sig)
    adjusted_nv_sig["coords"] = adjusted_coords

    # Define a few things
    config = common.get_config_dict()
    key = "expected_counts"
    expected_counts = adjusted_nv_sig[key] if key in adjusted_nv_sig else None
    if expected_counts is not None:
        lower_bound = 0.9 * expected_counts
        upper_bound = 1.2 * expected_counts

    start_time = time.time()
    tb.init_safe_stop()

    # Filter sets for imaging
    tb.set_filter(cxn, nv_sig, "collection")
    tb.set_filter(cxn, nv_sig, laser_key)

    # Default values for status variables
    opti_succeeded = False
    opti_necessary = True
    opti_coords = adjusted_coords.copy()

    def count_check(coords):
        return stationary_count_lite(
            cxn, nv_sig, coords, laser_key, scanning_drift_adjust=False
        )

    ### Check if we even need to optimize by reading counts at current coordinates

    count_format = config["count_format"]
    if count_format == CountFormat.RAW:
        print(f"Expected counts: {expected_counts}")
    elif count_format == CountFormat.KCPS:
        print(f"Expected count rate: {expected_counts} kcps")
    current_counts = count_check(adjusted_coords)
    print(f"Counts at initial coordinates: {current_counts}")
    if (expected_counts is not None) and (lower_bound < current_counts < upper_bound):
        print("No need to optimize.")
        opti_necessary = False

    ### Try to optimize.

    if opti_necessary:
        # Get which axes to optimize on
        if only_z_opt is None:
            only_z_opt = "only_z_opt" in nv_sig and nv_sig["only_z_opt"]
        disable_z_opt = "disable_z_opt" in nv_sig and nv_sig["disable_z_opt"]
        if only_z_opt:
            axes_to_optimize = [2]
        elif disable_z_opt:
            axes_to_optimize = [0, 1]
        else:
            axes_to_optimize = [0, 1, 2]

        # Loop through attempts until we succeed or give up
        num_attempts = 10
        for ind in range(num_attempts):
            ### Attempt setup

            if opti_succeeded or tb.safe_stop():
                break
            print(f"Attempt number {ind+1}")

            # Create a figure with a plot for each axis
            fig = _create_figure() if plot_data else None

            # Tracking lists for each axis
            opti_coords = adjusted_coords.copy()
            scan_vals_by_axis = [None] * 3
            counts_by_axis = [None] * 3

            ### Loop through the axes

            for axis_ind in axes_to_optimize:
                # Check if z optimization is necessary after xy optimization
                if axis_ind == 2 and axes_to_optimize == [0, 1, 2]:
                    test_coords = [*opti_coords[0:2], adjusted_coords[2]]
                    current_counts = count_check(test_coords)
                    if lower_bound < current_counts < upper_bound:
                        print("Z optimization unnecessary.")
                        scan_vals_by_axis.append(np.array([]))
                        opti_succeeded = True
                        break

                # Perform the optimization
                ret_vals = _optimize_on_axis(
                    cxn, adjusted_nv_sig, axis_ind, laser_key, fig
                )
                opti_coord = ret_vals[0]
                opti_coords[axis_ind] = opti_coord
                scan_vals_by_axis[axis_ind] = ret_vals[1]
                counts_by_axis[axis_ind] = ret_vals[2]
                # Update the coords on the nv_sig
                if opti_coord is not None:
                    adjusted_nv_sig["coords"][axis_ind] = opti_coord

            ### Attempt wrap-up

            # Try again if any individual axis failed
            if None in opti_coords:
                continue

            # Check the counts - if the threshold is not set, we just do one pass and succeed
            current_counts = count_check(opti_coords)
            print(f"Value at optimized coordinates: {round(current_counts, 1)}")
            if expected_counts is None:
                opti_succeeded = True
            else:
                if lower_bound < current_counts < upper_bound:
                    opti_succeeded = True
                else:
                    print("Value at optimized coordinates out of bounds.")

    ### Calculate the drift relative to the passed coordinates

    drift = (np.array(opti_coords) - np.array(passed_coords)).tolist()
    if opti_succeeded and set_scanning_drift:
        pos.set_drift(drift, nv_sig, laser_key)
    if opti_succeeded and set_pixel_drift:
        widefield.set_pixel_drift_from_scanning_drift(drift)

    ### Report the results and set to the optimized coordinates if requested

    if opti_succeeded:
        print("Optimization succeeded!")

    # Set to the coordinates and move on
    if set_to_opti_coords:
        if not opti_necessary or opti_succeeded:
            prepare_microscope(cxn, nv_sig, opti_coords)
        # Just crash if we failed and we were supposed to move to the optimized coordinates
        else:
            raise RuntimeError("Optimization failed.")
    # Or just report the results
    else:
        if not opti_necessary or opti_succeeded:
            print("Optimized coordinates: ")
            print("{:.3f}, {:.3f}, {:.2f}".format(*opti_coords))
            print("Drift: ")
            print("{:.3f}, {:.3f}, {:.2f}".format(*drift))
            prepare_microscope(cxn, nv_sig)
        else:
            print("Optimization failed.")
            prepare_microscope(cxn, nv_sig)

    print("\n")

    ### Clean up and save the data

    tb.reset_cfm(cxn)
    end_time = time.time()
    time_elapsed = end_time - start_time

    if save_data and opti_necessary:
        timestamp = tb.get_time_stamp()
        for ind in range(3):
            scan_vals = scan_vals_by_axis[ind]
            if scan_vals is not None:
                scan_vals = scan_vals.tolist()
            counts = counts_by_axis[ind]
            if counts is not None:
                counts = counts.tolist()
        rawData = {
            "timestamp": timestamp,
            "time_elapsed": time_elapsed,
            "nv_sig": nv_sig,
            "opti_coords": opti_coords,
            "axes_to_optimize": axes_to_optimize,
            "x_scan_vals": scan_vals_by_axis[0],
            "y_scan_vals": scan_vals_by_axis[1],
            "z_scan_vals": scan_vals_by_axis[2],
            "x_counts": counts_by_axis[0],
            "x_counts-units": "number",
            "y_counts": counts_by_axis[1],
            "y_counts-units": "number",
            "z_counts": counts_by_axis[2],
            "z_counts-units": "number",
        }

        filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
        if fig is not None:
            tb.save_figure(fig, filePath)
        tb.save_raw_data(rawData, filePath)

    # Return the optimized coordinates we found and the final counts
    return opti_coords, current_counts


# endregion

if __name__ == "__main__":
    file_name = "2023_09_21-21_07_51-widefield_calibration_nv1"
    data = tb.get_raw_data(file_name)

    fig = _create_figure()
    nv_sig = data["nv_sig"]
    keys = ["x", "y", "z"]
    for axis_ind in range(3):
        scan_vals = data[f"{keys[axis_ind]}_scan_vals"]
        count_vals = data[f"{keys[axis_ind]}_counts"]
        _update_figure(fig, axis_ind, scan_vals, count_vals)
        _fit_gaussian(nv_sig, scan_vals, count_vals, axis_ind, fig=fig)

    plt.show(block=True)
