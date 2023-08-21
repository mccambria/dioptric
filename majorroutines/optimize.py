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
import labrad
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import positioning
from utils import common
from utils import widefield
from utils.constants import ControlStyle, CountFormat, CollectionMode

# endregion
# region Plotting functions


def _create_figure():
    kpl.init_kplotlib()
    config = common.get_config_dict()
    fig, axes_pack = plt.subplots(1, 3, figsize=kpl.double_figsize)
    axis_titles = ["X Axis", "Y Axis", "Z Axis"]
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
# region Misc functions


def _read_counts(
    cxn, nv_sig, num_steps, period, control_style, axis_write_func, scan_vals
):
    config = common.get_config_dict()
    collection_mode = config["collection_mode"]
    if collection_mode == CollectionMode.CONFOCAL:
        fn = _read_counts_confocal
    if collection_mode == CollectionMode.WIDEFIELD:
        fn = _read_counts_widefield
    return fn(cxn, nv_sig, num_steps, period, control_style, axis_write_func, scan_vals)


def _read_counts_confocal(
    cxn, nv_sig, num_steps, period, control_style, axis_write_func, scan_vals
):
    counter = tb.get_server_counter(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)
    counter.start_tag_stream()

    counts = []
    timeout_duration = ((period * (10**-9)) * num_steps) + 10
    timeout_inst = time.time() + timeout_duration

    if control_style == ControlStyle.STREAM:
        num_read_so_far = 0
        pulse_gen.stream_start(num_steps)
        while num_read_so_far < num_steps:
            # Break if user says stop or timeout
            if tb.safe_stop() or time.time() > timeout_inst:
                break
            new_samples = counter.read_counter_simple()
            num_new_samples = len(new_samples)
            if num_new_samples > 0:
                counts.extend(new_samples)
                num_read_so_far += num_new_samples

    elif control_style == ControlStyle.STEP:
        for ind in range(len(scan_vals)):
            # Break if user says stop or timeout
            if tb.safe_stop() or time.time() > timeout_inst:
                break
            axis_write_func(scan_vals[ind])
            pulse_gen.stream_start(1)
            # Read the samples and update the image
            new_samples = counter.read_counter_simple(1)
            counts.extend(new_samples)

    counter.stop_tag_stream()

    return np.array(counts, dtype=int)


def _read_counts_widefield(
    cxn, nv_sig, num_steps, period, control_style, axis_write_func, scan_vals
):
    """Similar to confocal with step control_style"""

    pixel_coords = nv_sig["pixel_coords"]

    camera = tb.get_server_camera(cxn)
    pulse_gen = tb.get_server_pulse_gen(cxn)

    counts = []
    timeout_duration = ((period * (10**-9)) * num_steps) + 10
    timeout_inst = time.time() + timeout_duration

    camera.arm()
    for ind in range(len(scan_vals)):
        # Break if user says stop or timeout
        if tb.safe_stop() or time.time() > timeout_inst:
            break
        axis_write_func(scan_vals[ind])
        pulse_gen.stream_start(1)
        img_array = camera.read()
        sample = widefield.counts_from_img_array(img_array, pixel_coords)
        counts.extend(sample)

    camera.disarm()

    return np.array(counts, dtype=int)


def _stationary_count_lite(cxn, nv_sig, coords, laser_name=None):
    # Set up
    config = common.get_config_dict()
    collection_mode = config["collection_mode"]
    pulse_gen = tb.get_server_pulse_gen(cxn)
    if laser_name == None:
        laser_name = nv_sig["imaging_laser"]
        readout = nv_sig["imaging_readout_dur"]
    else:
        readout = nv_sig[f"readout-{laser_name}"]
    laser_power = tb.set_laser_power(cxn, nv_sig, laser_name=laser_name)
    num_samples = 1
    x_center, y_center, z_center = coords

    # Set coordinates
    positioning.set_xyz(cxn, [x_center, y_center, z_center])

    # Load the sequence
    config_positioning = config["Positioning"]
    delay = 0
    seq_args = [delay, readout, laser_name, laser_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    if collection_mode == CollectionMode.CONFOCAL:
        seq_file_name = "simple_readout.py"
    elif collection_mode == CollectionMode.WIDEFIELD:
        seq_file_name = "simple_readout-camera.py"
    pulse_gen.stream_load(seq_file_name, seq_args_string)

    # Collect the data
    if collection_mode == CollectionMode.CONFOCAL:
        counter_server = tb.get_server_counter(cxn)
        counter_server.start_tag_stream()
        pulse_gen.stream_start(num_samples)
        new_samples = counter_server.read_counter_simple(num_samples)
        counter_server.stop_tag_stream()
    elif collection_mode == CollectionMode.WIDEFIELD:
        pixel_coords = nv_sig["pixel_coords"]
        camera = tb.get_server_camera(cxn)
        camera.arm()
        new_samples = []
        for ind in range(num_samples):
            pulse_gen.stream_start(1)
            img_array = camera.read()
            sample = widefield.counts_from_img_array(img_array, pixel_coords)
            new_samples.extend(sample)
        camera.disarm()

    # Return
    avg_counts = np.average(new_samples)
    config = common.get_config_dict()
    count_format = config["count_format"]
    if count_format == CountFormat.RAW:
        return avg_counts
    elif count_format == CountFormat.KCPS:
        count_rate = (avg_counts / 1000) / (readout / 10**9)
        return count_rate


def _optimize_on_axis(cxn, nv_sig, axis_ind, laser_name=None, fig=None):
    """Optimize on just one axis (0, 1, 2) for (x, y, z)"""

    # Basic setup and definitions
    num_steps = 31
    config = common.get_config_dict()
    collection_mode = config["collection_mode"]
    config_positioning = config["Positioning"]
    pulse_gen = tb.get_server_pulse_gen(cxn)
    if collection_mode == CollectionMode.CONFOCAL:
        seq_file_name = "simple_readout.py"
    elif collection_mode == CollectionMode.WIDEFIELD:
        seq_file_name = "simple_readout-widefield.py"
    if laser_name == None:
        laser_name = nv_sig["imaging_laser"]
        readout = nv_sig["imaging_readout_dur"]
    else:
        readout = nv_sig[f"readout-{laser_name}"]
    laser_power = tb.set_laser_power(cxn, nv_sig, laser_name=laser_name)

    # This flag allows a different NV at a specified offset to be used as a proxy for
    # optiimizing on the actual target NV. Useful if, e.g., the target is poorly isolated
    coords = nv_sig["coords"]
    if "opti_offset" in nv_sig and nv_sig["opti_offset"] is not None:
        adj_coords = np.array(coords)
        opti_offset = np.array(nv_sig["opti_offset"])
        adj_coords += opti_offset
        axis_center = adj_coords[axis_ind]
        x_center, y_center, z_center = adj_coords
    else:
        axis_center = coords[axis_ind]
        x_center, y_center, z_center = coords

    # Axis-specific definitions
    config_axis_labels = {0: "xy", 1: "xy", 2: "z"}
    label = config_axis_labels[axis_ind]
    scan_range = config_positioning[f"{label}_optimize_range"]
    scan_dtype = config_positioning[f"{label}_dtype"]
    delay = config_positioning[f"{label}_delay"]
    control_style = config_positioning[f"{label}_control_style"]

    if axis_ind == 0:
        server = positioning.get_server_pos_xy(cxn)
        axis_write_func = server.write_x
        if control_style == ControlStyle.STREAM:
            load_stream = server.load_stream_xy
    if axis_ind == 1:
        server = positioning.get_server_pos_xy(cxn)
        axis_write_func = server.write_y
        if control_style == ControlStyle.STREAM:
            load_stream = server.load_stream_xy
    if axis_ind == 2:
        server = positioning.get_server_pos_z(cxn)
        axis_write_func = server.write_z
        if control_style == ControlStyle.STREAM:
            load_stream = server.load_stream_z

    # Move to first point in scan if we're in step mode
    if control_style == ControlStyle.STEP:
        half_scan_range = scan_range / 2
        lower = axis_center - half_scan_range
        if axis_ind == 0:
            start_coords = [lower, coords[1], coords[2]]
        elif axis_ind == 1:
            start_coords = [coords[0], lower, coords[2]]
        elif axis_ind == 2:
            start_coords = [coords[0], coords[1], lower]
        positioning.set_xyz(cxn, start_coords)

    # Sequence loading
    seq_args = [delay, readout, laser_name, laser_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    ret_vals = pulse_gen.stream_load(seq_file_name, seq_args_string)
    period = ret_vals[0]

    # Get the scan values
    if control_style == ControlStyle.STREAM:
        if axis_ind == 0:
            scan_vals, fixed_vals = positioning.get_scan_one_axis_2d(
                x_center, y_center, scan_range, num_steps
            )
        elif axis_ind == 1:
            scan_vals, fixed_vals = positioning.get_scan_one_axis_2d(
                y_center, x_center, scan_range, num_steps
            )
        elif axis_ind == 2:
            scan_vals = positioning.get_scan_1d(z_center, scan_range, num_steps)
        load_stream(scan_vals, fixed_vals)
    elif control_style == ControlStyle.STEP:
        scan_vals = positioning.get_scan_1d(axis_center, scan_range, num_steps)

    counts = _read_counts(
        cxn, nv_sig, num_steps, period, control_style, axis_write_func, scan_vals
    )

    # Plot and fit
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
# region User-callable functions


def prepare_microscope(cxn, nv_sig, coords=None):
    """
    Prepares the microscope for a measurement. In particular,
    sets up the optics (positioning, collection filter, etc) and magnet,
    and sets the coordinates. The laser set up must be handled by each routine

    If coords are not passed, the nv_sig coords (plus drift) will be used
    """

    if coords is None:
        coords = nv_sig["coords"]
        coords = positioning.adjust_coords_for_drift(coords, cxn)

    positioning.set_xyz(cxn, coords)

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
    with labrad.connect(username="", password="") as cxn:
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
    nv_sig, set_to_opti_coords=True, save_data=False, plot_data=False, set_drift=True
):
    with labrad.connect(username="", password="") as cxn:
        return main_with_cxn(
            cxn, nv_sig, set_to_opti_coords, save_data, plot_data, set_drift
        )


def main_with_cxn(
    cxn,
    nv_sig,
    set_to_opti_coords=True,
    save_data=False,
    plot_data=False,
    set_drift=True,
    laser_name=None,
):
    # If optimize is disabled, just do prep and return
    if nv_sig["disable_opt"]:
        prepare_microscope(cxn, nv_sig, adjusted_coords)
        return [], None

    ### Setup

    tb.reset_cfm(cxn)

    # Adjust the sig we use for drift
    passed_coords = nv_sig["coords"]
    adjusted_coords = positioning.adjust_coords_for_drift(passed_coords, cxn)
    adjusted_nv_sig = copy.deepcopy(nv_sig)
    adjusted_nv_sig["coords"] = adjusted_coords

    # Define a few things
    config = common.get_config_dict()
    key = "expected_counts"
    expected_counts = adjusted_nv_sig[key] if key in adjusted_nv_sig else None
    lower_bound = 0.9 * expected_counts
    upper_bound = 1.2 * expected_counts

    start_time = time.time()
    tb.init_safe_stop()

    # Default values for status variables
    opti_succeeded = False
    opti_necessary = True
    opti_coords = None

    # Filter sets for imaging
    tb.set_filter(cxn, nv_sig, "collection")
    if laser_name == None:
        tb.set_filter(cxn, nv_sig, "imaging_laser")

    ### Check if we even need to optimize by reading counts at current coordinates

    count_format = config["count_format"]
    if count_format == CountFormat.RAW:
        print(f"Expected counts: {expected_counts}")
    if count_format == CountFormat.KCPS:
        print(f"Expected count rate: {expected_counts} kcps")
    current_counts = _stationary_count_lite(cxn, nv_sig, adjusted_coords, laser_name)
    print(f"Counts at initial coordinates: {current_counts}")
    if (expected_counts is not None) and (lower_bound < current_counts < upper_bound):
        print("No need to optimize.")
        opti_necessary = False
        opti_coords = adjusted_coords

    ### Try to optimize.

    if opti_necessary:
        num_attempts = 10
        for ind in range(num_attempts):
            # Break out of the loop if we succeeded or user canceled
            if opti_succeeded or tb.safe_stop():
                break

            if ind > 0:
                print("Trying again...")

            # Create 3 plots in the figure, one for each axis
            fig = _create_figure() if plot_data else None

            # Tracking lists for each axis
            opti_coords = []
            scan_vals_by_axis = []
            counts_by_axis = []

            ### xy

            only_z_opt = "only_z_opt" in nv_sig and nv_sig["only_z_opt"]
            if only_z_opt:
                opti_coords = [adjusted_coords[0], adjusted_coords[1]]
                for i in range(2):
                    scan_vals_by_axis.append(np.array([]))
                    counts_by_axis.append(np.array([]))
            else:
                for axis_ind in range(2):
                    ret_vals = _optimize_on_axis(
                        cxn, adjusted_nv_sig, axis_ind, laser_name, fig
                    )
                    opti_coords.append(ret_vals[0])
                    scan_vals_by_axis.append(ret_vals[1])
                    counts_by_axis.append(ret_vals[2])
                # Check the counts before moving on to z, stop if xy optimization was sufficient
                if expected_counts is not None:
                    test_coords = [*opti_coords[0:2], adjusted_coords[2]]
                    current_counts = _stationary_count_lite(
                        cxn, nv_sig, test_coords, laser_name
                    )
                    if lower_bound < current_counts < upper_bound:
                        print("Z optimization unnecessary.")
                        scan_vals_by_axis.append(np.array([]))
                        opti_succeeded = True
                        break

            ### z

            disable_z_opt = "disable_z_opt" in nv_sig and nv_sig["disable_z_opt"]
            if disable_z_opt:
                opti_coords = [*opti_coords[0:2], adjusted_coords[2]]
                scan_vals_by_axis.append(np.array([]))
                counts_by_axis.append(np.array([]))
            else:
                # Help z out by ensuring we're centered in xy first
                if None not in opti_coords:
                    int_coords = [*opti_coords[0:2], adjusted_coords[2]]
                    positioning.set_xyz(cxn, int_coords)
                axis_ind = 2
                ret_vals = _optimize_on_axis(
                    cxn, adjusted_nv_sig, axis_ind, laser_name, fig
                )
                opti_coords.append(ret_vals[0])
                scan_vals_by_axis.append(ret_vals[1])
                counts_by_axis.append(ret_vals[2])

            ### Attempt wrap-up

            # Try again if any individual axis failed
            if None in opti_coords:
                continue

            # Check the counts
            current_counts = _stationary_count_lite(
                cxn, nv_sig, opti_coords, laser_name
            )
            print(f"Value at optimized coordinates: {round(current_counts, 1)}")
            if expected_counts is not None:
                if lower_bound < current_counts < upper_bound:
                    opti_succeeded = True
                else:
                    print("Value at optimized coordinates out of bounds.")
                    # Next pass use the coordinates we found this pass
                    adjusted_nv_sig["coords"] = opti_coords
            # If the threshold is not set, we just do one pass and succeed
            else:
                opti_succeeded = True

            # Reset opti_coords if the coords didn't cut it
            if not opti_succeeded:
                opti_coords = None

    if opti_succeeded:
        print("Optimization succeeded!")

    ### Calculate the drift relative to the passed coordinates

    if opti_succeeded and set_drift:
        drift = (np.array(opti_coords) - np.array(passed_coords)).tolist()
        positioning.set_drift(cxn, drift)

    ### Set to the optimized coordinates, or just tell the user what they are

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
        rawData = {
            "timestamp": timestamp,
            "time_elapsed": time_elapsed,
            "nv_sig": nv_sig,
            "opti_coords": opti_coords,
            "x_scan_vals": scan_vals_by_axis[0].tolist(),
            "y_scan_vals": scan_vals_by_axis[1].tolist(),
            "z_scan_vals": scan_vals_by_axis[2].tolist(),
            "x_counts": counts_by_axis[0].tolist(),
            "x_counts-units": "number",
            "y_counts": counts_by_axis[1].tolist(),
            "y_counts-units": "number",
            "z_counts": counts_by_axis[2].tolist(),
            "z_counts-units": "number",
        }

        filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
        if fig is not None:
            tb.save_figure(fig, filePath)
        tb.save_raw_data(rawData, filePath)

    # Return the optimized coordinates we found and the final counts
    return opti_coords, current_counts


# endregion
