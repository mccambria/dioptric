# -*- coding: utf-8 -*-
"""
Optimize on an NV. Cleaned-up/simplified version of the optimize major routine

Created on August 16th, 2023

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
from utils.constants import ControlStyle

# endregion
# region Plotting functions


def create_figure():
    kpl.init_kplotlib()
    config = common.get_config_dict()
    fig, axes_pack = plt.subplots(1, 3, figsize=(17, 8.5))
    axis_titles = ["X Axis", "Y Axis", "Z Axis"]
    for ind in range(3):
        ax = axes_pack[ind]
        ax.set_title(axis_titles[ind])
        xlabel_key = "xy_units" if ind < 2 else "z_units"
        xlabel = config["Positioning"][xlabel_key]
        ax.set_xlabel(xlabel)
        ylabel = "Count rate (kcps)" if config["count_rate"] else "Counts"
        ax.set_ylabel(ylabel)
    return fig


def update_figure(fig, axis_ind, scan_vals, count_vals, text=None):
    axes = fig.get_axes()
    ax = axes[axis_ind]
    ax.plot(scan_vals, count_vals)
    if text is not None:
        kpl.anchored_text(ax, text, kpl.Loc.UPPER_RIGHT)
    kpl.flush_update(fig=fig)


def fit_gaussian(nv_sig, scan_vals, count_vals, axis_ind, fig=None):
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
        linspace_voltages = np.linspace(low, high, num=1000)
        fit_count_rates = fit_func(linspace_voltages, *popt)
        # Add popt to the axes
        text = "a={:.3f}\n $\mu$={:.3f}\n $\sigma$={:.3f}\n offset={:.3f}".format(*popt)
        update_figure(fig, axis_ind, linspace_voltages, fit_count_rates, text)

    center = None
    if popt is not None:
        center = popt[1]

    return center


# endregion
# region Misc functions


def read_counts(
    cxn, num_steps, period, control_style, axis_write_func=None, scan_vals=None
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


def stationary_count_lite(cxn, nv_sig, coords):
    # Set up
    config = common.get_config_dict()
    counter_server = tb.get_server_counter(cxn)
    pulsegen_server = tb.get_server_pulse_gen(cxn)
    seq_file_name = "simple_readout.py"
    laser_name = nv_sig["imaging_laser"]
    laser_power = tb.set_laser_power(cxn, nv_sig, "imaging_laser")
    readout = nv_sig["imaging_readout_dur"]
    num_samples = 2
    x_center, y_center, z_center = coords

    # Set coordinates
    ramp = "set_xyz_ramp" in config and config["set_xyz_ramp"]
    positioning.set_xyz(cxn, [x_center, y_center, z_center], ramp)

    # Load the sequence
    config_positioning = config["Positioning"]
    delay = config_positioning["xy_delay"]
    seq_args = [delay, readout, laser_name, laser_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    pulsegen_server.stream_load(seq_file_name, seq_args_string)

    # Collect the data
    counter_server.start_tag_stream()
    pulsegen_server.stream_start(num_samples)
    new_samples = counter_server.read_counter_simple(num_samples)
    counter_server.stop_tag_stream()

    # Return
    avg_counts = np.average(new_samples)
    config = common.get_config_dict()
    if config["count_rate"]:
        count_rate = (avg_counts / 1000) / (readout / 10**9)
        return count_rate
    else:
        return avg_counts


# endregion
# region User-callable functions


def prepare_microscope(cxn, nv_sig, coords=None):
    """
    Prepares the microscope for a measurement. In particular,
    sets up the optics (positioning, collection filter, etc) and magnet,
    and sets the coordinates. The laser set up must be handled by each routine

    If coords are not passed, the nv_sig coords (plus drift) will be used
    """

    config = common.get_config_dict()
    ramp = "set_xyz_ramp" in config and config["set_xyz_ramp"]

    if coords is None:
        coords = nv_sig["coords"]
        coords = positioning.adjust_coords_for_drift(coords, cxn)

    positioning.set_xyz(cxn, coords, ramp)

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
    opti_counts_list = []
    for ind in range(len(nv_sig_list)):
        print("Optimizing on NV {}...".format(ind))

        if tb.safe_stop():
            break

        nv_sig = nv_sig_list[ind]
        opti_coords, opti_counts = main_with_cxn(
            cxn, nv_sig, set_to_opti_coords=False, set_drift=False
        )

        if opti_coords is not None:
            opti_coords_list.append("[{:.3f}, {:.3f}, {:.2f}],".format(*opti_coords))
            opti_counts_list.append("{},".format(opti_counts))
        else:
            opti_coords_list.append("Optimization failed for NV {}.".format(ind))

    for coords in opti_coords_list:
        print(coords)


def optimize_on_axis(cxn, nv_sig, axis_ind, config, fig=None):
    """Optimize on just one axis (0, 1, 2) for (x, y, z)"""
    xy_control_style = positioning.get_xy_control_style()
    z_control_style = positioning.get_z_control_style()
    control_style = xy_control_style if axis_ind < 2 else z_control_style
    if axis_ind == 0:
        axis_write_func = xy_server.write_x
    if axis_ind == 1:
        axis_write_func = xy_server.write_y
    if axis_ind == 2:
        axis_write_func = z_server.write_z

    num_steps = 31

    pulsegen_server = tb.get_server_pulse_gen(cxn)

    seq_file_name = "simple_readout.py"

    coords = nv_sig["coords"]

    x_center, y_center, z_center = coords

    if "opti_offset" in nv_sig:
        adj_coords = np.array(coords)
        opti_offset = np.array(nv_sig["opti_offset"])
        adj_coords += opti_offset
        sweep_x_center, sweep_y_center, sweep_z_center = adj_coords
    else:
        sweep_x_center, sweep_y_center, sweep_z_center = coords

    readout = nv_sig["imaging_readout_dur"]
    laser_key = "imaging_laser"
    laser_name = nv_sig[laser_key]
    laser_power = tb.set_laser_power(cxn, nv_sig, laser_key)

    # xy
    if axis_ind in [0, 1]:
        xy_server = positioning.get_server_pos_xy(cxn)

        config_positioning = config["Positioning"]
        scan_range = config_positioning["xy_optimize_range"]
        scan_dtype = config_positioning["xy_dtype"]
        if "xy_small_response_delay" in config_positioning:
            delay = config["Positioning"]["xy_small_response_delay"]
        else:
            delay = config["Positioning"]["xy_delay"]

        if xy_control_style == ControlStyle.STEP:
            # Move to first point in scan
            half_scan_range = scan_range / 2
            x_low = sweep_x_center - half_scan_range
            y_low = sweep_y_center - half_scan_range
            if axis_ind == 0:
                start_coords = [x_low, coords[1], coords[2]]
            elif axis_ind == 1:
                start_coords = [coords[0], y_low, coords[2]]

            if nv_sig["ramp_voltages"] == True:
                positioning.set_xyz_ramp(cxn, start_coords)
            else:
                positioning.set_xyz(cxn, start_coords)
            auto_scan = False

        elif xy_control_style == ControlStyle.STREAM:
            # no need to move to first position. loading the daq already does that
            auto_scan = True

        seq_args = [delay, readout, laser_name, laser_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        ret_vals = pulsegen_server.stream_load(seq_file_name, seq_args_string)
        period = ret_vals[0]

        if axis_ind == 0:
            if auto_scan:
                scan_func = xy_server.load_stream_xy
                scan_vals, fixed_vals = positioning.get_scan_one_axis_2d(
                    sweep_x_center, sweep_y_center, scan_range, num_steps
                )
                scan_func(scan_vals, fixed_vals)
            else:
                axis_write_func = xy_server.write_x
                scan_vals = positioning.get_scan_1d(
                    sweep_x_center, scan_range, num_steps
                )

        elif axis_ind == 1:
            if auto_scan:
                scan_func = xy_server.load_stream_xy
                scan_vals, fixed_vals = positioning.get_scan_one_axis_2d(
                    sweep_y_center, sweep_x_center, scan_range, num_steps
                )
                scan_func(fixed_vals, scan_vals)
            else:
                axis_write_func = xy_server.write_y
                scan_vals = positioning.get_scan_1d(
                    sweep_y_center, scan_range, num_steps
                )

    # z
    elif axis_ind == 2:
        scan_range = config["Positioning"]["z_optimize_range"]
        scan_dtype = config["Positioning"]["z_dtype"]
        delay = config["Positioning"]["z_delay"]

        control_style = z_control_style
        if z_control_style == ControlStyle.STEP:
            auto_scan = False
        elif z_control_style == ControlStyle.STREAM:
            # no need to move to first position. loading the daq already does that
            auto_scan = True

        # Move to first point in scan
        half_scan_range = scan_range / 2
        z_low = sweep_z_center - half_scan_range
        start_coords = [x_center, y_center, z_low]
        if "ramp_voltages" in nv_sig and nv_sig["ramp_voltages"]:
            positioning.set_xyz_ramp(cxn, start_coords)
        else:
            positioning.set_xyz(cxn, start_coords)

        z_server = positioning.get_server_pos_z(cxn)

        seq_args = [delay, readout, laser_name, laser_power]
        seq_args_string = tb.encode_seq_args(seq_args)
        ret_vals = pulsegen_server.stream_load(seq_file_name, seq_args_string)
        period = ret_vals[0]

        if auto_scan:
            scan_func = z_server.load_stream_z
            scan_vals = positioning.get_scan_1d(sweep_z_center, scan_range, num_steps)
            scan_func(scan_vals)

        else:
            axis_write_func = z_server.write_z
            scan_vals = positioning.get_scan_1d(sweep_z_center, scan_range, num_steps)

    counts = read_counts(
        cxn, num_steps, period, control_style, axis_write_func, scan_vals
    )
    count_rates = (counts / 1000) / (readout / 10**9)

    if fig is not None:
        update_figure(fig, axis_ind, scan_vals, count_rates)

    opti_coord = fit_gaussian(nv_sig, scan_vals, count_rates, axis_ind, fig)

    return opti_coord, scan_vals, counts


# endregion


# region Main
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
    xy_control_style = positioning.get_xy_control_style()
    z_control_style = positioning.get_z_control_style()
    start_time = time.time()
    tb.init_safe_stop()
    expected_counts = (
        None
        if "expected_counts" not in adjusted_nv_sig
        else adjusted_nv_sig["expected_counts"]
    )
    config = common.get_config_dict()

    # Default values for status variables
    opti_succeeded = False
    opti_necessary = True
    opti_coords = None

    # Filter sets for imaging
    tb.set_filter(cxn, nv_sig, "collection")
    tb.set_filter(cxn, nv_sig, "imaging_laser")

    ### Check if we even need to optimize by reading counts at current coordinates

    print(f"Expected count rate: {expected_counts}")

    if expected_counts is not None:
        lower_threshold = expected_counts * 9 / 10
        upper_threshold = expected_counts * 6 / 5

    # Check the count rate
    opti_counts = stationary_count_lite(cxn, nv_sig, adjusted_coords)

    print(f"Counts at optimized coordinates: {opti_counts}")

    # If the count rate close to what we expect, we succeeded!
    if expected_counts is not None:
        if lower_threshold <= opti_counts <= upper_threshold:
            print("No need to optimize.")
            opti_necessary = False
            opti_coords = adjusted_coords
        else:
            print("Count rate at optimized coordinates out of bounds.")

    ### Try to optimize.

    if opti_necessary:
        if xy_control_style == ControlStyle.STREAM:
            num_attempts = 20
        elif xy_control_style == ControlStyle.STEP:
            num_attempts = 4

        for ind in range(num_attempts):
            # Break out of the loop if we succeeded or user canceled
            if opti_succeeded or tb.safe_stop():
                break

            if ind > 0:
                print("Trying again...")

            # Create 3 plots in the figure, one for each axis
            fig = create_figure() if plot_data else None

            # Optimize on each axis
            opti_coords = []
            scan_vals_by_axis = []
            counts_by_axis = []

            # X, Y
            only_z_opt = "only_z_opt" in nv_sig and nv_sig["only_z_opt"]
            if only_z_opt:
                opti_coords = [adjusted_coords[0], adjusted_coords[1]]
                for i in range(2):
                    scan_vals_by_axis.append(np.array([]))
                    counts_by_axis.append(np.array([]))
            else:
                for axis_ind in range(2):
                    ret_vals = optimize_on_axis(
                        cxn, adjusted_nv_sig, axis_ind, config, fig
                    )
                    opti_coords.append(ret_vals[0])
                    scan_vals_by_axis.append(ret_vals[1])
                    counts_by_axis.append(ret_vals[2])
                # Check the count rate before moving on to z, stop if xy optimization was sufficient
                if expected_counts is not None:
                    if z_control_style == ControlStyle.STREAM:
                        test_coords = [*opti_coords[0:2], adjusted_coords[2]]
                        opti_counts = stationary_count_lite(cxn, nv_sig, test_coords)
                        r_opti_counts = round(opti_counts, 1)
                        if lower_threshold <= opti_counts <= upper_threshold:
                            opti_coords = test_coords
                            print("Z optimization unnecessary.")
                            print(
                                f"Count rate at optimized coordinates: {r_opti_counts}"
                            )
                            print("Optimization succeeded!")
                            opti_succeeded = True
                            break
                    elif z_control_style == ControlStyle.STEP:
                        pass  # Not implemented yet

            # Z
            if z_control_style == ControlStyle.STREAM:
                disable_z_opt = "disable_z_opt" in nv_sig and nv_sig["disable_z_opt"]
                if disable_z_opt:
                    opti_coords = [*opti_coords[0:2], adjusted_coords[2]]
                    scan_vals_by_axis.append(np.array([]))
                    counts_by_axis.append(np.array([]))
                else:
                    # Help z out by ensuring we're centered in xy first
                    if None not in opti_coords:
                        int_coords = [
                            opti_coords[0],
                            opti_coords[1],
                            adjusted_coords[2],
                        ]
                        positioning.set_xyz(cxn, int_coords)
                    axis_ind = 2
                    ret_vals = optimize_on_axis(
                        cxn, adjusted_nv_sig, axis_ind, config, fig
                    )
                    opti_coords.append(ret_vals[0])
                    scan_vals_by_axis.append(ret_vals[1])
                    counts_by_axis.append(ret_vals[2])

            elif z_control_style == ControlStyle.STEP:
                if None not in opti_coords:
                    int_coords = [opti_coords[0], opti_coords[1], adjusted_coords[2]]
                    adjusted_nv_sig_z = copy.deepcopy(nv_sig)
                    adjusted_nv_sig_z["coords"] = int_coords
                else:
                    adjusted_nv_sig_z = copy.deepcopy(nv_sig)
                    adjusted_nv_sig_z["coords"] = adjusted_coords
                axis_ind = 2
                ret_vals = optimize_on_axis(
                    cxn, adjusted_nv_sig_z, axis_ind, config, fig
                )
                opti_coords.append(ret_vals[0])
                scan_vals_by_axis.append(ret_vals[1])
                counts_by_axis.append(ret_vals[2])

            # We failed to get optimized coordinates, try again
            if None in opti_coords:
                continue

            # Check the count rate
            opti_counts = stationary_count_lite(cxn, nv_sig, opti_coords)

            # Verify that our optimization found a reasonable spot by checking
            # the count rate at the center against the expected count rate
            if expected_counts is not None:
                print("Count rate at optimized coordinates: {:.1f}".format(opti_counts))

                # If the count rate close to what we expect, we succeeded!
                if lower_threshold <= opti_counts <= upper_threshold:
                    print("Optimization succeeded!")
                    opti_succeeded = True
                else:
                    print("Count rate at optimized coordinates out of bounds.")
                    # If we failed by expected counts, try again with the
                    # coordinates we found. If x/y are off initially, then
                    # z will give a false optimized coordinate. x/y will give
                    # true optimized coordinates regardless of the other initial
                    # coordinates, however. So we might succeed by trying z again
                    # at the optimized x/y.
                    adjusted_nv_sig["coords"] = opti_coords

            # If the threshold is not set, we succeed based only on optimize
            else:
                print("Count rate at optimized coordinates: {:.1f}".format(opti_counts))
                print("Optimization succeeded! (No expected count rate passed.)")
                opti_succeeded = True

            # Reset opti_coords if the coords didn't cut it
            if not opti_succeeded:
                opti_coords = None

    ### Calculate the drift relative to the passed coordinates

    if opti_succeeded and set_drift:
        drift = (np.array(opti_coords) - np.array(passed_coords)).tolist()
        positioning.set_drift(cxn, drift)

    ### Set to the optimized coordinates, or just tell the user what they are

    if set_to_opti_coords:
        if opti_succeeded or opti_necessary:
            prepare_microscope(cxn, nv_sig, opti_coords)
        else:
            msg = "Optimization failed."
            # Just crash
            raise RuntimeError(msg)
            # Let the user know something went wrong
            # msg = ("Optimization failed. Resetting to coordinates "
            #        "about which we attempted to optimize.")
            # print(
            #     "Optimization failed. Resetting to coordinates "
            #     "about which we attempted to optimize."
            # )
            # prepare_microscope(cxn, nv_sig, adjusted_coords)
    else:
        if opti_succeeded or opti_necessary:
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

    # Don't bother saving the data if we're just using this to find the
    # optimized coordinates
    if save_data and not opti_necessary:
        if len(scan_vals_by_axis) < 3:
            z_scan_vals = None
        else:
            z_scan_vals = scan_vals_by_axis[2].tolist()

        timestamp = tb.get_time_stamp()
        rawData = {
            "timestamp": timestamp,
            "time_elapsed": time_elapsed,
            "nv_sig": nv_sig,
            "nv_sig-units": tb.get_nv_sig_units(cxn),
            "opti_coords": opti_coords,
            "x_scan_vals": scan_vals_by_axis[0].tolist(),
            "y_scan_vals": scan_vals_by_axis[1].tolist(),
            "z_scan_vals": z_scan_vals,
            "x_counts": counts_by_axis[0].tolist(),
            "x_counts-units": "number",
            "y_counts": counts_by_axis[1].tolist(),
            "y_counts-units": "number",
            "z_counts": z_scan_vals,
            "z_counts-units": "number",
            "xy_control_type": xy_control_style.name,
            "z_control_type": z_control_style.name,
        }

        filePath = tb.get_file_path(__file__, timestamp, nv_sig["name"])
        if fig is not None:
            tb.save_figure(fig, filePath)
        tb.save_raw_data(rawData, filePath)

    # Return the optimized coordinates we found
    return opti_coords, opti_counts


# endregion
