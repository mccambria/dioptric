# -*- coding: utf-8 -*-
"""Various functions for positioning microscope focus. Includes
functions for generating lists of coordinates used in scans

Created on Decemeber 1st, 2022

@author: mccambria
"""

# region Imports and constants

import time

import numpy as np

from utils import common
from utils import tool_belt as tb
from utils.constants import ControlMode, CoordsKey, NVSig

# endregion
# region Simple sets
"""
If a specific laser is not passed, then the set will just use the global
coords (nv_sig key "coords"). Otherwise we'll use the laser specific coords
(nv_sig key f"coords-{coords_key}")
"""


def set_xyz(coords, coords_key=CoordsKey.GLOBAL, drift_adjust=False, ramp=None):
    if drift_adjust:
        coords = adjust_coords_for_drift(coords, coords_key=coords_key)
    if ramp is None:
        config = common.get_config_dict()
        key = "set_xyz_ramp"
        ramp = key in config and config[key]
    if ramp:
        return _set_xyz_ramp(coords)
    else:
        return _set_xyz(coords, coords_key)


def _set_xyz(coords, coords_key):
    xy_dtype = get_xy_dtype(coords_key=coords_key)
    z_dtype = get_z_dtype(coords_key=coords_key)
    pos_xy_server = get_server_pos_xy(coords_key=coords_key)
    pos_z_server = get_server_pos_z(coords_key=coords_key)
    if pos_xy_server is not None:
        pos_xy_server.write_xy(xy_dtype(coords[0]), xy_dtype(coords[1]))
    if pos_z_server is not None:
        pos_z_server.write_z(z_dtype(coords[2]))
    # # Force some delay before proceeding to account for the effective write time
    # time.sleep(0.002)


def _set_xyz_ramp(coords):
    """Not up to date: Step incrementally to this position from the current position"""

    config = common.get_config_dict()
    config_positioning = config["Positioning"]

    xy_dtype = config_positioning["xy_dtype"]
    z_dtype = config_positioning["z_dtype"]

    step_size_xy = config_positioning["xy_incremental_step_size"]
    step_size_z = config_positioning["z_incremental_step_size"]

    xy_delay = config_positioning["xy_delay"]
    z_delay = config_positioning["z_delay"]

    # Take whichever one is longer
    if xy_delay > z_delay:
        total_movement_delay = xy_delay
    else:
        total_movement_delay = z_delay

    xyz_server = get_server_pos_xyz()
    pulse_gen = tb.get_server_pulse_gen()

    # if the movement type is int, just skip this and move to the desired position
    if xy_dtype is int or z_dtype is int:
        set_xyz(coords)
        return

    # Get current and final position
    current_x, current_y = xyz_server.read_xy()
    current_z = xyz_server.read_z()
    final_x, final_y, final_z = coords

    dx = final_x - current_x
    dy = final_y - current_y
    dz = final_z - current_z
    # print('dx: {}'.format(dx))
    # print('dy: {}'.format(dy))

    # If we are moving a distance smaller than the step size,
    # just set the coords, don't try to run a sequence

    if abs(dx) <= step_size_xy and abs(dy) <= step_size_xy and abs(dz) <= step_size_z:
        # print('just setting coords without ramp')
        set_xyz(coords)

    else:
        # Determine num of steps to get to final destination based on step size
        num_steps_x = np.ceil(abs(dx) / step_size_xy)
        num_steps_y = np.ceil(abs(dy) / step_size_xy)
        num_steps_z = np.ceil(abs(dz) / step_size_z)

        # Determine max steps for this move
        max_steps = int(max([num_steps_x, num_steps_y, num_steps_z]))

        # The delay between steps will be the total delay divided by the num of incr steps
        movement_delay = int(total_movement_delay / max_steps)

        x_points = [current_x]
        y_points = [current_y]
        z_points = [current_z]

        # set up the voltages to step thru. Once x, y, or z reach their final
        # value, just pass the final position for the remaining steps
        for n in range(max_steps):
            if n > num_steps_x - 1:
                x_points.append(final_x)
            else:
                move_x = (n + 1) * step_size_xy * dx / abs(dx)
                incr_x_val = move_x + current_x
                x_points.append(incr_x_val)

            if n > num_steps_y - 1:
                y_points.append(final_y)
            else:
                move_y = (n + 1) * step_size_xy * dy / abs(dy)
                incr_y_val = move_y + current_y
                y_points.append(incr_y_val)

            if n > num_steps_z - 1:
                z_points.append(final_z)
            else:
                move_z = (n + 1) * step_size_z * dz / abs(dz)
                incr_z_val = move_z + current_z
                z_points.append(incr_z_val)
        # Run a simple clock pulse repeatedly to move through votlages
        file_name = "simple_clock.py"
        seq_args = [movement_delay]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(file_name, seq_args_string)
        # period = ret_vals[0]
        # print(z_points)

        xyz_server.load_stream_xyz(x_points, y_points, z_points)
        pulse_gen.stream_load(file_name, seq_args_string)
        pulse_gen.stream_start(max_steps)

    # Force some delay before proceeding to account
    # for the effective write time, as well as settling time for movement
    time.sleep(total_movement_delay / 1e9)


def set_xyz_on_nv(nv_sig, coords_key=CoordsKey.GLOBAL, drift_adjust=True):
    """Returns the coords actually used in the set"""
    coords = get_nv_coords(nv_sig, coords_key, drift_adjust)
    set_xyz(coords, coords_key=coords_key, drift_adjust=False)
    return coords


def get_nv_coords(
    nv_sig: NVSig, coords_key=CoordsKey.GLOBAL, drift_adjust=True, drift=None
):
    coords = nv_sig.coords
    if isinstance(coords, dict):
        coords = coords[coords_key]
    if drift_adjust:
        coords = adjust_coords_for_drift(
            coords=coords, drift=drift, coords_key=coords_key
        )
    return coords


def set_nv_coords(nv_sig, coords, coords_key=CoordsKey.GLOBAL):
    coords = nv_sig.coords
    if isinstance(coords, list):
        nv_sig.coords = coords
    if isinstance(coords, dict):
        nv_sig.coords[coords_key] = coords


# endregion
# region Getters


def get_laser_pos_mode(laser_name):
    config = common.get_config_dict()
    config_laser = config["Optics"][laser_name]
    if "pos_mode" in config_laser:
        return config_laser["pos_mode"]
    else:
        return None


def _get_axis_value_sub(base_key, axis_ind, coords_key=CoordsKey.GLOBAL):
    label_dict = {0: "xy", 1: "xy", 2: "z"}
    label = label_dict[axis_ind]
    key = f"{label}_{base_key}"
    config = common.get_config_dict()
    config_pos = config["Positioning"]
    if coords_key in config_pos:
        return config_pos[coords_key][key]
    else:
        return config_pos[key]


def get_axis_delay(axis_ind, coords_key=CoordsKey.GLOBAL):
    return _get_axis_value_sub("delay", axis_ind, coords_key)


def get_axis_units(axis_ind, coords_key=CoordsKey.GLOBAL):
    return _get_axis_value_sub("units", axis_ind, coords_key)


def get_axis_control_mode(axis_ind, coords_key=CoordsKey.GLOBAL):
    return _get_axis_value_sub("control_mode", axis_ind, coords_key)


def get_axis_optimize_range(axis_ind, coords_key=CoordsKey.GLOBAL):
    return _get_axis_value_sub("optimize_range", axis_ind, coords_key)


def get_axis_dtype(axis_ind, coords_key=CoordsKey.GLOBAL):
    return _get_axis_value_sub("dtype", axis_ind, coords_key)


def get_xy_dtype(coords_key=CoordsKey.GLOBAL):
    axis_ind = 0
    return get_axis_dtype(axis_ind, coords_key)


def get_z_dtype(coords_key=CoordsKey.GLOBAL):
    axis_ind = 2
    return get_axis_dtype(axis_ind, coords_key)


def get_xy_control_mode(coords_key=CoordsKey.GLOBAL):
    axis_ind = 0
    return get_axis_control_mode(axis_ind, coords_key)


def get_z_control_mode(coords_key=CoordsKey.GLOBAL):
    axis_ind = 2
    return get_axis_control_mode(axis_ind, coords_key)


def _get_positioning_server(base_key, coords_key):
    try:
        return common.get_server(f"{base_key}-{coords_key}")
    except Exception:
        return common.get_server(base_key)


def get_server_pos_xy(coords_key=CoordsKey.GLOBAL):
    return _get_positioning_server("pos_xy", coords_key)


def get_server_pos_z(coords_key=CoordsKey.GLOBAL):
    return _get_positioning_server("pos_z", coords_key)


def get_server_pos_xyz(coords_key=CoordsKey.GLOBAL):
    return _get_positioning_server("pos_xyz", coords_key)


def get_axis_write_fn(axis_ind, coords_key=CoordsKey.GLOBAL):
    """Return the write function for a given axis (0:x, 1:y, 2:z)"""
    if axis_ind in [0, 1]:
        server = get_server_pos_xy(coords_key)
    elif axis_ind == 2:
        server = get_server_pos_z(coords_key)
    if server is None:
        return None

    if axis_ind == 0:
        write_fn = server.write_x
    if axis_ind == 1:
        write_fn = server.write_y
    if axis_ind == 2:
        write_fn = server.write_z

    return write_fn


def get_axis_stream_fn(axis_ind, coords_key=CoordsKey.GLOBAL):
    """Return the stream function for a given axis (0:x, 1:y, 2:z)"""
    control_mode = get_axis_control_mode(axis_ind)
    if control_mode != ControlMode.STREAM:
        return None

    if axis_ind in [0, 1]:
        server = get_server_pos_xy(coords_key)
    elif axis_ind == 2:
        server = get_server_pos_z(coords_key)
    if server is None:
        return None

    if axis_ind == 0:
        stream_fn = server.load_stream_x
    if axis_ind == 1:
        stream_fn = server.load_stream_y
    if axis_ind == 2:
        stream_fn = server.load_stream_z

    return stream_fn


# endregion
# region Drift
"""Implemented with a drift tracking global stored on the registry"""


def _get_drift_key(coords_key=CoordsKey.GLOBAL):
    try:
        return common.get_server(f"drift-{coords_key}")
    except Exception:
        return common.get_server("drift")


def get_drift(coords_key=CoordsKey.GLOBAL):
    key = _get_drift_key(coords_key)
    drift = common.get_registry_entry(["State"], key)
    drift_dtype = []
    for ind in range(len(drift)):
        axis_dtype = get_axis_dtype(ind, coords_key)
        if axis_dtype is not None:
            drift_dtype.append(axis_dtype(drift[ind]))
        else:
            drift_dtype.append(None)
    return np.array(drift_dtype)


def set_drift(drift, coords_key=CoordsKey.GLOBAL):
    key = _get_drift_key(coords_key)
    return common.set_registry_entry(["State"], key, drift)


def reset_drift(coords_key=CoordsKey.GLOBAL):
    try:
        drift = get_drift(coords_key)
        len_drift = len(drift)
    except Exception:
        len_drift = 3
    return set_drift([0.0] * len_drift, coords_key)


def reset_xy_drift(coords_key=CoordsKey.GLOBAL):
    drift = get_drift(coords_key)
    if len(drift) == 2:
        return set_drift([0.0, 0.0], coords_key)
    else:
        return set_drift([0.0, 0.0, drift[2]], coords_key)


def adjust_coords_for_drift(
    coords=None, drift=None, nv_sig=None, coords_key=CoordsKey.GLOBAL
):
    """Current drift will be retrieved from registry if passed drift is None"""
    if coords is None:
        coords = get_nv_coords(nv_sig, coords_key, drift_adjust=False)
    if drift is None:
        drift = get_drift(coords_key)
    adjusted_coords = []
    for ind in range(len(coords)):
        coords_val = coords[ind]
        drift_val = drift[ind]
        if coords_val is not None and drift_val is not None:
            adj_val = coords_val + drift_val
        else:
            adj_val = None
        adjusted_coords.append(adj_val)
    return adjusted_coords


# endregion
# region Scans
"""These are really just calculator functions for generating lists of coordinates
used in scans. Since the functions don't care about whether your scan is xy or xz
or whatever, variables are named axis-agnostically as <var>_<axis_ind>
"""


def get_scan_1d(center, scan_range, num_steps, dtype=np.float64):
    """Get a linear spacing of coords about the passed center

    Parameters
    ----------
    center : numeric
        Center of the scan along the axis
    range : numeric
        Full range of the scan along the first axis
    num_steps : _type_
        Number of steps in the scan

    Returns
    -------
    array(numeric)
        Scan coords
    """

    half_range = scan_range / 2
    low = center - half_range
    high = center + half_range
    coords = np.linspace(low, high, num_steps, dtype=dtype)
    return coords


# load_sweep_scan_xy
def get_scan_grid_2d(
    center_1,
    center_2,
    scan_range_1,
    scan_range_2,
    num_steps_1,
    num_steps_2,
    dtype=np.float64,
):
    """Create a grid of points for a snake scan

    Parameters
    ----------
    center_1 : numeric
        Center of the scan along the first axis
    center_2 : numeric
        Center of the scan along the second axis
    scan_range_1 : numeric
        Full range of the scan along the first axis
    scan_range_2 : numeric
        Full range of the scan along the second axis
    num_steps_1 : int
        Number of steps along the first axis
    num_steps_2 : int
        Number of steps along the second axis

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the snake scan
        e.g.
    array(numeric)
        Values to write to the second axis for the snake scan
    array(numeric)
        First-axis coordinates (i.e. coordinates scanned through if second axis were fixed)
    array(numeric)
        Second-axis coordinates (i.e. coordinates scanned through if first axis were fixed)
    list(float)
        Extent of the grid in the form [left, right, bottom, top] - includes half-pixel adjusment to
        min/max written vals for each axis so that the pixels in an image are properly centered
    """

    coords_1_1d = get_scan_1d(center_1, scan_range_1, num_steps_1, dtype)
    coords_2_1d = get_scan_1d(center_2, scan_range_2, num_steps_2, dtype)

    ### Winding cartesian product
    # The first axis values are repeated - the second axis values are mirrored and tiled
    # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

    # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
    inter_1 = np.concatenate((coords_1_1d, np.flipud(coords_1_1d)))
    # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
    if num_steps_2 % 2 == 0:
        coords_1 = np.tile(inter_1, num_steps_2 // 2)
    else:  # Odd x size
        coords_1 = np.tile(inter_1, num_steps_2 // 2)
        coords_1 = np.concatenate((coords_1, coords_1_1d))

    # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
    coords_2 = np.repeat(coords_2_1d, num_steps_1)

    x_low = min(coords_1_1d)
    x_high = max(coords_1_1d)
    y_low = min(coords_2_1d)
    y_high = max(coords_2_1d)

    # For the image extent, we need to bump out the min/max axis coords by half the
    # pixel size in each direction so that the center of each pixel is properly aligned
    # with its coords
    x_half_pixel = (coords_1_1d[1] - coords_1_1d[0]) / 2
    y_half_pixel = (coords_2_1d[1] - coords_2_1d[0]) / 2
    img_extent = [
        x_high + x_half_pixel,
        x_low - x_half_pixel,
        y_low - y_half_pixel,
        y_high + y_half_pixel,
    ]

    return coords_1, coords_2, coords_1_1d, coords_2_1d, img_extent


def get_scan_cross_2d(
    center_1, center_2, scan_range_1, scan_range_2, num_steps_1, num_steps_2
):
    """Scan in a cross pattern. The first axis will be scanned while the second is held at its center,
    then the second axis will be scanned while the first is held at its center. This is useful for optimization

    Parameters
    ----------
    center_1 : numeric
        Center of the scan along the first axis
    center_2 : numeric
        Center of the scan along the second axis
    scan_range_1 : numeric
        Full range of the scan along the first axis
    scan_range_2 : numeric
        Full range of the scan along the second axis
    num_steps_1 : int
        Number of steps along the first axis
    num_steps_2 : int
        Number of steps along the second axis

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the cross scan
    array(numeric)
        Values to write to the second axis for the cross scan
    array(numeric)
        First-axis coordinates (i.e. coordinates scanned through while second axis is fixed)
    array(numeric)
        Second-axis coordinates (i.e. coordinates scanned through while first axis is fixed)
    """

    coords_1_1d = get_scan_1d(center_1, scan_range_1, num_steps_1)
    coords_2_1d = get_scan_1d(center_2, scan_range_2, num_steps_2)

    coords_1 = np.concatenate([coords_1_1d, np.full(num_steps_2, center_1)])
    coords_2 = np.concatenate([np.full(num_steps_1, center_2), coords_2_1d])

    return coords_1, coords_2, coords_1_1d, coords_2_1d


def get_scan_cross_3d(
    center_1,
    center_2,
    center_3,
    scan_range_1,
    scan_range_2,
    scan_range_3,
    num_steps_1,
    num_steps_2,
    num_steps_3,
):
    """Extension of get_scan_cross_2d to 3D"""

    coords_1_1d = get_scan_1d(center_1, scan_range_1, num_steps_1)
    coords_2_1d = get_scan_1d(center_2, scan_range_2, num_steps_2)
    coords_3_1d = get_scan_1d(center_3, scan_range_3, num_steps_3)

    coords_1 = np.concatenate(
        [coords_1_1d, np.full(num_steps_2 + num_steps_3, center_1)]
    )
    coords_2 = np.concatenate(
        [np.full(num_steps_1, center_2), coords_2_1d, np.full(num_steps_3, center_2)]
    )
    coords_3 = np.concatenate(
        [np.full(num_steps_1 + num_steps_2, center_3), coords_3_1d]
    )

    return coords_1, coords_2, coords_3, coords_1_1d, coords_2_1d, coords_3_1d


def get_scan_one_axis_2d(center_1, center_2, scan_range_1, num_steps_1):
    """Scan through the first axis, keeping the second fixed

    Parameters
    ----------
    center_1 : numeric
        Center of the scan along the first axis
    center_2 : numeric
        Center of the scan along the second axis
    scan_range_1 : numeric
        Full range of the scan along the first axis
    num_steps_1 : int
        Number of steps along the first axis

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the scan
    array(numeric)
        Values to write to the second axis for the scan
    """

    coords_1 = get_scan_1d(center_1, scan_range_1, num_steps_1)
    coords_2 = np.full(num_steps_1, center_2)
    return coords_1, coords_2


def get_scan_circle_2d(center_1, center_2, radius, num_steps):
    """Get coordinates for a scan around in a circle. Useful for checking galvo alignment

    Parameters
    ----------
    center_1 : numeric
        First-axis center of the circle
    center_2 : numeric
        First-axis center of the circle
    radius : numeric
        Radius of the circle
    num_steps : int
        Number of steps to discretize the circle into

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the scan
    array(numeric)
        Values to write to the second axis for the scan
    """

    angles = np.linspace(0, 2 * np.pi, num_steps)
    coords_1 = center_1 + (radius * np.sin(angles))
    coords_2 = center_2 + (radius * np.cos(angles))
    return coords_1, coords_2


def get_scan_two_point_2d(first_coord_1, first_coord_2, second_coord_1, second_coord_2):
    """Flip back an forth between two points - designed to be run continuously

    Parameters
    ----------
    first_coord_1 : numeric
        First point, first axis coordinate
    first_coord_2 : numeric
        First point, second axis coordinate
    second_coord_1 : numeric
        Second point, first axis coordinate
    second_coord_2 : numeric
        Second point, second axis coordinate

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the scan
    array(numeric)
        Values to write to the second axis for the scan
    """

    # Sometimes a minimum number of points is required in a stream, so
    # return a list of 64 coords to be safe
    coords_1 = [first_coord_1, second_coord_1] * 32
    coords_2 = [first_coord_2, second_coord_2] * 32
    return coords_1, coords_2


# endregion
