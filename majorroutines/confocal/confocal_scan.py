# -*- coding: utf-8 -*-
"""
Uses the generic confocal_base_routine to run a Rabi experiment
Created on August 2th, 2026

@author: schand
"""
# confocal_scanning.py

import numpy as np
import time
from utils import tool_belt as tb
from utils import data_manager as dm


def confocal_image(
    experiment_fn,
    scan_coords,
    image_range,
    num_steps,
    scan_axis="xy",
    drift_correct=True,
    live_plot_fn=None,
    **experiment_kwargs,
):
    """
    Scan the specified region and run the provided experiment function at each step.

    Args:
        experiment_fn: callable that returns a measurement (e.g., ESR contrast, photon count)
        scan_coords: center [x, y, z] position
        image_range: [x_range, y_range] or [x_range, z_range]
        num_steps: number of points per axis
        scan_axis: "xy", "xz", or "yz"
        drift_correct: apply drift compensation from NV center tracking
        live_plot_fn: optional function to update live scan image
        experiment_kwargs: keyword arguments to pass into experiment_fn
    Returns:
        data_dict with shape (num_steps, num_steps)
    """
    if drift_correct:
        drift = tb.get_drift()
    else:
        drift = [0, 0, 0]
    x0, y0, z0 = np.array(scan_coords) + np.array(drift)

    scan_axes = {
        "xy": (x0, y0),
        "xz": (x0, z0),
        "yz": (y0, z0),
    }
    center_1, center_2 = scan_axes[scan_axis]
    v1_vals, v2_vals = tb.calc_image_scan_vals(
        center_1, center_2, image_range, num_steps
    )

    # Setup positioners
    xy_server = tb.get_xy_server()
    z_server = tb.get_z_server()

    result_map = np.zeros((num_steps, num_steps))

    for i2, v2 in enumerate(v2_vals):
        parity = 1 if i2 % 2 == 0 else -1
        for i1 in range(num_steps)[::parity]:
            v1 = v1_vals[i1]

            # Move to scan position
            if scan_axis == "xy":
                xy_server.write_xy(v1, v2)
            elif scan_axis == "xz":
                xy_server.write_xy(v1, y0)
                z_server.write_z(v2)
            elif scan_axis == "yz":
                xy_server.write_xy(x0, v1)
                z_server.write_z(v2)

            time.sleep(0.1)  # Wait for stabilization

            # Run experiment at this point
            result = experiment_fn(**experiment_kwargs)
            result_map[i2, i1] = result

            if live_plot_fn:
                live_plot_fn(result_map)

    # Wrap-up
    xy_server.write_xy(x0, y0)
    tb.reset_cfm()

    # Save
    timestamp = dm.get_time_stamp()
    data = {
        "timestamp": timestamp,
        "scan_axis": scan_axis,
        "scan_center": scan_coords,
        "drift": drift,
        "image_range": image_range,
        "num_steps": num_steps,
        "result_map": result_map,
    }
    return data
