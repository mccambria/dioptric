# -*- coding: utf-8 -*-
"""
Z-axis calibration routine for Attocube ANC300 piezo.

Scans the Z-axis from top to bottom while monitoring photon counts to find
the sample surface. Sets the peak photon count position as Z=0 reference.
Includes safety monitoring to prevent sample collision.

Created on November 5th, 2025

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import NVSig


def main(
    nv_sig,
    scan_range=600,
    step_size=5,
    num_averages=100,
    safety_threshold=100,
    settling_time_ms=10,
):
    """
    Calibrate Z-axis by finding sample surface (peak photon counts).

    The routine moves the piezo to the top of its range, then scans downward
    while monitoring photon counts. The peak count position (surface) is set
    as Z=0 reference for all subsequent experiments.

    Parameters
    ----------
    nv_sig : dict
        NV center parameters (name, laser settings, etc.)
    scan_range : int
        Total number of steps to scan (default: 600)
    step_size : int
        Step increment during scan (default: 5)
    num_averages : int
        Number of photon count samples per step (default: 100)
    safety_threshold : int
        Minimum safe photon count - abort if below this (default: 100)
    settling_time_ms : float
        Wait time after each step in milliseconds (default: 10)

    Returns
    -------
    dict
        Calibration results containing:
        - z_surface_steps: Z position of surface (peak counts)
        - peak_counts: Maximum photon count value
        - all_steps: Array of all Z positions scanned
        - all_counts: Array of photon counts at each position
        - safety_triggered: Whether safety stop was activated
    """

    ### Setup

    # Get servers
    piezo = common.get_server("pos_xyz_ATTO_piezos")
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    # Get calibration config if available
    config = common.get_config_dict()
    if "z_calibration" in config.get("Positioning", {}):
        cal_config = config["Positioning"]["z_calibration"]
        scan_range = cal_config.get("scan_range", scan_range)
        step_size = cal_config.get("step_size", step_size)
        num_averages = cal_config.get("num_averages", num_averages)
        safety_threshold = cal_config.get("safety_threshold", safety_threshold)
        settling_time_ms = cal_config.get("settling_time_ms", settling_time_ms)

    settling_time_sec = settling_time_ms / 1000.0

    # Setup laser for imaging
    laser_key = "imaging_laser"
    readout_laser = nv_sig.get(laser_key, nv_sig.get("spin_readout_laser_key"))
    readout_dur = nv_sig.get("imaging_readout_dur", nv_sig.get("spin_readout_dur", 1e6))
    tb.set_filter(nv_sig, laser_key)
    readout_power = tb.set_laser_power(nv_sig, laser_key)

    # Load pulse sequence for photon counting (same as stationary_count)
    delay = 0  # No delay needed for continuous counting
    seq_args = [delay, readout_dur, readout_laser, readout_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "simple_readout.py"

    period = pulse_gen.stream_load(seq_file, seq_args_string)[0]

    ### Move to top of Z range

    print(f"Moving to top of Z range (+{scan_range} steps)...")
    start_position = piezo.get_z_position()
    top_position = start_position + scan_range
    piezo.write_z(top_position)
    time.sleep(0.5)  # Allow piezo to settle

    # Set this as the starting reference (Z = scan_range)
    piezo.set_z_reference(scan_range)
    print(f"Starting scan from Z={scan_range} steps")

    ### Scan downward and collect photon counts

    num_steps = int(scan_range / step_size) + 1
    z_steps = []
    photon_counts = []
    safety_triggered = False

    print(f"Scanning {num_steps} positions...")
    print(f"Safety threshold: {safety_threshold} counts")
    print()

    # Create figure for real-time monitoring (same as stationary_count)
    kpl.init_kplotlib()
    fig, ax = plt.subplots()

    # Initialize plot arrays with expected Z positions and NaN counts
    z_array_init = np.array([scan_range - (i * step_size) for i in range(num_steps)])
    counts_array_init = np.full(num_steps, np.nan)

    # Initialize plot with kpl.plot_line (same as stationary_count)
    kpl.plot_line(ax, z_array_init, counts_array_init)
    ax.set_xlabel("Z position (steps)")
    ax.set_ylabel("Photon counts")
    ax.set_title("Z-axis calibration")

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass

    # Start continuous streaming (same pattern as stationary_count)
    counter.start_tag_stream()
    pulse_gen.stream_start(-1)  # -1 means run continuously until stopped
    tb.init_safe_stop()

    # Pre-allocate arrays for all data (filled with NaN, matching the initialized plot)
    z_array = z_array_init.copy()
    counts_array = counts_array_init.copy()

    for step_ind in range(num_steps):
        if tb.safe_stop():
            print("User stopped calibration")
            safety_triggered = True
            break

        # Calculate target Z position (moving downward)
        target_z = scan_range - (step_ind * step_size)

        # Move to position
        piezo.write_z(target_z)
        time.sleep(settling_time_sec)

        # Clear buffer and collect fresh counts at this position
        counter.clear_buffer()
        time.sleep(0.01)  # Brief wait for buffer to fill

        # Read counts continuously until we have enough samples (like stationary_count pattern)
        counts = []
        read_start = time.time()
        timeout = 2.0  # 2 second timeout per position

        while len(counts) < num_averages:
            if tb.safe_stop():
                print("User stopped calibration")
                safety_triggered = True
                break
            if time.time() - read_start > timeout:
                print(f"Warning: timeout at Z={target_z}, got {len(counts)}/{num_averages} samples")
                break

            new_samples = counter.read_counter_simple()
            if len(new_samples) > 0:
                counts.extend(new_samples)
                # Update plot immediately with partial data (like stationary_count)
                mean_counts = np.mean(counts)
                counts_array[step_ind] = mean_counts
                kpl.plot_line_update(ax, x=z_array, y=counts_array, relim_x=False)

        if safety_triggered:
            break

        mean_counts = np.mean(counts) if len(counts) > 0 else 0
        z_steps.append(target_z)
        photon_counts.append(mean_counts)

        # Final update for this position
        counts_array[step_ind] = mean_counts
        kpl.plot_line_update(ax, x=z_array, y=counts_array, relim_x=False)
        print(f"Step {step_ind + 1}/{num_steps}: Z={target_z} steps, counts={mean_counts:.0f}")

        # Safety check
        if mean_counts < safety_threshold and len(photon_counts) > 3:
            print(f"WARNING: Photon counts ({mean_counts:.0f}) below safety threshold!")
            print(f"Stopping at Z={target_z} steps to prevent collision")
            safety_triggered = True
            break

    counter.stop_tag_stream()

    ### Analyze results

    z_steps = np.array(z_steps)
    photon_counts = np.array(photon_counts)

    if len(photon_counts) < 5:
        print("ERROR: Calibration aborted too early, insufficient data")
        return None

    # Find peak photon count position (surface)
    peak_idx = np.argmax(photon_counts)
    z_surface = z_steps[peak_idx]
    peak_counts = photon_counts[peak_idx]

    print()
    print(f"=== Calibration Results ===")
    print(f"Surface found at Z={z_surface} steps")
    print(f"Peak photon counts: {peak_counts:.0f}")
    print(f"Safety triggered: {safety_triggered}")

    # Move to surface position and set as Z=0
    print(f"Moving to surface position and setting as Z=0...")
    piezo.write_z(z_surface)
    time.sleep(0.2)
    piezo.set_z_reference(0)  # Set current position as Z=0

    print("Z-axis calibration complete! Surface set as Z=0")

    ### Create final plot

    ax.clear()
    ax.plot(z_steps, photon_counts, 'bo-', label='Measured counts')
    ax.plot(z_surface, peak_counts, 'r*', markersize=15, label=f'Surface (Z={z_surface})')
    ax.axhline(safety_threshold, color='r', linestyle='--', alpha=0.5, label=f'Safety threshold')
    ax.set_xlabel("Z position (steps)")
    ax.set_ylabel("Photon counts")
    ax.set_title(f"Z-axis calibration - Surface at Z={z_surface} steps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ### Save data

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "scan_range": scan_range,
        "step_size": step_size,
        "num_averages": num_averages,
        "safety_threshold": safety_threshold,
        "settling_time_ms": settling_time_ms,
        "z_surface_steps": int(z_surface),
        "peak_counts": float(peak_counts),
        "all_steps": z_steps.tolist(),
        "all_counts": photon_counts.tolist(),
        "safety_triggered": safety_triggered,
    }

    nv_name = nv_sig.get("name", "unknown")
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(fig, file_path)

    print(f"Data saved to: {file_path}")

    # Return results
    return raw_data


if __name__ == "__main__":
    # Example usage
    from utils.constants import NVSig

    # Create a minimal nv_sig for testing
    nv_sig = {
        "name": "test_calibration",
        "imaging_laser": "laser_INTE_520",
        "imaging_readout_dur": 1e6,  # 1 ms
    }

    results = main(
        nv_sig,
        scan_range=600,
        step_size=5,
        num_averages=100,
        safety_threshold=100,
    )
