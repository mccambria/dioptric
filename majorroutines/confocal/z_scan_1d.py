# -*- coding: utf-8 -*-
"""
1D Z-axis scan with photon counting and real-time plotting.

Performs a scan along the Z-axis without moving X/Y position, collecting
photon counts at each Z position and displaying a line plot.


Created on November 5th, 2025

"""

import time

import matplotlib.pyplot as plt
import numpy as np

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CountFormat, CoordsKey, NVSig, VirtualLaserKey

SEQ_FILE_PIXEL_READOUT = "simple_readout.py"


def main(
    nv_sig: NVSig,
    num_steps,
    step_size,
    num_averages=1,
    min_threshold=None,
    nv_minus_init=False,
    save_data=True,
):
    """
    Perform a 1D scan along the Z-axis, collecting photon counts at each position.

    Uses the confocal "step and measure" pattern:
    - Move Z-axis relatively from current position
    - Trigger pulse sequence for each measurement
    - Read photon counts
    - Average multiple samples per position if requested
    - Monitor threshold and pause for user input if counts drop too low

    This routine does NOT move X or Y coordinates - only Z-axis scanning.

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters including pulse durations, laser powers, and initial coordinates
    num_steps : int
        Number of Z positions to scan. Total distance = num_steps * step_size
    step_size : int
        Step size in piezo units per iteration.
        Negative values move toward surface (down), positive values move away (up).
        Example: step_size=-10 moves down 10 units per step
    num_averages : int, optional
        Number of photon count samples to collect and average at each Z position.
        Default: 1 (matches confocal scan behavior for single sample per position)
        Higher values improve statistics but slow down the scan.
    min_threshold : float or None, optional
        Minimum photon count threshold for safety monitoring.
        If counts drop below this value, scan pauses and prompts:
        "Continue scanning? (y/n)"
        User must type 'y' to continue or 'n' to abort.
        None = no threshold monitoring (default: None)
    nv_minus_init : bool, optional
        Whether to use NV- charge state initialization (two-gate readout).
        True: uses modulo gates, subtracts background
        False: simple single-gate readout (default: False)
    save_data : bool, optional
        Whether to save data and plot to disk (default: True)
        Saves to: nvdata/pc_{hostname}/z_scan_1d/{date}/{timestamp}/

    Returns
    -------
    counts_array : numpy.ndarray
        1D array of photon counts at each Z position.
        Units: kcps (kilocounts per second) if config["count_format"] == KCPS,
               raw counts otherwise
        Length: num_steps_completed (may be less than num_steps if aborted early)
    z_positions_array : numpy.ndarray
        1D array of actual Z positions in piezo steps.
        These are the real positions returned by the piezo after each move.
        Length: matches counts_array

    """

    ### Setup

    cfg = common.get_config_dict()
    count_fmt: CountFormat = cfg["count_format"]

    # Get servers
    piezo = pos.get_positioner_server(CoordsKey.Z)
    ctr = tb.get_server_counter()
    pulse = tb.get_server_pulse_streamer()

    # Setup laser for imaging
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"]))
    )
    readout_s = readout_ns * 1e-9
    readout_laser = vld["physical_name"]

    tb.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tb.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Initialize data storage (will be populated during scan)
    z_positions = []  # Actual positions from piezo
    counts_1d = []    # Raw counts at each position

    ### Setup figure for real-time plotting

    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Z Position (steps)")
    ax.set_ylabel("Kcps" if count_fmt == CountFormat.KCPS else "Raw Counts")
    ax.set_title(f"{readout_laser}, {readout_ns/1e6:.1f} ms readout")

    # Initial empty plot
    (line,) = ax.plot([], [], "b.-", markersize=4)
    ax.grid(True, alpha=0.3)
    plt.ion()  # Enable interactive mode

    tb.reset_cfm()
    tb.init_safe_stop()

    # Sequence loading - load once, trigger per position (confocal pattern)
    delay_ns = 0  # No delay for single-pixel readout
    period_ns = pulse.stream_load(
        SEQ_FILE_PIXEL_READOUT,
        tb.encode_seq_args([delay_ns, readout_ns, readout_laser, readout_power]),
    )[0]

    # Start tag stream once (confocal pattern)
    ctr.start_tag_stream()

    ### Scan through Z positions

    print(f"Starting Z scan: {num_steps} steps of {step_size} units")
    print(f"Direction: {'toward surface (down)' if step_size < 0 else 'away from surface (up)'}")
    if min_threshold is not None:
        print(f"Threshold monitoring enabled: {min_threshold:.0f} counts")
    print()

    try:
        for i in range(num_steps):
            if tb.safe_stop():
                print("\n[STOPPED] User interrupt detected")
                break

            # Move Z position relatively
            current_z_pos = piezo.move_z_steps(step_size)
            time.sleep(0.01)  # Brief settling time

            # Collect num_averages samples at this position (confocal pattern)
            counts = []
            for _ in range(num_averages):
                if tb.safe_stop():
                    break

                # Trigger exactly 1 pulse sequence (confocal pattern)
                pulse.stream_start(1)

                # Read exactly 1 sample (blocking call)
                if nv_minus_init:
                    raw = ctr.read_counter_modulo_gates(2, 1)  # [[a,b]]
                    val_single = max(int(raw[0][0]) - int(raw[0][1]), 0)
                else:
                    raw = ctr.read_counter_simple(1)  # [c]
                    val_single = int(raw[0])

                counts.append(val_single)

            # Calculate mean of samples
            val = int(np.mean(counts)) if len(counts) > 0 else 0

            # Store actual position and counts
            z_positions.append(current_z_pos)
            counts_1d.append(val)

            # Status printing (every 20 steps)
            if (i + 1) % 20 == 0 or i == 0:
                print(f"Step {i+1}/{num_steps}: Z={current_z_pos} steps, Counts={val:.0f}")

            # Check threshold
            if min_threshold is not None and val < min_threshold:
                print(f"\n[THRESHOLD REACHED]")
                print(f"  Current counts: {val:.0f}")
                print(f"  Threshold: {min_threshold:.0f}")
                print(f"  Position: Z={current_z_pos} steps")
                response = input("  Continue scanning? (y/n): ").strip().lower()
                if response != 'y':
                    print("[STOPPED] Scan aborted by user")
                    break
                print("  Continuing scan...\n")

            # Update plot (every step for real-time visualization)
            # Convert to arrays for plotting
            z_array = np.array(z_positions)
            counts_array = np.array(counts_1d)

            # Convert to kcps if needed
            if count_fmt == CountFormat.KCPS:
                plot_data = (counts_array / 1000.0) / readout_s
            else:
                plot_data = counts_array

            # Update line data
            line.set_data(z_array, plot_data)

            # Auto-scale axes
            ax.relim()
            ax.autoscale_view()

            # Refresh plot
            plt.pause(0.01)

    finally:
        ctr.stop_tag_stream()
        tb.reset_safe_stop()
        tb.reset_cfm()
        print(f"\nScan complete: collected {len(z_positions)} data points")

    ### Save data

    print("Processing data...", flush=True)

    # Convert lists to numpy arrays for processing and output
    z_positions_array = np.array(z_positions)
    counts_1d_array = np.array(counts_1d)

    is_kcps = count_fmt == CountFormat.KCPS
    counts_out = (counts_1d_array / 1000.0) / readout_s if is_kcps else counts_1d_array
    units_out = "kcps" if is_kcps else "counts"

    ts = dm.get_time_stamp()
    raw_data = {
        "timestamp": ts,
        "nv_sig": nv_sig,
        "mode": "z_scan_1d",
        "num_steps_requested": num_steps,
        "num_steps_completed": len(z_positions),
        "num_averages": num_averages,
        "step_size": step_size,
        "min_threshold": min_threshold,
        "readout_ns": readout_ns,
        "readout_units": "ns",
        "counts_array": counts_out.astype(float).tolist(),
        "counts_array_units": units_out,
        "z_positions": z_positions_array.tolist(),
    }

    if save_data:
        print("Saving data...", flush=True)
        path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
        dm.save_figure(fig, path)
        dm.save_raw_data(raw_data, path)
        print(f"Data saved to: {path}")

    # Turn off interactive mode and show final plot
    plt.ioff()
    print("Displaying final plot (close window to exit)...", flush=True)
    kpl.show()

    return counts_out, z_positions_array


if __name__ == "__main__":
    # Example usage
    from utils.constants import CoordsKey, NVSig

    nv_sig = NVSig(
        name="test_z_scan",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.Z: 0},
        pulse_durations={VirtualLaserKey.IMAGING: int(5e6)},
    )

    results = main(
        nv_sig,
        num_steps=500,           # Number of steps to scan
        step_size=-1,          # Negative = toward surface (down)
        num_averages=1,       # Samples per position
        min_threshold=100,     # Pause if counts drop below 1000
    )
