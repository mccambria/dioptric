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

    This routine:
    - Does NOT move X or Y coordinates
    - Starts from current Z position and moves relatively by step_size
    - Scans Z-axis for num_steps iterations
    - Collects photon counts at each Z position
    - Displays real-time line plot of counts vs Z position
    - Can pause if counts drop below threshold (requires user input to continue)
    - Saves data and plot

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters
    num_steps : int
        Number of Z positions to scan
    step_size : int
        Step size in piezo units (negative = toward surface/down, positive = away/up)
    num_averages : int
        Number of photon count samples to average at each Z position (default: 1)
    min_threshold : float or None
        Minimum photon count threshold. If counts drop below this value, scan pauses
        and prompts user to continue or abort. None = no threshold check (default: None)
    nv_minus_init : bool
        Whether to use charge initialization (default: False)
    save_data : bool
        Whether to save the data and plot (default: True)

    Returns
    -------
    tuple
        (counts_array, z_positions) - counts in kcps or raw depending on config
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

    # Sequence loading (same as stationary_count - load once)
    delay_ns = 0  # No delay for continuous counting
    period_ns = pulse.stream_load(
        SEQ_FILE_PIXEL_READOUT,
        tb.encode_seq_args([delay_ns, readout_ns, readout_laser, readout_power]),
    )[0]

    # Start continuous streaming (same pattern as stationary_count)
    ctr.start_tag_stream()
    pulse.stream_start(-1)  # -1 means run continuously until stopped

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
            print(f"[DEBUG] Step {i+1}: Moving Z by {step_size}...", flush=True)
            current_z_pos = piezo.move_z_steps(step_size)
            print(f"[DEBUG] Step {i+1}: Now at Z={current_z_pos}", flush=True)
            time.sleep(0.05)  # Settling time

            # Clear buffer and collect fresh counts
            ctr.clear_buffer()
            time.sleep(0.05)  # Wait for fresh data

            # Read counts (collect num_averages samples)
            print(f"[DEBUG] Step {i+1}: Reading {num_averages} samples...", flush=True)
            counts = []
            read_start = time.time()
            timeout = 2.0

            while len(counts) < num_averages:
                if tb.safe_stop():
                    break
                if time.time() - read_start > timeout:
                    print(f"[WARNING] Timeout reading counts at step {i+1}")
                    break
                new_samples = ctr.read_counter_simple()
                if len(new_samples) > 0:
                    counts.extend(new_samples)

            print(f"[DEBUG] Step {i+1}: Collected {len(counts)} samples", flush=True)

            # Calculate mean
            if nv_minus_init and len(counts) >= 2:
                # For charge init, pair samples and subtract
                val = max(int(counts[0]) - int(counts[1]), 0) if len(counts) >= 2 else 0
            else:
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

            # Update plot (throttled to every 5 steps)
            # if (i + 1) % 5 == 0 or i == 0 or i == num_steps - 1:
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
        pulse.stream_stop()
        tb.reset_safe_stop()
        tb.reset_cfm()
        print(f"\nScan complete: collected {len(z_positions)} data points")

    ### Save data

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
        path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
        dm.save_figure(fig, path)
        dm.save_raw_data(raw_data, path)

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
