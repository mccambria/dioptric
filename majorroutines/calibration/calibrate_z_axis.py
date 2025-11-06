# -*- coding: utf-8 -*-
"""
Z-axis calibration routine for Attocube ANC300 piezo.

Scans the Z-axis from top to bottom while monitoring photon counts to find
the sample surface. Sets the peak photon count position as Z=0 reference.
Includes safety monitoring to prevent sample collision.

Created on November 5th, 2025


"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, NVSig, VirtualLaserKey


def main(
    nv_sig,
    scan_range=600,
    step_size=5,
    num_averages=100,
    safety_threshold=150,
    settling_time_ms=10,
):
    """
    Calibrate Z-axis by finding sample surface

    The routine moves the piezo to the top of its range, then scans downward
    while monitoring photon counts. The peak count position (surface) is set
    as Z=0 reference

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
        Minimum safe photon count - abort if below this (default: 150)
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

    # Get servers using proper positioning utilities
    piezo = pos.get_positioner_server(CoordsKey.Z)
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    print(f"[DEBUG] Piezo server: {piezo}")
    print(f"[DEBUG] Counter server: {counter}")
    print(f"[DEBUG] Pulse gen server: {pulse_gen}")

    # Ensure clean state before starting (in case previous run was cancelled)
    print("[DEBUG] Cleaning up any previous streams...")
    tb.reset_cfm()
    tb.reset_safe_stop()
    try:
        counter.stop_tag_stream()
    except:
        pass
    time.sleep(0.1)

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
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_dur = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"]))
    )
    readout_laser = vld["physical_name"]
    tb.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tb.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Load pulse sequence for photon counting 
    delay = 0  # No delay needed for continuous counting
    seq_args = [delay, readout_dur, readout_laser, readout_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "simple_readout.py"

    period = pulse_gen.stream_load(seq_file, seq_args_string)[0]

    ### Setup scan parameters and plot

    num_steps = int(scan_range / step_size) + 1
    z_steps = []
    photon_counts = []
    safety_triggered = False

    print(f"Scanning {num_steps} positions...")
    print(f"Safety threshold: {safety_threshold} counts")
    print()

    # Create figure for real-time monitoring
    kpl.init_kplotlib()
    fig, ax = plt.subplots()

    # Initialize plot with placeholder data (will update x-axis after we know actual positions)
    placeholder_x = np.linspace(0, scan_range, num_steps)
    placeholder_y = np.full(num_steps, np.nan)

    # Initialize plot with kpl.plot_line
    kpl.plot_line(ax, placeholder_x, placeholder_y)
    ax.set_xlabel("Piezo steps (relative position)")
    ax.set_ylabel("Photon counts")
    ax.set_title("Z-axis calibration scan")
    # Will set proper x-limits after we move to starting position

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass

    # Start continuous streaming immediately for instant feedback
    print("[DEBUG] Starting counter tag stream...")
    counter.start_tag_stream()
    print("[DEBUG] Starting pulse generator stream...")
    pulse_gen.stream_start(-1)  # -1 means run continuously until stopped
    tb.init_safe_stop()

    # Verify we're getting samples before proceeding
    print("[DEBUG] Verifying photon counting is working...")
    time.sleep(0.2)  # Allow time for samples to accumulate
    test_samples = counter.read_counter_simple()
    if len(test_samples) == 0:
        print("[ERROR] No photon counts detected!")

        counter.stop_tag_stream()
        return None
    else:
        avg_count = np.mean(test_samples)
        print(f"[DEBUG] Photon counting working! Initial average: {avg_count:.0f} counts")
        print(f"[DEBUG] Got {len(test_samples)} samples in 0.2s")

    ### Continuous upward scan until user stops or peak found

    print(f"\n=== Continuous upward scan - monitoring for surface ===")
    pos_start = piezo.get_z_position()
    print(f"Starting position: {pos_start} steps")
    print(f"Moving up continuously in 10-step increments")
    print(f"Press CTRL+C to stop when you see the peak\n")

    # Get initial counts
    counter.clear_buffer()
    time.sleep(0.1)
    baseline_samples = counter.read_counter_simple()
    baseline_counts = np.mean(baseline_samples) if len(baseline_samples) > 0 else 0
    print(f"Initial counts: {baseline_counts:.0f}\n")

    MOVE_INCREMENT = 10  # Small steps for continuous monitoring
    PEAK_THRESHOLD = 2500  # Surface detection
    MAX_SAFE_COUNTS = 4000  # Auto-stop if we go way past reasonable peak

    counts_history = [baseline_counts]
    surface_peak_value = 0
    surface_peak_position = None
    move_count = 0

    try:
        while True:
            if tb.safe_stop():
                print("\nUser stopped scan")
                break

            move_count += 1
            pos_before = piezo.get_z_position()

            # Move up
            pos_after = piezo.move_z_steps(MOVE_INCREMENT)
            time.sleep(0.15)  # Wait for movement

            # Get counts
            counter.clear_buffer()
            time.sleep(0.05)

            try:
                samples = counter.read_counter_simple()
                if len(samples) == 0:
                    print(f"Move {move_count}: step {pos_after}, NO SAMPLES")
                    continue

                current_counts = np.mean(samples)
                count_change = current_counts - counts_history[-1]
                counts_history.append(current_counts)

                # Show current status
                print(f"Move {move_count}: step {pos_after}, counts={current_counts:.0f} (change:{count_change:+.0f})")

                # Track peak
                if current_counts > PEAK_THRESHOLD:
                    if current_counts > surface_peak_value:
                        surface_peak_value = current_counts
                        surface_peak_position = pos_after
                        print(f"    >>> NEW PEAK DETECTED: {current_counts:.0f} at step {pos_after} <<<")

                # Safety: auto-stop if counts way too high (something wrong)
                if current_counts > MAX_SAFE_COUNTS:
                    print(f"\n[WARNING] Counts unexpectedly high ({current_counts:.0f}), stopping for safety")
                    break

            except Exception as e:
                print(f"Move {move_count}: Error reading counts - {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nScan interrupted by user")

    scan_start_position = piezo.get_z_position()
    total_moved = scan_start_position - pos_start

    print(f"\n=== Scan Summary ===")
    print(f"Starting position: {pos_start}")
    print(f"Final position: {scan_start_position}")
    print(f"Total moved up: {total_moved} steps")
    if surface_peak_value > 0:
        print(f"Peak detected: {surface_peak_value:.0f} counts at step {surface_peak_position}")
    else:
        print(f"No significant peak found (max seen: {max(counts_history):.0f})")

    # Use the detected peak or current position
    z_steps = []
    photon_counts = []
    safety_triggered = False

    if surface_peak_value > 0 and surface_peak_position is not None:
        print(f"\nUsing detected peak for calibration")
        z_steps = [surface_peak_position]
        photon_counts = [surface_peak_value]
        chosen_position = surface_peak_position
    else:
        print(f"\nNo peak detected - would you like to use current position as reference?")
        # For now, skip calibration if no peak found
        counter.stop_tag_stream()
        tb.reset_cfm()
        return None

    # Move to the chosen position
    print(f"\nMoving to surface position (step {chosen_position})...")
    piezo.write_z(chosen_position)
    time.sleep(0.2)

    # Set as reference
    print(f"Setting this position as Z=0 reference...")
    piezo.set_z_reference(0)

    print(f"\nCalibration complete!")
    print(f"Surface is now at step=0")

    # Stop streams
    counter.stop_tag_stream()
    tb.reset_cfm()

    ### Save and return
    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "calibration_method": "continuous_upward_scan",
        "surface_step_position": int(chosen_position),
        "peak_counts": float(surface_peak_value) if surface_peak_value > 0 else 0,
        "total_steps_moved": int(total_moved),
        "counts_history": [float(c) for c in counts_history],
        "note": "Surface position set as step=0 reference after calibration",
    }

    nv_name = getattr(nv_sig, "name", "unknown")
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_raw_data(raw_data, file_path)

    print(f"Data saved to: {file_path}")
    kpl.show()

    return raw_data

    # OLD CODE BELOW - NOT EXECUTED
    if False:
        # Continue in SAME direction as Phase 1
        # Data shows: INCREASING step numbers = moving toward sample
        # Phase 1 went UP (increased steps), Phase 2 should CONTINUE UP
        phase2_step_size = 1  # Check every step to avoid collision
        num_steps_phase2 = scan_range  # One measurement per step
        scan_end_position = scan_start_position + scan_range  # CONTINUE UPWARD

        print(f"\n=== PHASE 2: Continuing upward scan toward surface ===")
        print(f"[DEBUG] Starting from: {scan_start_position}")
        print(f"[DEBUG] Target end: {scan_end_position}")
        print(f"[DEBUG] Step size: {phase2_step_size} (checking EVERY step for safety)")
        print(f"[DEBUG] Safety threshold: {safety_threshold} counts")
        print(f"[DEBUG] Direction: INCREASING steps = continuing toward surface")
        print(f"[DEBUG] Will STOP EARLY if peak detected and confirmed")

        ### Continue scanning upward toward surface

        # Pre-allocate arrays for all data (filled with NaN)
        actual_z_positions = np.array([scan_start_position + (i * phase2_step_size) for i in range(num_steps_phase2)])
        counts_array = np.full(num_steps_phase2, np.nan)

        # Update plot x-axis
        ax.set_xlim(scan_start_position - 5, scan_end_position + 5)
        print(f"[DEBUG] Plot x-axis set to [{scan_start_position}, {scan_end_position}]")

        # Smart early stopping variables
        found_peak_value = 0
        found_peak_position = None
        past_peak = False
        PEAK_CONFIRM_THRESHOLD = 2000  # Stop when counts drop below this after seeing peak

        try:
            for step_ind in range(num_steps_phase2):
                if tb.safe_stop():
                    print("User stopped calibration")
                    safety_triggered = True
                    break

                # Calculate target step position (continuing upward toward sample)
                target_steps = scan_start_position + (step_ind * phase2_step_size)

                # Move to position (absolute positioning using cached coordinates)
                print(f"\n[DEBUG] Moving to step {target_steps}...")
                piezo.write_z(target_steps)

                # Use longer settling time for more reliable measurements
                settle_time = max(settling_time_sec, 0.05)  # At least 50ms
                time.sleep(settle_time)

                # Clear buffer and collect fresh counts at this position
                counter.clear_buffer()
                time.sleep(0.05)  # Longer wait for buffer to stabilize

                # Read counts continuously until we have enough samples
                counts = []
                read_start = time.time()
                timeout = 3.0  # 3 second timeout per position
                read_attempts = 0
                max_read_attempts = 50

                while len(counts) < num_averages and read_attempts < max_read_attempts:
                    if tb.safe_stop():
                        print("User stopped calibration")
                        safety_triggered = True
                        break
                    if time.time() - read_start > timeout:
                        print(f"\n[WARNING] Timeout at step {target_steps}! Got {len(counts)}/{num_averages} samples")
                        break

                    read_attempts += 1
                    try:
                        new_samples = counter.read_counter_simple()
                        if len(new_samples) > 0:
                            counts.extend(new_samples)
                            # Update plot every 10 samples to reduce overhead
                            if len(counts) % 10 == 0 or len(counts) >= num_averages:
                                mean_counts = np.mean(counts)
                                counts_array[step_ind] = mean_counts
                                kpl.plot_line_update(ax, x=actual_z_positions, y=counts_array, relim_x=False)
                        else:
                            time.sleep(0.01)  # Small delay if no samples available
                    except Exception as e:
                        print(f"\n[ERROR] Counter read failed: {e}")
                        time.sleep(0.05)  # Wait before retry
                        counter.clear_buffer()  # Try clearing buffer on error
                        time.sleep(0.02)

                if safety_triggered:
                    break

                mean_counts = np.mean(counts) if len(counts) > 0 else 0
                z_steps.append(target_steps)
                photon_counts.append(mean_counts)

                # Final update for this position
                counts_array[step_ind] = mean_counts
                kpl.plot_line_update(ax, x=actual_z_positions, y=counts_array, relim_x=False)
                print(f"==> Position {step_ind + 1}/{num_steps_phase2}: step {target_steps}, counts={mean_counts:.0f}")

                # Smart peak detection with early stopping
                if mean_counts > PEAK_THRESHOLD:
                    if mean_counts > found_peak_value:
                        found_peak_value = mean_counts
                        found_peak_position = target_steps
                        print(f"    >>> NEW PEAK: {mean_counts:.0f} at step {target_steps}")
                elif found_peak_value > 0:
                    # We've seen a peak, check if we've dropped below confirmation threshold
                    if mean_counts < PEAK_CONFIRM_THRESHOLD:
                        print(f"\n[SUCCESS] Peak confirmed! Stopping scan early")
                        print(f"    Peak: {found_peak_value:.0f} counts at step {found_peak_position}")
                        print(f"    Current: {mean_counts:.0f} counts (dropped below {PEAK_CONFIRM_THRESHOLD})")
                        break

                # Safety check - absolute minimum
                if mean_counts < safety_threshold and len(photon_counts) > 3:
                    print(f"\n[WARNING] Photon counts ({mean_counts:.0f}) below safety threshold!")
                    print(f"[WARNING] Stopping at step {target_steps} to prevent collision")
                    safety_triggered = True
                    break

        finally:
            # Always stop streams and reset, even if cancelled or error occurs
            print("\n[DEBUG] Stopping streams...")
            try:
                counter.stop_tag_stream()
            except:
                pass
            tb.reset_cfm()
            tb.reset_safe_stop()

    ### Analyze results

    z_steps = np.array(z_steps)
    photon_counts = np.array(photon_counts)

    if len(photon_counts) < 5:
        print("ERROR: Calibration aborted too early, insufficient data")
        return None

    # Find peak photon count position (surface)
    peak_idx = np.argmax(photon_counts)
    surface_steps = z_steps[peak_idx]
    peak_counts = photon_counts[peak_idx]

    print()
    print(f"=== Calibration Results ===")
    print(f"Surface found at step position: {surface_steps}")
    print(f"Peak photon counts: {peak_counts:.0f}")
    print(f"Safety triggered: {safety_triggered}")

    # Move to surface position
    print(f"\nMoving to surface position (step {surface_steps})...")
    piezo.write_z(surface_steps)
    time.sleep(0.2)

    # Set surface as step=0 reference point
    print(f"Setting surface as reference point (step=0)...")
    piezo.set_z_reference(0)

    print(f"\n Z-axis calibration complete")
    print(f"  Surface is now at step=0")
    print(f"  You can move relative to surface using positive (away) or negative (toward) steps")

    ### Create final plot

    ax.clear()
    ax.plot(z_steps, photon_counts, 'bo-', label='Measured counts')
    ax.plot(surface_steps, peak_counts, 'r*', markersize=15, label=f'Surface (was step {surface_steps})')
    ax.axhline(safety_threshold, color='r', linestyle='--', alpha=0.5, label=f'Safety threshold')
    ax.set_xlabel("Piezo steps (relative position)")
    ax.set_ylabel("Photon counts")
    ax.set_title(f"Z-axis calibration - Surface at step {surface_steps} (now set to step=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # X-limits based on actual scan range
    if len(z_steps) > 0:
        ax.set_xlim(max(z_steps) + 5, min(z_steps) - 5)

    ### Save data

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "scan_range_steps": scan_range,
        "step_size": step_size,
        "num_averages": num_averages,
        "safety_threshold": safety_threshold,
        "settling_time_ms": settling_time_ms,
        "surface_step_position": int(surface_steps),
        "peak_counts": float(peak_counts),
        "all_step_positions": z_steps.tolist(),
        "all_counts": photon_counts.tolist(),
        "safety_triggered": safety_triggered,
        "note": "Surface position set as step=0 reference after calibration",
    }

    nv_name = getattr(nv_sig, "name", "unknown")
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(fig, file_path)

    print(f"Data saved to: {file_path}")

    # Return results
    return raw_data


if __name__ == "__main__":
    # Example usage
    from utils.constants import NVSig, CoordsKey, VirtualLaserKey

    # Create a minimal nv_sig for testing
    nv_sig = NVSig(
        name="test_calibration",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.Z: 0},
        pulse_durations={VirtualLaserKey.IMAGING: int(1e6)},  # 1 ms
    )

    results = main(
        nv_sig,
        scan_range=600,
        step_size=5,
        num_averages=100,
        safety_threshold=150,
    )
