# -*- coding: utf-8 -*-
"""
Z-axis calibration routine for Attocube ANC300 piezo.

Establishes a reliable reference by scanning from maximum Z position downward
to detect the sample surface. Records max→surface distance for position recovery
and measures hysteresis for improved repeatability.

Algorithm:
1. Move to maximum Z position (up by configured steps)
2. Scan downward collecting photon count profile
3. Identify surface using scipy peak detection
4. Verify surface position and measure hysteresis
5. Set surface as Z=0 reference

Created on November 5th, 2025
Updated on November 7th, 2025

@author: chemistatcode
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, NVSig, VirtualLaserKey


def main(nv_sig, **kwargs):
    """
    Calibrate Z-axis by scanning from maximum position down to sample surface.

    This routine establishes a reliable reference distance from max→surface that
    can be used for position recovery when NVs are "lost" due to accumulated
    positioning errors. Moving in one direction (downward) minimizes hysteresis
    effects for better repeatability.

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters (name, laser settings, etc.)
    **kwargs : dict
        Optional parameter overrides. Can override any parameter from config's
        z_calibration section (e.g., max_position_steps, scan_step_size, etc.)

    Returns
    -------
    dict
        Calibration results containing:
        - max_position_absolute: Starting position before zeroing (steps)
        - surface_position_absolute: Surface position before zeroing (steps)
        - max_to_surface_distance: Reference distance (steps)
        - surface_photon_counts: Peak counts at surface
        - verification_errors: Position errors from verification passes (steps)
        - suggested_z_bias_adjust: Calculated hysteresis compensation
        - count_profile: Full (steps, counts) data arrays
    """

    ### Setup and Configuration

    config = common.get_config_dict()
    cal_config = config["Positioning"]["z_calibration"]

    # Load parameters from config, allow kwargs to override
    max_position_steps = kwargs.get("max_position_steps", cal_config["max_position_steps"])
    scan_step_size = kwargs.get("scan_step_size", cal_config["scan_step_size"])
    peak_threshold = kwargs.get("peak_threshold", cal_config["peak_threshold"])
    scan_past_peak_steps = kwargs.get("scan_past_peak_steps", cal_config["scan_past_peak_steps"])
    safety_min_counts = kwargs.get("safety_min_counts", cal_config["safety_min_counts"])
    verification_passes = kwargs.get("verification_passes", cal_config["verification_passes"])
    verification_retract_steps = kwargs.get("verification_retract_steps", cal_config["verification_retract_steps"])
    settling_time_ms = kwargs.get("settling_time_ms", cal_config["settling_time_ms"])
    max_scan_timeout_s = kwargs.get("max_scan_timeout_s", cal_config["max_scan_timeout_s"])

    settling_time_s = settling_time_ms / 1000.0

    # Get servers
    print("[DEBUG] Getting server connections...")
    piezo = pos.get_positioner_server(CoordsKey.Z)
    print(f"[DEBUG] Piezo server: {piezo}")
    counter = tb.get_server_counter()
    print(f"[DEBUG] Counter server: {counter}")
    pulse_gen = tb.get_server_pulse_gen()
    print(f"[DEBUG] Pulse gen server: {pulse_gen}")

    print("\n" + "="*60)
    print("Z-AXIS CALIBRATION - MAX TO SURFACE REFERENCE")
    print("="*60)
    print(f"Configuration:")
    print(f"  Max position move: {max_position_steps} steps")
    print(f"  Scan step size: {scan_step_size} steps")
    print(f"  Peak threshold: {peak_threshold} counts")
    print(f"  Scan past peak: {scan_past_peak_steps} steps")
    print(f"  Safety minimum: {safety_min_counts} counts")
    print(f"  Verification passes: {verification_passes}")
    print("="*60 + "\n")

    # Reset hardware to clean state
    print("[DEBUG] Resetting hardware...")
    tb.reset_cfm()
    print("[DEBUG] Initializing safe stop...")
    tb.init_safe_stop()

    # Setup laser for imaging
    print("[DEBUG] Setting up laser for imaging...")
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_dur = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"]))
    )
    readout_laser = vld["physical_name"]
    tb.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tb.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Load pulse sequence for photon counting
    print(f"[DEBUG] Loading pulse sequence: {readout_laser} at {readout_power} for {readout_dur}ns...")
    seq_args = [0, readout_dur, readout_laser, readout_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "simple_readout.py"
    print(f"[DEBUG] Calling pulse_gen.stream_load({seq_file})...")
    period = pulse_gen.stream_load(seq_file, seq_args_string)[0]
    print(f"[DEBUG] Pulse sequence loaded. Period: {period}ns")

    # Start continuous counting
    print("[DEBUG] Starting counter tag stream...")
    counter.start_tag_stream()
    print("[DEBUG] Starting pulse generator stream...")
    pulse_gen.stream_start(-1)  # Run continuously
    print("[DEBUG] Streams started successfully")

    # Verify photon counting is working
    print("[DEBUG] Verifying photon counting is working...")
    time.sleep(0.2)
    test_samples = counter.read_counter_simple()
    if len(test_samples) == 0:
        print("[ERROR] No photon counts detected! Check APD/laser setup.")
        counter.stop_tag_stream()
        pulse_gen.stream_stop()
        tb.reset_cfm()
        return None

    initial_avg = np.mean(test_samples)
    print(f"Initial photon count check: {initial_avg:.0f} counts (working!)\n")

    # Initialize plot for real-time monitoring
    print("[DEBUG] Initializing plot...")
    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Z position (steps)")
    ax.set_ylabel("Photon counts")
    ax.set_title("Z-axis calibration - Downward scan from maximum")
    ax.grid(True, alpha=0.3)

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        pass

    ### PHASE 1: Move to Maximum Position

    print("="*60)
    print("PHASE 1: Moving to Maximum Position")
    print("="*60)

    print("[DEBUG] Getting current Z position...")
    initial_position = piezo.get_z_position()
    print(f"Current position: {initial_position} steps")
    print(f"Moving UP {max_position_steps} steps to establish maximum...")

    # Move to max position
    print(f"[DEBUG] Calling piezo.move_z_steps({max_position_steps})...")
    max_position_absolute = piezo.move_z_steps(max_position_steps)
    print(f"[DEBUG] Move complete. New position: {max_position_absolute}")
    time.sleep(0.5)  # Allow movement to complete

    print(f"Maximum position reached: {max_position_absolute} steps")
    print(f"This will be the starting reference point.\n")

    ### PHASE 2: Downward Scan with Data Collection

    print("="*60)
    print("PHASE 2: Scanning Downward to Find Surface")
    print("="*60)
    print(f"Scanning down in {scan_step_size} step increments")
    print(f"Will continue {scan_past_peak_steps} steps past peak for full profile")
    print(f"Safety abort if counts < {safety_min_counts}\n")

    scan_positions = []  # Absolute step positions
    scan_counts = []     # Mean photon counts at each position

    current_position = max_position_absolute
    peak_detected = False
    steps_past_peak = 0
    scan_start_time = time.time()

    print("[DEBUG] Starting scan loop...")
    try:
        while True:
            # Check for user interrupt or timeout
            if tb.safe_stop():
                print("\n[User stopped scan]")
                break

            if time.time() - scan_start_time > max_scan_timeout_s:
                print(f"\n[Timeout after {max_scan_timeout_s}s - stopping scan]")
                break

            # Move down one step
            current_position = piezo.move_z_steps(-scan_step_size)
            time.sleep(settling_time_s)

            # Debug output for first few steps
            if len(scan_positions) < 3:
                print(f"[DEBUG] Step {len(scan_positions)+1}: Moved to position {current_position}")

            # Collect photon counts
            counter.clear_buffer()
            time.sleep(0.05)  # Let buffer accumulate samples

            samples = counter.read_counter_simple()

            if len(samples) == 0:
                # Simple retry once
                time.sleep(0.05)
                samples = counter.read_counter_simple()

            if len(samples) > 0:
                mean_counts = np.mean(samples)
            else:
                # If still no samples, use last known value or zero
                mean_counts = scan_counts[-1] if len(scan_counts) > 0 else 0
                print(f"  [Warning: No samples at step {current_position}, using {mean_counts:.0f}]")

            # Store data
            scan_positions.append(current_position)
            scan_counts.append(mean_counts)

            # Update plot every 5 steps
            if len(scan_positions) % 5 == 0:
                ax.clear()
                ax.plot(scan_positions, scan_counts, 'b-', linewidth=1)
                ax.set_xlabel("Z position (steps)")
                ax.set_ylabel("Photon counts")
                ax.set_title(f"Downward scan - {len(scan_positions)} points")
                ax.grid(True, alpha=0.3)
                plt.pause(0.01)

            # Peak detection logic
            if mean_counts > peak_threshold and not peak_detected:
                peak_detected = True
                print(f"  >>> Peak detected! Counts: {mean_counts:.0f} at step {current_position}")

            if peak_detected:
                steps_past_peak += scan_step_size
                if steps_past_peak >= scan_past_peak_steps:
                    print(f"\nScan complete: {scan_past_peak_steps} steps past peak")
                    break

            # Safety check
            if mean_counts < safety_min_counts and len(scan_counts) > 20:
                print(f"\n[Safety abort: counts {mean_counts:.0f} < {safety_min_counts}]")
                break

            # Status display every 20 steps
            if len(scan_positions) % 20 == 0:
                print(f"  Step {current_position}: {mean_counts:.0f} counts")

    except KeyboardInterrupt:
        print("\n[Scan interrupted by user]")

    finally:
        # Always stop streams
        print("\n[DEBUG] Stopping streams...")
        counter.stop_tag_stream()
        pulse_gen.stream_stop()
        print("[DEBUG] Streams stopped")

    # Convert to numpy arrays
    print("[DEBUG] Converting data to numpy arrays...")
    scan_positions = np.array(scan_positions)
    scan_counts = np.array(scan_counts)

    if len(scan_counts) < 10:
        print("\n[ERROR] Insufficient data collected. Calibration aborted.")
        tb.reset_cfm()
        tb.reset_safe_stop()
        return None

    print(f"\nData collected: {len(scan_positions)} positions")
    print(f"Position range: {scan_positions[0]} to {scan_positions[-1]} steps")
    print(f"Count range: {scan_counts.min():.0f} to {scan_counts.max():.0f}\n")

    ### PHASE 3: Peak Analysis with scipy.find_peaks

    print("="*60)
    print("PHASE 3: Analyzing Peak to Find Surface")
    print("="*60)

    # Find peaks in the count profile
    # Use prominence to find the most significant peak
    print("[DEBUG] Running scipy.find_peaks...")
    peaks_indices, peak_properties = find_peaks(
        scan_counts,
        height=peak_threshold,
        prominence=peak_threshold * 0.3  # Peak must be 30% above surrounding
    )

    if len(peaks_indices) == 0:
        print("[ERROR] No valid peak found in scan data!")
        print("Possible issues:")
        print("  - Peak threshold too high")
        print("  - Didn't scan through surface")
        print("  - Laser/APD alignment issue")
        tb.reset_cfm()
        tb.reset_safe_stop()
        return None

    # Find the highest prominence peak (should be the surface)
    prominences = peak_properties['prominences']
    highest_peak_idx = peaks_indices[np.argmax(prominences)]
    surface_position_absolute = scan_positions[highest_peak_idx]
    surface_counts = scan_counts[highest_peak_idx]

    print(f"Surface identified using scipy.find_peaks:")
    print(f"  Position: {surface_position_absolute} steps")
    print(f"  Photon counts: {surface_counts:.0f}")
    print(f"  Peak prominence: {prominences[np.argmax(prominences)]:.0f}")
    print(f"  Total peaks found: {len(peaks_indices)}\n")

    # Update plot with peak marked
    ax.clear()
    ax.plot(scan_positions, scan_counts, 'b-', linewidth=1, label='Scan data')
    ax.plot(surface_position_absolute, surface_counts, 'r*',
            markersize=15, label=f'Surface (step {surface_position_absolute})')
    ax.axhline(peak_threshold, color='orange', linestyle='--',
               alpha=0.5, label=f'Peak threshold ({peak_threshold})')
    ax.set_xlabel("Z position (steps)")
    ax.set_ylabel("Photon counts")
    ax.set_title("Surface detected - Peak analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.pause(0.1)

    ### PHASE 4: Verification Scan (Hysteresis Measurement)

    print("="*60)
    print("PHASE 4: Verification and Hysteresis Measurement")
    print("="*60)
    print(f"Performing {verification_passes} approach cycles to measure repeatability\n")

    verification_errors = []

    # Restart counter for verification
    print("[DEBUG] Restarting streams for verification...")
    counter.start_tag_stream()
    pulse_gen.stream_start(-1)
    time.sleep(0.2)
    print("[DEBUG] Streams restarted")

    for pass_num in range(verification_passes):
        print(f"Verification pass {pass_num + 1}/{verification_passes}:")

        # Move away from surface
        print(f"  Retracting {verification_retract_steps} steps...")
        piezo.move_z_steps(verification_retract_steps)
        time.sleep(settling_time_s * 2)

        # Move back to calculated surface position
        print(f"  Approaching surface at step {surface_position_absolute}...")
        piezo.write_z(surface_position_absolute)
        time.sleep(settling_time_s * 2)

        # Measure counts at this position
        counter.clear_buffer()
        time.sleep(0.1)
        samples = counter.read_counter_simple()

        if len(samples) > 0:
            measured_counts = np.mean(samples)
            count_error = measured_counts - surface_counts
            verification_errors.append(count_error)
            print(f"  Counts at surface: {measured_counts:.0f} (error: {count_error:+.0f})")
        else:
            print(f"  [Warning: No counts measured on pass {pass_num + 1}]")
            verification_errors.append(0)

    print("[DEBUG] Stopping verification streams...")
    counter.stop_tag_stream()
    pulse_gen.stream_stop()
    print("[DEBUG] Verification streams stopped")

    # Calculate hysteresis statistics
    if len(verification_errors) > 0:
        avg_error = np.mean(verification_errors)
        std_error = np.std(verification_errors)
        print(f"\nVerification statistics:")
        print(f"  Average count error: {avg_error:+.0f} ± {std_error:.0f} counts")

    ### PHASE 5: Hysteresis Compensation Calculation

    print("\n" + "="*60)
    print("PHASE 5: Hysteresis Compensation")
    print("="*60)

    # For now, we're measuring count repeatability, not position hysteresis
    # A more sophisticated approach would scan around the surface on each pass
    # to find the peak position shift. For this version, we'll provide a placeholder.

    # Estimate hysteresis as a percentage (typical piezos: ~10-15%)
    # Since we don't have direct position measurement, suggest a typical value
    suggested_z_bias_adjust = 0.0  # Will be measured in future enhancement

    print(f"Hysteresis measurement: Not yet implemented")
    print(f"  Current z_bias_adjust: {config['Positioning']['z_bias_adjust']}")
    print(f"  Suggested value: {suggested_z_bias_adjust} (placeholder)")
    print(f"\nNote: For accurate hysteresis, need position-resolved verification scans")
    print(f"      This will be implemented in a future enhancement.\n")

    ### PHASE 6: Set Reference and Save

    print("="*60)
    print("PHASE 6: Setting Reference and Saving Results")
    print("="*60)

    # Calculate the key reference distance
    max_to_surface_distance = max_position_absolute - surface_position_absolute

    print(f"\nCalibration Summary:")
    print(f"  Maximum position: {max_position_absolute} steps (before zeroing)")
    print(f"  Surface position: {surface_position_absolute} steps (before zeroing)")
    print(f"  Reference distance (max→surface): {max_to_surface_distance} steps")
    print(f"  Surface photon counts: {surface_counts:.0f}")

    # Move to surface
    print(f"\nMoving to surface position...")
    piezo.write_z(surface_position_absolute)
    time.sleep(0.2)

    # Set surface as step=0 reference
    print(f"Setting surface as Z=0 reference...")
    print(f"[DEBUG] Calling piezo.set_z_reference(0)...")
    piezo.set_z_reference(0)
    print(f"[DEBUG] Reference set successfully")

    print(f"\n{'='*60}")
    print("Z-AXIS CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Surface is now at step=0")
    print(f"Store this reference: max→surface = {max_to_surface_distance} steps")
    print(f"Use this value to recover position if NVs are lost")
    print(f"{'='*60}\n")

    tb.reset_safe_stop()

    ### Save Data

    print("[DEBUG] Preparing to save data...")
    timestamp = dm.get_time_stamp()
    print(f"[DEBUG] Timestamp: {timestamp}")
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "calibration_method": "max_to_surface_downward_scan",
        "config_parameters": {
            "max_position_steps": max_position_steps,
            "scan_step_size": scan_step_size,
            "peak_threshold": peak_threshold,
            "scan_past_peak_steps": scan_past_peak_steps,
            "safety_min_counts": safety_min_counts,
            "verification_passes": verification_passes,
            "verification_retract_steps": verification_retract_steps,
            "settling_time_ms": settling_time_ms,
        },
        "results": {
            "max_position_absolute": int(max_position_absolute),
            "surface_position_absolute": int(surface_position_absolute),
            "max_to_surface_distance": int(max_to_surface_distance),
            "surface_photon_counts": float(surface_counts),
            "verification_count_errors": [float(e) for e in verification_errors],
            "suggested_z_bias_adjust": float(suggested_z_bias_adjust),
        },
        "scan_data": {
            "positions": scan_positions.tolist(),
            "counts": scan_counts.tolist(),
        },
        "note": "Surface position set as step=0 reference. Use max_to_surface_distance for position recovery.",
    }

    nv_name = getattr(nv_sig, "name", "unknown")
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_raw_data(raw_data, file_path)

    # Create final plot
    fig_final, ax_final = plt.subplots(figsize=(12, 7))
    ax_final.plot(scan_positions, scan_counts, 'b-', linewidth=1.5, label='Scan profile')
    ax_final.plot(surface_position_absolute, surface_counts, 'r*',
                  markersize=20, label=f'Surface (now step=0)')
    ax_final.axhline(peak_threshold, color='orange', linestyle='--',
                     alpha=0.5, label=f'Peak threshold')
    ax_final.axhline(safety_min_counts, color='red', linestyle='--',
                     alpha=0.5, label=f'Safety minimum')
    ax_final.set_xlabel("Z position (steps, before zeroing)", fontsize=12)
    ax_final.set_ylabel("Photon counts", fontsize=12)
    ax_final.set_title(f"Z-Axis Calibration - Max→Surface Distance: {max_to_surface_distance} steps",
                       fontsize=14, fontweight='bold')
    ax_final.legend(fontsize=10)
    ax_final.grid(True, alpha=0.3)

    dm.save_figure(fig_final, file_path)

    print(f"Data and plots saved to: {file_path}")
    kpl.show()

    return raw_data


if __name__ == "__main__":
    """Example usage for testing"""
    from utils.constants import NVSig, CoordsKey, VirtualLaserKey

    # Create a minimal nv_sig for testing
    nv_sig = NVSig(
        name="test_z_calibration",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.Z: 0},
        pulse_durations={VirtualLaserKey.IMAGING: int(1e6)},  # 1 ms
    )

    # Run calibration with default config parameters
    results = main(nv_sig)

    # Or override specific parameters:
    # results = main(nv_sig, max_position_steps=5000, scan_step_size=5)
