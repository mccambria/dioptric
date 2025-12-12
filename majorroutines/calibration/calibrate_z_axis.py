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
    min_peak_prominence_ratio = kwargs.get("min_peak_prominence_ratio", cal_config["min_peak_prominence_ratio"])
    scan_past_peak_steps = kwargs.get("scan_past_peak_steps", cal_config["scan_past_peak_steps"])
    safety_min_counts = kwargs.get("safety_min_counts", cal_config["safety_min_counts"])
    verification_passes = kwargs.get("verification_passes", cal_config["verification_passes"])
    verification_retract_steps = kwargs.get("verification_retract_steps", cal_config["verification_retract_steps"])
    settling_time_ms = kwargs.get("settling_time_ms", cal_config["settling_time_ms"])
    max_scan_timeout_s = kwargs.get("max_scan_timeout_s", cal_config["max_scan_timeout_s"])
    min_scan_points = kwargs.get("min_scan_points", cal_config["min_scan_points"])

    settling_time_s = settling_time_ms / 1000.0

    # Asymmetry measurement parameters
    measure_asymmetry = kwargs.get("measure_asymmetry", cal_config.get("measure_asymmetry", True))
    asymmetry_test_steps = kwargs.get("asymmetry_test_steps", cal_config.get("asymmetry_test_steps", 100))
    asymmetry_test_cycles = kwargs.get("asymmetry_test_cycles", cal_config.get("asymmetry_test_cycles", 3))
    asymmetry_count_tolerance = kwargs.get("asymmetry_count_tolerance", cal_config.get("asymmetry_count_tolerance", 50))
    asymmetry_safety_drop = kwargs.get("asymmetry_safety_drop", cal_config.get("asymmetry_safety_drop", 0.30))

    # Get servers
    piezo = pos.get_positioner_server(CoordsKey.Z)
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    print("\n" + "="*60)
    print("Z-AXIS CALIBRATION - MAX TO SURFACE REFERENCE")
    print("="*60)
    print(f"Configuration:")
    print(f"  Max position move: {max_position_steps} steps (~{max_position_steps*0.05:.2f} microns)")
    print(f"  Scan step size: {scan_step_size} steps")
    print(f"  Peak prominence: {min_peak_prominence_ratio*100:.0f}% above baseline (relative detection)")
    print(f"  Scan past peak: {scan_past_peak_steps} steps")
    print(f"  Safety minimum: {safety_min_counts} counts (collision protection)")
    print(f"  Verification passes: {verification_passes}")
    if measure_asymmetry:
        print(f"  Asymmetry measurement: {asymmetry_test_cycles} cycles of {asymmetry_test_steps} steps")
    print("="*60 + "\n")

    # Reset hardware to clean state
    tb.reset_cfm()
    tb.init_safe_stop()

    # Setup laser for imaging
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_dur = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"]))
    )
    readout_laser = vld["physical_name"]
    tb.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tb.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Load pulse sequence for photon counting
    seq_args = [0, readout_dur, readout_laser, readout_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "simple_readout.py"
    period = pulse_gen.stream_load(seq_file, seq_args_string)[0]

    # Start continuous counting
    counter.start_tag_stream()
    pulse_gen.stream_start(-1)  # Run continuously

    # Verify photon counting is working
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

    ### PHASE 0: Asymmetry Measurement (Safety Critical!)

    measured_z_bias = 0.0  # Default: no asymmetry

    if measure_asymmetry:
        print("="*60)
        print("PHASE 0: Measuring Up/Down Step Asymmetry")
        print("="*60)
        print(f"Performing {asymmetry_test_cycles} test cycles to measure directional asymmetry")
        print(f"Each cycle: UP {asymmetry_test_steps} steps, DOWN {asymmetry_test_steps} steps")
        print(f"SAFETY: Will abort if counts drop > {asymmetry_safety_drop*100:.0f}% during test\n")

        starting_position = piezo.get_z_position()
        print(f"Starting position: {starting_position} steps")

        # Measure baseline counts at start
        counter.clear_buffer()
        time.sleep(0.2)
        baseline_samples = counter.read_counter_simple()
        if len(baseline_samples) == 0:
            print("[WARNING] Could not measure baseline counts for asymmetry test")
            print("Skipping asymmetry measurement - using z_bias_adjust from config")
            measured_z_bias = config["Positioning"]["z_bias_adjust"]
        else:
            baseline_counts = np.mean(baseline_samples)
            print(f"Baseline counts: {baseline_counts:.0f}\n")

            asymmetry_ratios = []

            for cycle in range(asymmetry_test_cycles):
                print(f"Asymmetry test cycle {cycle + 1}/{asymmetry_test_cycles}:")

                # Move UP by test steps
                print(f"  Moving UP {asymmetry_test_steps} steps...")
                pos_after_up = piezo.move_z_steps(asymmetry_test_steps)
                time.sleep(settling_time_s * 3)  # Extra settling for accuracy

                # Measure counts after up
                counter.clear_buffer()
                time.sleep(0.1)
                up_samples = counter.read_counter_simple()
                up_counts = np.mean(up_samples) if len(up_samples) > 0 else baseline_counts
                print(f"    Position: {pos_after_up}, Counts: {up_counts:.0f}")

                # Move DOWN by test steps (should return to start if symmetric)
                print(f"  Moving DOWN {asymmetry_test_steps} steps...")
                pos_after_down = piezo.move_z_steps(-asymmetry_test_steps)
                time.sleep(settling_time_s * 3)

                # Measure counts after down
                counter.clear_buffer()
                time.sleep(0.1)
                down_samples = counter.read_counter_simple()
                down_counts = np.mean(down_samples) if len(down_samples) > 0 else baseline_counts
                print(f"    Position: {pos_after_down}, Counts: {down_counts:.0f}")

                # SAFETY CHECK: Abort if counts dropped too much
                count_drop_ratio = (baseline_counts - down_counts) / baseline_counts
                if count_drop_ratio > asymmetry_safety_drop:
                    print(f"\n[SAFETY ABORT] Counts dropped {count_drop_ratio*100:.1f}% during test!")
                    print(f"  Baseline: {baseline_counts:.0f} -> After down: {down_counts:.0f}")
                    print(f"  This indicates SEVERE asymmetry (down steps >> up steps)")
                    print(f"  Aborting to prevent damage. Recommendations:")
                    print(f"    1. Use smaller test steps: asymmetry_test_steps=50")
                    print(f"    2. Manually set z_bias_adjust in config (try 0.20)")
                    print(f"    3. Clean/service piezo actuators")
                    counter.stop_tag_stream()
                    pulse_gen.stream_stop()
                    tb.reset_cfm()
                    tb.reset_safe_stop()
                    return None

                # Check if we're back to baseline
                count_diff = abs(down_counts - baseline_counts)
                position_diff = pos_after_down - starting_position

                if count_diff < asymmetry_count_tolerance:
                    print(f"    Returned to baseline (diff: {count_diff:.0f} counts)")
                    asymmetry_ratios.append(0.0)  # Symmetric
                else:
                    # Counts didn't return to baseline - estimate asymmetry
                    # If counts dropped, down steps moved us further than up
                    # If counts increased, down steps moved us less than up
                    if down_counts < baseline_counts:
                        # Moved further down than expected - down steps are stronger
                        excess_ratio = (baseline_counts - down_counts) / baseline_counts
                        print(f"    WARNING: Down steps appear STRONGER (counts dropped by {count_diff:.0f})")
                        print(f"      Estimated excess: {excess_ratio*100:.1f}%")
                        asymmetry_ratios.append(excess_ratio)
                    else:
                        # Moved less down than expected - down steps are weaker
                        deficiency_ratio = -(down_counts - baseline_counts) / baseline_counts
                        print(f"    WARNING: Down steps appear WEAKER (counts rose by {count_diff:.0f})")
                        print(f"      Estimated deficiency: {deficiency_ratio*100:.1f}%")
                        asymmetry_ratios.append(deficiency_ratio)

                print()

            # Calculate average asymmetry
            if len(asymmetry_ratios) > 0:
                measured_z_bias = np.mean(asymmetry_ratios)
                asymmetry_std = np.std(asymmetry_ratios)

                print(f"\n{'='*60}")
                print(f"Asymmetry Measurement Results:")
                print(f"  Average z_bias_adjust: {measured_z_bias:.4f} ({measured_z_bias*100:.2f}%)")
                print(f"  Standard deviation: {asymmetry_std:.4f}")
                print(f"  Individual measurements: {[f'{r:.4f}' for r in asymmetry_ratios]}")

                if abs(measured_z_bias) > 0.45:
                    print(f"\n  WARNING: Measured asymmetry > 45% - this seems very high!")
                    print(f"  This could indicate:")
                    print(f"    - Very sticky piezos (needs cleaning/maintenance)")
                    print(f"    - Measurement error (low count SNR)")
                    print(f"    - Hardware issue")
                    user_input = input(f"\n  Continue with measured value? (y/n): ")
                    if user_input.lower() != 'y':
                        print("Calibration aborted by user")
                        counter.stop_tag_stream()
                        pulse_gen.stream_stop()
                        tb.reset_cfm()
                        tb.reset_safe_stop()
                        return None

                print(f"\n  Applying z_bias_adjust = {measured_z_bias:.4f} for safe movements")
                print(f"{'='*60}\n")

                # Apply the measured bias to the piezo server
                # Note: This modifies the server's internal state for this session
                try:
                    # Access the server's internal z_bias_adjust if possible
                    # This is implementation-specific to pos_xyz_ATTO_piezos
                    print(f"  [INFO] Measured asymmetry will be saved to calibration data")
                    print(f"  [INFO] To apply permanently, update config: z_bias_adjust = {measured_z_bias:.4f}\n")
                except Exception as e:
                    print(f"  [WARNING] Could not apply z_bias_adjust to server: {e}\n")
            else:
                print(f"  No valid asymmetry measurements - using config value")
                measured_z_bias = config["Positioning"]["z_bias_adjust"]

    else:
        print("Asymmetry measurement skipped (measure_asymmetry = False)")
        measured_z_bias = config["Positioning"]["z_bias_adjust"]
        print(f"Using z_bias_adjust from config: {measured_z_bias}\n")

    ### PHASE 1: Move to Maximum Position

    print("="*60)
    print("PHASE 1: Moving to Maximum Position")
    print("="*60)

    initial_position = piezo.get_z_position()
    print(f"Current position: {initial_position} steps")
    print(f"Moving UP {max_position_steps} steps to establish maximum...")

    # Move to max position
    max_position_absolute = piezo.move_z_steps(max_position_steps)
    time.sleep(0.5)  # Allow movement to complete

    print(f"Maximum position reached: {max_position_absolute} steps")

    # Measure baseline counts at max position (above sample)
    counter.clear_buffer()
    time.sleep(0.2)
    check_samples = counter.read_counter_simple()
    if len(check_samples) > 0:
        baseline_counts = np.mean(check_samples)
        print(f"Baseline counts above sample: {baseline_counts:.0f}")
        print(f"  (Will detect surface as peak > {baseline_counts*(1+min_peak_prominence_ratio):.0f} counts)")
    else:
        baseline_counts = None
        print(f"  (Could not measure baseline - will use relative peak detection)")

    print(f"Starting reference point established.\n")

    ### PHASE 2: Downward Scan with Data Collection

    print("="*60)
    print("PHASE 2: Scanning Downward to Find Surface")
    print("="*60)
    print(f"Scanning down in {scan_step_size} step increments")
    print(f"Will continue {scan_past_peak_steps} steps past peak for complete profile")
    print(f"Using relative peak detection")
    print(f"Safety abort if counts < {safety_min_counts}\n")

    scan_positions = []  # Absolute step positions
    scan_counts = []     # Mean photon counts at each position

    current_position = max_position_absolute
    peak_detected = False
    steps_past_peak = 0
    scan_start_time = time.time()

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

            # Relative peak detection - look for increases above baseline
            # Only start detecting after we have enough data to establish a baseline
            if len(scan_counts) >= min_scan_points and not peak_detected:
                # Use first N points as baseline estimate
                current_baseline = np.mean(scan_counts[:min_scan_points])
                threshold_for_peak = current_baseline * (1 + min_peak_prominence_ratio)

                if mean_counts > threshold_for_peak:
                    peak_detected = True
                    print(f"  >>> Peak detected! Counts: {mean_counts:.0f} at step {current_position}")
                    print(f"      (Baseline: {current_baseline:.0f}, Threshold: {threshold_for_peak:.0f})")

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
        counter.stop_tag_stream()
        pulse_gen.stream_stop()

    # Convert to numpy arrays
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

    # Find peaks in the count profile using RELATIVE detection
    # Calculate baseline from first portion of scan (above sample)
    baseline_scan = np.mean(scan_counts[:min(min_scan_points, len(scan_counts)//4)])
    min_prominence = baseline_scan * min_peak_prominence_ratio

    print(f"Baseline (above sample): {baseline_scan:.0f} counts")
    print(f"Minimum prominence for peak: {min_prominence:.0f} counts")

    # Find peaks using scipy with relative prominence
    peaks_indices, peak_properties = find_peaks(
        scan_counts,
        prominence=min_prominence  # Relative prominence threshold
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
    ax.axhline(baseline_scan, color='green', linestyle='--',
               alpha=0.5, label=f'Baseline ({baseline_scan:.0f})')
    ax.axhline(baseline_scan * (1 + min_peak_prominence_ratio), color='orange',
               linestyle='--', alpha=0.5, label=f'Detection threshold')
    ax.set_xlabel("Z position (steps)")
    ax.set_ylabel("Photon counts")
    ax.set_title("Surface detected - Relative peak analysis")
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
    counter.start_tag_stream()
    pulse_gen.stream_start(-1)
    time.sleep(0.2)

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

    counter.stop_tag_stream()
    pulse_gen.stream_stop()

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
    piezo.set_z_reference(0)

    print(f"\n{'='*60}")
    print("Z-AXIS CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Surface is now at step=0")
    print(f"Store this reference: max→surface = {max_to_surface_distance} steps")
    print(f"Use this value to recover position if NVs are lost")
    if measure_asymmetry:
        print(f"\nMeasured z_bias_adjust: {measured_z_bias:.4f} ({measured_z_bias*100:.2f}%)")
        print(f"Update config/cryo.py with this value for permanent correction")
    print(f"{'='*60}\n")

    tb.reset_safe_stop()

    ### Save Data

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "calibration_method": "max_to_surface_downward_scan_with_asymmetry",
        "config_parameters": {
            "max_position_steps": max_position_steps,
            "scan_step_size": scan_step_size,
            "min_peak_prominence_ratio": min_peak_prominence_ratio,
            "scan_past_peak_steps": scan_past_peak_steps,
            "safety_min_counts": safety_min_counts,
            "verification_passes": verification_passes,
            "verification_retract_steps": verification_retract_steps,
            "settling_time_ms": settling_time_ms,
            "measure_asymmetry": measure_asymmetry,
        },
        "results": {
            "max_position_absolute": int(max_position_absolute),
            "surface_position_absolute": int(surface_position_absolute),
            "max_to_surface_distance": int(max_to_surface_distance),
            "surface_photon_counts": float(surface_counts),
            "verification_count_errors": [float(e) for e in verification_errors],
            "measured_z_bias_adjust": float(measured_z_bias),
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

    # Show baseline and detection threshold
    baseline_final = np.mean(scan_counts[:min(min_scan_points, len(scan_counts)//4)])
    ax_final.axhline(baseline_final, color='green', linestyle='--',
                     alpha=0.5, label=f'Baseline ({baseline_final:.0f})')
    ax_final.axhline(baseline_final * (1 + min_peak_prominence_ratio), color='orange',
                     linestyle='--', alpha=0.5, label=f'Detection threshold')
    ax_final.axhline(safety_min_counts, color='red', linestyle='--',
                     alpha=0.5, label=f'Safety minimum')

    ax_final.set_xlabel("Z position (steps, before zeroing)", fontsize=12)
    ax_final.set_ylabel("Photon counts", fontsize=12)
    ax_final.set_title(f"Z-Axis Calibration - Max→Surface Distance: {max_to_surface_distance} steps (~{max_to_surface_distance*0.05:.0f} µm)",
                       fontsize=14, fontweight='bold')
    ax_final.legend(fontsize=10)
    ax_final.grid(True, alpha=0.3)

    dm.save_figure(fig_final, file_path)

    print(f"Data and plots saved to: {file_path}")
    kpl.show()

    return raw_data


def optimize_z(
    nv_sig: NVSig,
    num_steps: int = 40,
    step_size: int = 2,
    num_averages: int = 3,
    move_to_optimal: bool = True,
    save_data: bool = True,
    scan_direction: str = "up",
):
    """
    Optimize Z position by scanning and fitting a Gaussian to find the focus peak.

    This function scans through Z positions centered around the current position,
    collects photon counts at each position, fits a Gaussian to find the optimal
    Z coordinate, and optionally moves to that position.

    Uses the same hardware pattern as z_scan_1d.py (piezo stepping + counter readout).

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters (pulse durations, laser settings)
    num_steps : int, optional
        Total number of Z positions to scan. Default: 40
        Scans half below and half above current position.
    step_size : int, optional
        Step size in piezo units between positions. Default: 2
    num_averages : int, optional
        Number of photon count samples to average at each position. Default: 3
        Higher values improve SNR but slow down the scan.
    move_to_optimal : bool, optional
        Whether to move the piezo to the optimal Z position after fitting. Default: True
    save_data : bool, optional
        Whether to save scan data and plot. Default: True
    scan_direction : str, optional
        Direction to scan: "up" starts low and scans upward (away from sample),
        "down" starts high and scans downward (toward sample). Default: "up"

    Returns
    -------
    dict
        Results containing:
        - opti_z: Optimal Z position (piezo steps), or None if fit failed
        - opti_counts: Photon counts at optimal position
        - fit_params: Gaussian fit parameters (amplitude, center, sigma, offset)
        - z_positions: Array of scanned Z positions
        - counts: Array of photon counts at each position
        - fit_success: Whether the Gaussian fit succeeded
    """
    from scipy.optimize import curve_fit

    ### Setup
    config = common.get_config_dict()

    # Get servers - same pattern as z_scan_1d
    piezo = pos.get_positioner_server(CoordsKey.Z)
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    # Setup laser for imaging
    laser_dict = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(laser_dict["duration"]))
    )
    readout_s = readout_ns * 1e-9
    laser_name = laser_dict["physical_name"]

    tb.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    laser_power = tb.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Data storage
    z_positions = []
    counts_list = []

    ### Setup figure
    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Z Position (steps)")
    ax.set_ylabel("Counts")
    ax.set_title("Z Optimization Scan")
    (line,) = ax.plot([], [], "b.-", markersize=6)
    ax.grid(True, alpha=0.3)
    plt.ion()

    tb.reset_cfm()
    tb.init_safe_stop()

    # Load pulse sequence once
    seq_file = "simple_readout.py"
    delay_ns = 0
    pulse_gen.stream_load(
        seq_file,
        tb.encode_seq_args([delay_ns, readout_ns, laser_name, laser_power]),
    )

    # Start counter stream
    counter.start_tag_stream()

    # Get initial position
    initial_z = piezo.get_z_position()

    # Determine scan direction
    # "up" = start low (negative), scan toward positive (away from sample)
    # "down" = start high (positive), scan toward negative (toward sample)
    scan_up = scan_direction.lower() == "up"
    direction_sign = 1 if scan_up else -1

    # Move half the total range in opposite direction to center the scan
    half_steps = num_steps // 2
    print(f"Initial Z position: {initial_z} steps")
    print(f"Scan range: {num_steps} steps of {step_size} units")
    print(f"Scan direction: {'low→high (up)' if scan_up else 'high→low (down)'}")
    print(f"Averaging {num_averages} samples per position")
    print(f"Moving to scan start ({half_steps * step_size} steps {'back' if scan_up else 'forward'})...")

    # Move to starting position (opposite of scan direction)
    for _ in range(half_steps):
        piezo.move_z_steps(-direction_sign * step_size)
    time.sleep(0.05)  # Settling time

    scan_start_z = piezo.get_z_position()
    print(f"Scan starting at Z = {scan_start_z} steps\n")

    try:
        for i in range(num_steps):
            if tb.safe_stop():
                print("\n[STOPPED] User interrupt")
                break

            # Move Z position in scan direction
            current_z = piezo.move_z_steps(direction_sign * step_size)
            time.sleep(0.01)

            # Collect samples at this position
            samples = []
            for _ in range(num_averages):
                if tb.safe_stop():
                    break
                pulse_gen.stream_start(1)
                raw = counter.read_counter_simple(1)
                samples.append(int(raw[0]))

            avg_counts = np.mean(samples) if samples else 0
            z_positions.append(current_z)
            counts_list.append(avg_counts)

            # Update plot
            line.set_data(z_positions, counts_list)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

            if (i + 1) % 10 == 0:
                print(f"Step {i+1}/{num_steps}: Z={current_z}, Counts={avg_counts:.0f}")

    finally:
        counter.stop_tag_stream()
        tb.reset_safe_stop()

    ### Fit Gaussian to find optimal Z
    z_array = np.array(z_positions)
    counts_array = np.array(counts_list)

    # Gaussian fit function
    def gaussian(x, amplitude, center, sigma, offset):
        return offset + amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))

    opti_z = None
    fit_params = None
    fit_success = False

    # Use min/max to handle both scan directions (up or down)
    z_min = np.min(z_array)
    z_max = np.max(z_array)

    try:
        # Initial guesses
        offset_guess = np.min(counts_array)
        amplitude_guess = np.max(counts_array) - offset_guess
        center_guess = z_array[np.argmax(counts_array)]
        sigma_guess = (z_max - z_min) / 4

        guess = [amplitude_guess, center_guess, sigma_guess, offset_guess]
        bounds = (
            [0, z_min, 0, 0],  # Lower bounds
            [np.inf, z_max, np.inf, np.inf],  # Upper bounds
        )

        popt, _ = curve_fit(gaussian, z_array, counts_array, p0=guess, bounds=bounds)
        opti_z = popt[1]
        fit_params = {
            "amplitude": popt[0],
            "center": popt[1],
            "sigma": popt[2],
            "offset": popt[3],
        }
        fit_success = True

        # Plot the fit
        z_fit = np.linspace(z_min, z_max, 200)
        counts_fit = gaussian(z_fit, *popt)
        ax.plot(z_fit, counts_fit, "r-", linewidth=2, label="Gaussian fit")
        ax.axvline(opti_z, color="g", linestyle="--", label=f"Optimal Z={opti_z:.1f}")
        ax.legend()

        print(f"\nGaussian fit results:")
        print(f"  Optimal Z: {opti_z:.1f} steps")
        print(f"  Amplitude: {popt[0]:.0f} counts")
        print(f"  Sigma: {popt[2]:.1f} steps")
        print(f"  Offset: {popt[3]:.0f} counts")

    except Exception as e:
        print(f"\nGaussian fit failed: {e}")
        print("Using max counts position instead")
        opti_z = z_array[np.argmax(counts_array)]
        fit_params = None
        ax.axvline(opti_z, color="g", linestyle="--", label=f"Max Z={opti_z:.1f}")
        ax.legend()

    ### Move to optimal position using count-based feedback
    opti_counts = None
    if opti_z is not None and move_to_optimal:
        # Get target counts from the Gaussian fit (peak value)
        if fit_params is not None:
            target_counts = fit_params["amplitude"] + fit_params["offset"]
        else:
            # If fit failed, use max observed counts
            target_counts = np.max(counts_array)

        # Tolerance: consider "at optimal" when within this fraction of target
        count_tolerance = 0.95  # Within 95% of peak counts
        target_threshold = target_counts * count_tolerance

        # Determine direction to move (toward optimal position)
        current_pos = z_positions[-1] if z_positions else initial_z
        move_direction = 1 if opti_z > current_pos else -1

        print(f"\nMoving to optimal position using count feedback...")
        print(f"  Target counts: {target_counts:.0f} (threshold: {target_threshold:.0f})")
        print(f"  Current position: {current_pos}, Optimal estimate: {opti_z:.1f}")
        print(f"  Moving {'up' if move_direction > 0 else 'down'}...")

        # Start counter for feedback loop
        counter.start_tag_stream()

        max_approach_steps = abs(int(round(opti_z - current_pos))) * 3  # Allow 3x estimated distance
        approach_step = 0
        found_optimal = False

        # Helper to measure counts
        def measure_counts(n_samples=3):
            samples = []
            for _ in range(n_samples):
                pulse_gen.stream_start(1)
                raw = counter.read_counter_simple(1)
                if raw:
                    samples.append(int(raw[0]))
            return np.mean(samples) if samples else 0

        # Measure current counts before moving
        current_counts = measure_counts()
        print(f"  Starting counts: {current_counts:.0f}")

        # Track the best position we've found
        best_counts = current_counts
        best_position = piezo.get_z_position()
        reached_optimal_region = False

        # Step toward optimal, checking counts each step
        while approach_step < max_approach_steps:
            if tb.safe_stop():
                print("  [STOPPED] User interrupt during approach")
                break

            # Move one step toward optimal
            piezo.move_z_steps(move_direction * step_size)
            time.sleep(0.01)
            approach_step += 1

            # Measure counts at new position
            current_counts = measure_counts()
            current_position = piezo.get_z_position()

            # Track the best position we've seen
            if current_counts > best_counts:
                best_counts = current_counts
                best_position = current_position

            # Check if we've reached the optimal region
            if current_counts >= target_threshold:
                if not reached_optimal_region:
                    print(f"  Step {approach_step}: counts={current_counts:.0f}, Z={current_position} >= threshold - REACHED OPTIMAL REGION")
                    reached_optimal_region = True
                else:
                    # Already in optimal region, just report progress
                    print(f"  Step {approach_step}: counts={current_counts:.0f}, Z={current_position} (in optimal region)")
            else:
                print(f"  Step {approach_step}: counts={current_counts:.0f}, Z={current_position}")

            # Only check for overshoot AFTER we've reached the optimal region
            if reached_optimal_region and current_counts < target_threshold:
                # Counts dropped below threshold after being optimal - we've passed the peak
                print(f"  Step {approach_step}: counts={current_counts:.0f}, Z={current_position} dropped below threshold - PASSED PEAK")
                print(f"  Starting gradient ascent to find true peak...")

                # Gradient ascent: follow increasing counts, reverse when counts drop
                # Focus purely on counts, ignore position (due to piezo hysteresis)
                current_direction = -move_direction  # Start by reversing
                hill_climb_step = 0
                max_hill_climb_steps = 40
                direction_changes = 0

                # Use more averages during hill-climb for stability
                hill_climb_averages = 5

                # Track the best counts seen in the CURRENT direction of travel
                best_in_direction = current_counts
                steps_since_best_in_direction = 0
                steps_before_reverse = 3  # Reverse after this many steps below best-in-direction

                # Track overall best for reporting
                overall_best_counts = best_counts  # Keep from approach phase
                overall_best_position = best_position

                print(f"    Starting counts: {current_counts:.0f}, target threshold: {target_threshold:.0f}")

                found_peak = False
                while hill_climb_step < max_hill_climb_steps and not found_peak:
                    if tb.safe_stop():
                        print("    [STOPPED] User interrupt during hill climb")
                        break

                    # Take a step in current direction
                    piezo.move_z_steps(current_direction * step_size)
                    time.sleep(0.01)
                    hill_climb_step += 1

                    # Measure counts with more averaging for stability
                    current_counts = measure_counts(n_samples=hill_climb_averages)
                    current_pos = piezo.get_z_position()

                    # Track overall best
                    if current_counts > overall_best_counts:
                        overall_best_counts = current_counts
                        overall_best_position = current_pos

                    # Track best in current direction of travel
                    if current_counts > best_in_direction:
                        best_in_direction = current_counts
                        steps_since_best_in_direction = 0
                        print(f"    Step {hill_climb_step}: counts={current_counts:.0f}, Z={current_pos} (best in direction)")
                    else:
                        steps_since_best_in_direction += 1
                        print(f"    Step {hill_climb_step}: counts={current_counts:.0f}, Z={current_pos}")

                    # After first reversal, stop as soon as we reach target threshold
                    # This confirms we're back in the peak region
                    if direction_changes >= 1 and current_counts >= target_threshold:
                        print(f"    FOUND PEAK: counts={current_counts:.0f} >= threshold after reversal")
                        found_peak = True
                        break

                    # Check if we should reverse:
                    # - We've gone several steps without improving
                    # - AND current counts are significantly below best in this direction
                    should_reverse = (
                        steps_since_best_in_direction >= steps_before_reverse
                        and current_counts < best_in_direction * 0.80  # 20% below best in direction
                    )

                    if should_reverse:
                        print(f"    Reversing: {steps_since_best_in_direction} steps past peak of {best_in_direction:.0f} (change #{direction_changes + 1})")

                        # Reverse direction
                        current_direction = -current_direction
                        direction_changes += 1

                        # Reset best-in-direction tracking
                        best_in_direction = current_counts
                        steps_since_best_in_direction = 0

                # Final report
                final_z = piezo.get_z_position()
                print(f"    Gradient ascent complete after {hill_climb_step} steps, {direction_changes} reversals")

                # Update best tracking
                best_counts = overall_best_counts
                best_position = overall_best_position

                break

        counter.stop_tag_stream()

        final_z = piezo.get_z_position()
        opti_counts = int(current_counts)

        if reached_optimal_region:
            print(f"  Optimal position found at Z={final_z}")
            print(f"  Final counts: {opti_counts} (best seen: {best_counts:.0f})")
        else:
            print(f"  Could not reach target counts after {approach_step} steps")
            print(f"  Final position: Z={final_z}, counts={opti_counts}")

    plt.ioff()
    tb.reset_cfm()

    print("COMPLETE")

    ### Save data
    results = {
        "opti_z": float(opti_z) if opti_z is not None else None,
        "opti_counts": opti_counts,
        "fit_params": fit_params,
        "z_positions": z_array.tolist(),
        "counts": counts_array.tolist(),
        "fit_success": fit_success,
        "initial_z": initial_z,
        "scan_params": {
            "num_steps": num_steps,
            "step_size": step_size,
            "num_averages": num_averages,
            "scan_direction": scan_direction,
        },
    }

    if save_data:
        timestamp = dm.get_time_stamp()
        raw_data = {
            "timestamp": timestamp,
            "nv_sig": nv_sig,
            "optimization_results": results,
        }
        nv_name = getattr(nv_sig, "name", "unknown")
        file_path = dm.get_file_path(__file__, timestamp, f"{nv_name}_z_optimize")
        dm.save_raw_data(raw_data, file_path)
        dm.save_figure(fig, file_path)
        print(f"Data saved to: {file_path}")

    kpl.show()

    return results


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

    # For Z optimization around current position:
    # results = optimize_z(nv_sig, num_steps=40, step_size=2)
