# -*- coding: utf-8 -*-
"""
Z-axis diagnostic - Step upward and observe count profile

Simple tool to visualize photon counts while moving up to find the surface
and determine how far above it you need to go.

Created on November 7th, 2025
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from utils import common
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, NVSig, VirtualLaserKey


def main(nv_sig, step_size=10, max_steps=50000, update_interval=20):
    """
    Step upward and plot photon counts in real-time to diagnose Z-axis behavior.

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters
    step_size : int
        Step increment (default: 10)
    max_steps : int
        Maximum total steps to move (default: 50000)
    update_interval : int
        Update plot every N steps (default: 20)

    Returns
    -------
    dict
        Position and count data
    """

    # Get servers
    piezo = pos.get_positioner_server(CoordsKey.Z)
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    print("\n" + "="*60)
    print("Z-AXIS DIAGNOSTIC - UPWARD SCAN")
    print("="*60)
    print(f"Step size: {step_size} steps")
    print(f"Max distance: {max_steps} steps (~{max_steps*0.05:.0f} microns)")
    print(f"Press CTRL+C to stop at any time")
    print("="*60 + "\n")

    # Setup
    tb.reset_cfm()
    tb.init_safe_stop()

    # Setup laser
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_dur = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"]))
    )
    readout_laser = vld["physical_name"]
    tb.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tb.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Load pulse sequence
    seq_args = [0, readout_dur, readout_laser, readout_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    seq_file = "simple_readout.py"
    period = pulse_gen.stream_load(seq_file, seq_args_string)[0]

    # Start counting
    counter.start_tag_stream()
    pulse_gen.stream_start(-1)

    # Verify counting works
    time.sleep(0.2)
    test_samples = counter.read_counter_simple()
    if len(test_samples) == 0:
        print("[ERROR] No photon counts detected!")
        counter.stop_tag_stream()
        pulse_gen.stream_stop()
        tb.reset_cfm()
        return None

    initial_avg = np.mean(test_samples)
    print(f"Initial photon count check: {initial_avg:.0f} counts (working!)\n")

    # Get starting position
    start_position = piezo.get_z_position()
    print(f"Starting position: {start_position} steps\n")
    print("Moving UP and monitoring counts...\n")

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlabel("Z position (steps)", fontsize=12)
    ax.set_ylabel("Photon counts", fontsize=12)
    ax.set_title("Z-Axis Diagnostic - Upward Scan", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        pass

    plt.ion()  # Interactive mode
    plt.show()

    # Data storage
    positions = []
    counts = []

    try:
        step_count = 0
        while step_count < max_steps:
            if tb.safe_stop():
                print("\n[User stopped scan]")
                break

            # Move up
            current_position = piezo.move_z_steps(step_size)
            time.sleep(0.05)  # Brief settling

            # Collect counts
            counter.clear_buffer()
            time.sleep(0.05)
            samples = counter.read_counter_simple()

            if len(samples) > 0:
                mean_counts = np.mean(samples)
            else:
                mean_counts = counts[-1] if len(counts) > 0 else initial_avg

            # Store data
            positions.append(current_position)
            counts.append(mean_counts)
            step_count += step_size

            # Print status
            if len(positions) % update_interval == 0:
                total_moved = current_position - start_position
                print(f"Step {total_moved:6d}: Position {current_position:7d}, Counts {mean_counts:6.0f}")

            # Update plot every update_interval steps
            if len(positions) % update_interval == 0:
                ax.clear()
                ax.plot(positions, counts, 'b-', linewidth=1.5)
                ax.set_xlabel("Z position (steps)", fontsize=12)
                ax.set_ylabel("Photon counts", fontsize=12)
                ax.set_title(f"Z-Axis Diagnostic - {len(positions)} points collected",
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Mark current position
                ax.plot(current_position, mean_counts, 'ro', markersize=8)

                # Mark peaks if any
                if len(counts) > 10:
                    max_idx = np.argmax(counts)
                    ax.plot(positions[max_idx], counts[max_idx], 'g*',
                           markersize=15, label=f'Peak: {counts[max_idx]:.0f} at step {positions[max_idx]}')
                    ax.legend()

                plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n[Scan interrupted by user]")

    finally:
        counter.stop_tag_stream()
        pulse_gen.stream_stop()

    # Final plot
    print(f"\n" + "="*60)
    print("SCAN COMPLETE")
    print("="*60)

    if len(counts) > 0:
        positions_array = np.array(positions)
        counts_array = np.array(counts)

        total_moved = positions[-1] - start_position
        max_idx = np.argmax(counts_array)

        print(f"Total moved: {total_moved} steps (~{total_moved*0.05:.0f} microns)")
        print(f"Starting counts: {counts[0]:.0f}")
        print(f"Final counts: {counts[-1]:.0f}")
        print(f"Peak counts: {counts_array[max_idx]:.0f} at step {positions[max_idx]}")
        print(f"Peak was at +{positions[max_idx] - start_position} steps from start")

        # Final detailed plot
        ax.clear()
        ax.plot(positions, counts, 'b-', linewidth=1.5, label='Scan data')
        ax.plot(positions[max_idx], counts_array[max_idx], 'g*',
               markersize=20, label=f'Peak: {counts_array[max_idx]:.0f}')
        ax.axvline(positions[max_idx], color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel("Z position (steps)", fontsize=12)
        ax.set_ylabel("Photon counts", fontsize=12)
        ax.set_title(f"Z-Axis Diagnostic - Peak at +{positions[max_idx] - start_position} steps",
                   fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.draw()

        print(f"\nLook for the peak in the plot - that's the surface!")
        print(f"To clear the surface, you need to move past the peak.")
        print(f"If counts haven't dropped after the peak, continue scanning.")
        print(f"\n" + "="*60)

    tb.reset_safe_stop()
    tb.reset_cfm()

    # Keep plot open
    plt.ioff()
    plt.show()

    return {
        "positions": positions,
        "counts": counts,
        "start_position": start_position,
    }


if __name__ == "__main__":
    """Example usage"""
    from utils.constants import NVSig, CoordsKey, VirtualLaserKey

    nv_sig = NVSig(
        name="z_diagnostic",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.Z: 0},
        pulse_durations={VirtualLaserKey.IMAGING: int(1e6)},
    )

    # Run diagnostic - steps up showing real-time counts
    results = main(nv_sig, step_size=10, max_steps=50000)
