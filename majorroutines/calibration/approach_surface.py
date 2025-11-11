# -*- coding: utf-8 -*-
"""
Safe surface approach - Move toward surface and stop at target count level

Safely approaches the surface by monitoring photon counts and stopping
when a target threshold is reached.

Created on November 7th, 2025
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from utils import common
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, NVSig, VirtualLaserKey


def main(nv_sig, target_counts=7000, step_size=10, max_steps=10000, direction="down"):
    """
    Safely approach surface by monitoring counts and stopping at target.

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters
    target_counts : float
        Stop when counts reach this level (default: 7000)
    step_size : int
        Step increment (default: 10)
    max_steps : int
        Maximum total steps to move (default: 10000)
    direction : str
        "up" or "down" (default: "down")

    Returns
    -------
    dict
        Final position and count data
    """

    # Get servers
    piezo = pos.get_positioner_server(CoordsKey.Z)
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    # Determine step direction
    if direction.lower() == "down":
        step_increment = -step_size
        direction_text = "DOWN (toward surface)"
    else:
        step_increment = step_size
        direction_text = "UP (toward surface)"

    print(f"Direction: {direction_text}")
    print(f"Target counts: {target_counts}")
    print(f"Step size: {step_size} steps")
    print(f"Press CTRL+C to stop at any time")

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
    print(f"Initial photon count: {initial_avg:.0f} counts\n")

    # Get starting position
    start_position = piezo.get_z_position()
    print(f"Starting position: {start_position} steps")
    print(f"Beginning approach to target counts = {target_counts}...\n")

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Safe Surface Approach", fontsize=16, fontweight='bold')

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        pass

    plt.ion()
    plt.show()

    # Data storage
    positions = [start_position]
    counts = [initial_avg]

    try:
        steps_moved = 0
        target_reached = False

        while steps_moved < max_steps:
            if tb.safe_stop():
                print("\n[User stopped approach]")
                break

            # Move one step
            current_position = piezo.move_z_steps(step_increment)
            time.sleep(0.05)

            # Collect counts
            counter.clear_buffer()
            time.sleep(0.05)
            samples = counter.read_counter_simple()

            if len(samples) > 0:
                mean_counts = np.mean(samples)
            else:
                mean_counts = counts[-1]

            # Store data
            positions.append(current_position)
            counts.append(mean_counts)
            steps_moved += abs(step_increment)

            # Print status every 20 steps
            if len(positions) % 20 == 1:
                distance_from_start = current_position - start_position
                print(f"Position {current_position:7d} ({distance_from_start:+6d} from start), Counts {mean_counts:6.0f}")

            # Update plot every step for continuous feedback
            if True:
                # Counts vs position
                ax.clear()
                ax.plot(positions, counts, 'b-', linewidth=2)
                ax.plot(current_position, mean_counts, 'ro', markersize=10)
                ax.axhline(target_counts, color='green', linestyle='--', linewidth=2,
                           label=f'Target: {target_counts}')
                ax.set_xlabel("Z position (steps)", fontsize=11)
                ax.set_ylabel("Photon counts", fontsize=11)
                ax.set_title(f"Current: {mean_counts:.0f} counts at step {current_position}", fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.pause(0.01)

            # Check if target reached
            if mean_counts >= target_counts:
                print(f"\n{'='*60}")
                print(f"TARGET REACHED")
                print(f"{'='*60}")
                print(f"Position: {current_position} steps")
                print(f"Counts: {mean_counts:.0f}")
                print(f"Moved {abs(current_position - start_position)} steps from start")

                # Set this position as Z=0 reference
                print(f"\nSetting current position as Z=0 reference...")
                piezo.set_z_reference(0)
                print(f"Z position reset complete. Surface is now at step=0")
                print(f"{'='*60}\n")

                target_reached = True
                break

    except KeyboardInterrupt:
        print("\n[Approach interrupted by user]")

    finally:
        counter.stop_tag_stream()
        pulse_gen.stream_stop()

    # Final summary
    print(f"\n" + "="*60)
    print("APPROACH COMPLETE")
    print("="*60)

    if len(counts) > 0:
        final_position = positions[-1]
        final_counts = counts[-1]
        total_moved = final_position - start_position

        print(f"Starting position: {start_position}")
        print(f"Final position: {final_position}")
        print(f"Total moved: {total_moved:+d} steps (~{abs(total_moved)*0.05:.0f} microns)")
        print(f"Starting counts: {counts[0]:.0f}")
        print(f"Final counts: {final_counts:.0f}")
        print(f"Target counts: {target_counts}")

        if target_reached:
            print(f"\nStatus: TARGET REACHED")
        else:
            print(f"\nStatus: STOPPED BEFORE TARGET")
            print(f"Remaining: {target_counts - final_counts:.0f} counts to target")

        print(f"="*60)

    tb.reset_safe_stop()
    tb.reset_cfm()

    # Keep plot open
    plt.ioff()
    plt.show()

    return {
        "positions": positions,
        "counts": counts,
        "start_position": start_position,
        "final_position": positions[-1] if positions else start_position,
        "target_reached": target_reached,
    }


if __name__ == "__main__":
    """Example usage"""
    from utils.constants import NVSig, CoordsKey, VirtualLaserKey

    nv_sig = NVSig(
        name="surface_approach",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.Z: 0},
        pulse_durations={VirtualLaserKey.IMAGING: int(1e6)},
    )

    # Approach surface from above, stop at 7000 counts
    results = main(nv_sig, target_counts=7000, direction="down")
