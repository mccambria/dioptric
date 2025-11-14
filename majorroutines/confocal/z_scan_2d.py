# -*- coding: utf-8 -*-
"""
3D confocal scan: 2D XY images at multiple Z depths.

Performs a Z-axis scan where each Z position generates a complete 2D XY confocal
image using galvo mirrors. Combines the modern piezo control from z_scan_1d.py
with the 2D scanning patterns from confocal_image_sample.py.

The "2D" in the name refers to the galvo XY movement, but this is technically
a 3D scan (XYZ).

Created on November 13th, 2025

"""

import time

import matplotlib.pyplot as plt
import numpy as np

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import (
    CountFormat,
    CoordsKey,
    NVSig,
    PosControlMode,
    VirtualLaserKey,
)

SEQ_FILE_PIXEL_READOUT = "simple_readout.py"


def _raster_fill(vals, img, state):
    """
    Row-major fill, bottom row first, left -> right on every row.
    `state` holds a single integer: number of pixels already written.
    """
    h, w = img.shape
    if not state:  # initialize write index
        state[:] = [0]
    i = state[0]
    for v in vals:
        if i >= h * w:
            break
        row = i // w
        col = i % w
        y = (h - 1) - row  # bottom row first (origin='lower')
        x = col  # left -> right
        img[y, x] = v
        i += 1
    state[:] = [i]


def main(
    nv_sig: NVSig,
    x_range,
    y_range,
    num_steps,
    num_z_steps,
    z_step_size,
    num_averages=1,
    min_threshold=None,
    nv_minus_init=False,
    save_data=True,
):
    """
    Perform a 3D scan: 2D XY confocal images at multiple Z depths.

    At each Z position:
    1. Move Z-axis relatively using modern piezo controls
    2. Perform complete 2D XY confocal scan using galvo mirrors
    3. Generate and display 2D image
    4. Check safety thresholds

    This combines:
    - Z-loop structure from z_scan_1d.py (relative piezo movement, safety)
    - XY scanning logic from confocal_image_sample.py (galvo raster scan)

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters including pulse durations, laser powers, and coordinates
    x_range : float
        X-axis scan range in voltage units (e.g., 0.2 V)
    y_range : float
        Y-axis scan range in voltage units (e.g., 0.2 V)
    num_steps : int
        Number of pixels per axis for XY images (e.g., 60 = 60x60 image)
        Same resolution used for both X and Y axes
    num_z_steps : int
        Number of Z positions to scan
    z_step_size : int
        Step size in piezo units per Z iteration.
        **Direction convention (absolute positioning):**
        - Negative values: move TOWARD sample (closer)
        - Positive values: move AWAY FROM sample (farther)
        Example: z_step_size=-10 moves 10 units closer to sample per step
    num_averages : int, optional
        Number of photon count samples to average at each XY pixel.
        Default: 1 (single sample per pixel, faster scanning)
        Higher values improve statistics but slow the scan significantly.
    min_threshold : float or None, optional
        Minimum photon count threshold for safety monitoring.
        Applied to the mean counts of each full XY image.
        If mean counts drop below this value, scan pauses and prompts:
        "Continue scanning? (y/n)"
        User must type 'y' to continue or 'n' to abort.
        None = no threshold monitoring (default: None)
    nv_minus_init : bool, optional
        Whether to use NV- charge state initialization (two-gate readout).
        True: uses modulo gates, subtracts background
        False: simple single-gate readout (default: False)
    save_data : bool, optional
        Whether to save data and figures to disk (default: True)
        Saves to: nvdata/pc_{hostname}/z_scan_2d/{date}/{timestamp}/

    Returns
    -------
    all_images : numpy.ndarray
        3D array of all XY images at each Z position.
        Shape: [num_z_steps_completed, num_steps, num_steps]
        Units: kcps (kilocounts per second) if config["count_format"] == KCPS,
               raw counts otherwise
    z_positions : numpy.ndarray
        1D array of actual Z positions in piezo steps.
        These are the real positions returned by the piezo after each move.
        Length: num_z_steps_completed (may be less than num_z_steps if aborted)

    Notes
    -----
    - XY scanning uses CoordsKey.PIXEL (galvo mirrors)
    - Z scanning uses CoordsKey.Z (piezo)
    - Total scan time ≈ (num_steps² × num_z_steps × readout_time)
      Example: 60² × 20 × 5ms = 360 seconds = 6 minutes
    - Memory usage: 60×60×20 images ≈ 5.7 MB (automatically compressed to NPZ)

    """

    ### Setup

    cfg = common.get_config_dict()
    count_fmt: CountFormat = cfg["count_format"]

    # Get servers
    piezo = pos.get_positioner_server(CoordsKey.Z)
    positioner = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
    mode = pos.get_positioner_control_mode(positioner)
    pulse = tb.get_server_pulse_streamer()
    ctr = tb.get_server_counter()

    # Setup laser for imaging
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"]))
    )
    readout_s = readout_ns * 1e-9
    readout_laser = vld["physical_name"]

    # Note: Unlike z_scan_1d, we do NOT call tb.set_filter() or tb.set_laser_power()
    # here because the pulse sequence handles laser control for galvo scanning.
    # Calling these functions can interfere with sequence-based control.

    # Build XY scan grid (same for all Z positions)
    x0, y0 = pos.get_nv_coords(nv_sig, coords_key=CoordsKey.PIXEL)
    X, Y, x1d, y1d, extent = pos.get_scan_grid_2d(
        x0, y0, x_range, y_range, num_steps, num_steps
    )
    h = w = num_steps
    xy_pixels_per_image = h * w

    # Storage for all Z slices
    all_images = []  # List of 2D numpy arrays
    z_positions = []  # Actual Z positions from piezo
    all_figures = []  # Store figure handles for each Z position

    kpl.init_kplotlib()

    tb.reset_cfm()
    tb.init_safe_stop()

    # Sequence loading - load once, trigger per position
    pos_key = CoordsKey.PIXEL
    delay_ns = int(cfg["Positioning"]["Positioners"][pos_key]["delay"])
    period_ns = pulse.stream_load(
        SEQ_FILE_PIXEL_READOUT,
        tb.encode_seq_args([delay_ns, readout_ns, readout_laser, 1.0]),  # Power controlled by sequence
    )[0]

    # Start tag stream once
    ctr.start_tag_stream()

    ### Z-axis scan loop

    print(f"\n{'='*60}")
    print(f"Starting 3D Z-Scan")
    print(f"{'='*60}")
    print(f"Z steps: {num_z_steps} steps of {z_step_size} units")
    print(
        f"Direction: {'TOWARD sample' if z_step_size < 0 else 'AWAY FROM sample'}"
    )
    print(f"XY scan: {num_steps}x{num_steps} pixels, range={x_range}V x {y_range}V")
    print(f"Total pixels: {xy_pixels_per_image * num_z_steps:,}")
    if min_threshold is not None:
        print(f"Threshold monitoring: {min_threshold:.0f} counts (mean per image)")
    print(f"{'='*60}\n")

    try:
        for z_idx in range(num_z_steps):
            if tb.safe_stop():
                print("\n[STOPPED] User interrupt detected")
                break

            print(f"[Z {z_idx+1}/{num_z_steps}] Moving Z position...", flush=True)

            # Move Z position relatively
            current_z_pos = piezo.move_z_steps(z_step_size)
            z_positions.append(current_z_pos)
            time.sleep(0.01)  # Brief settling time

            print(
                f"[Z {z_idx+1}/{num_z_steps}] Z position: {current_z_pos} steps",
                flush=True,
            )
            print(
                f"[Z {z_idx+1}/{num_z_steps}] Scanning XY plane ({num_steps}x{num_steps})...",
                flush=True,
            )

            # Create NEW figure for this Z position (one figure per Z slice)
            fig, ax = plt.subplots(figsize=(7, 5.2))

            # Create image array for this Z position
            img = np.full((h, w), np.nan, float)
            img_kcps = np.copy(img) if count_fmt == CountFormat.KCPS else None

            # Initialize the figure with empty image
            kpl.imshow(
                ax,
                img_kcps if img_kcps is not None else img,
                title=f"{readout_laser}, {readout_ns/1e6:.1f} ms, Z={current_z_pos} steps",
                cbar_label=("Kcps" if img_kcps is not None else "Counts"),
                extent=extent,
            )
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Voltage (V)")
            plt.show(block=False)  # Show this figure
            plt.pause(0.01)  # Force initial display

            # Store figure handle
            all_figures.append(fig)

            # === Perform 2D XY scan at this Z position ===
            # This is the core scanning logic from confocal_image_sample.py

            pixel_count = 0  # Track progress
            UPDATE_EVERY = 1  # Update every pixel (same as confocal_image_sample.py)

            if mode == PosControlMode.SEQUENCE:
                # Hardware-driven galvo scanning
                # Build raster-order coordinate lists
                Xr = []
                Yr = []
                for row in range(h):
                    y = y1d[(h - 1) - row]  # bottom row first
                    for col in range(w):
                        x = x1d[col]  # left -> right always
                        Xr.append(x)
                        Yr.append(y)

                # Note: SEQ_FILE_PIXEL_READOUT doesn't support XY lists
                # This would need a different sequence file
                # For now, we'll fall through to STEP mode
                raise NotImplementedError(
                    "SEQUENCE mode not yet supported for z_scan_2d. Use STEP/STREAM mode."
                )

            else:
                # Software-driven scanning (step-and-measure)
                written = []  # state for _raster_fill
                for row in range(h):
                    if tb.safe_stop():
                        break
                    y = y1d[(h - 1) - row]  # bottom row first
                    for col in range(w):
                        if tb.safe_stop():
                            break
                        x = x1d[col]  # left -> right
                        pos.set_xyz((x, y), positioner=positioner)

                        # Collect num_averages samples at this pixel
                        counts = []
                        for _ in range(num_averages):
                            if tb.safe_stop():
                                break

                            pulse.stream_start(1)
                            # Read exactly ONE sample per pixel
                            if nv_minus_init:
                                raw = ctr.read_counter_modulo_gates(2, 1)  # [[a,b]]
                                val_single = max(int(raw[0][0]) - int(raw[0][1]), 0)
                            else:
                                raw = ctr.read_counter_simple(1)  # [c]
                                val_single = int(raw[0])

                            counts.append(val_single)

                        # Average the samples
                        val = int(np.mean(counts)) if len(counts) > 0 else 0
                        vals = [val]

                        if not vals:
                            continue

                        _raster_fill(vals, img, written)

                        # Progress update every 10% of pixels
                        pixel_count += 1

                        # Update plot in real-time (throttled to avoid matplotlib overhead)
                        if (pixel_count % UPDATE_EVERY) == 0 or pixel_count == 1:
                            if img_kcps is not None:
                                img_kcps[:] = (img / 1000.0) / readout_s
                                kpl.imshow_update(ax, img_kcps)
                            else:
                                kpl.imshow_update(ax, img)

                        if pixel_count % (xy_pixels_per_image // 10) == 0:
                            progress = (pixel_count / xy_pixels_per_image) * 100
                            print(f"  Progress: {progress:.0f}% ({pixel_count}/{xy_pixels_per_image} pixels)", flush=True)

            # === End of XY scan for this Z ===

            # Convert to kcps if needed
            is_kcps = count_fmt == CountFormat.KCPS
            img_display = (img / 1000.0) / readout_s if is_kcps else img

            # Store the image (in output units)
            all_images.append(img_display.copy())

            # Calculate mean counts for threshold check
            mean_counts = np.nanmean(img_display)

            print(
                f"[Z {z_idx+1}/{num_z_steps}] Complete. Mean counts: {mean_counts:.1f}",
                flush=True,
            )

            # Update title to show final stats (image was already updated during scan)
            ax.set_title(f"{readout_laser}, {readout_ns/1e6:.1f} ms, Z={current_z_pos} steps (Mean={mean_counts:.1f})")

            # Final refresh
            plt.pause(0.01)

            # Check threshold
            if min_threshold is not None and mean_counts < min_threshold:
                print(f"\n[THRESHOLD REACHED]")
                print(f"  Mean counts: {mean_counts:.1f}")
                print(f"  Threshold: {min_threshold:.0f}")
                print(f"  Z position: {current_z_pos} steps")
                response = input("  Continue scanning? (y/n): ").strip().lower()
                if response != "y":
                    print("[STOPPED] Scan aborted by user")
                    break
                print("  Continuing scan...\n")

            print()  # Blank line between Z steps

    finally:
        ctr.stop_tag_stream()
        tb.reset_safe_stop()
        tb.reset_cfm()
        print(f"\nScan complete: {len(z_positions)} Z slices collected")

    ### Save data

    print("Processing data...", flush=True)

    # Convert lists to numpy arrays
    all_images_array = np.array(all_images)  # Shape: [num_z, h, w]
    z_positions_array = np.array(z_positions)

    units_out = "kcps" if is_kcps else "counts"

    ts = dm.get_time_stamp()
    raw_data = {
        "timestamp": ts,
        "nv_sig": nv_sig,
        "mode": "z_scan_2d",
        "num_z_steps_requested": num_z_steps,
        "num_z_steps_completed": len(z_positions),
        "z_step_size": z_step_size,
        "num_xy_steps": num_steps,
        "x_range": x_range,
        "y_range": y_range,
        "x_center": x0,
        "y_center": y0,
        "extent": extent,
        "num_averages": num_averages,
        "min_threshold": min_threshold,
        "readout_ns": readout_ns,
        "readout_units": "ns",
        "img_arrays": all_images_array.astype(float),  # Compressed to NPZ
        "img_arrays_units": units_out,
        "img_arrays_shape": list(all_images_array.shape),
        "z_positions": z_positions_array.tolist(),
        "x_coords_1d": x1d,
        "y_coords_1d": y1d,
    }

    if save_data:
        print("Saving data and figures...", flush=True)
        base_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))

        # Save each figure separately with Z index in filename
        for z_idx, (fig, z_pos) in enumerate(zip(all_figures, z_positions)):
            # Create filename with Z index
            fig_path = base_path.parent / f"{base_path.stem}_z{z_idx:03d}_pos{z_pos}.png"
            dm.save_figure(fig, fig_path)
            print(f"  Saved figure {z_idx+1}/{len(all_figures)}: {fig_path.name}")

        # Save raw data once (contains all images)
        dm.save_raw_data(raw_data, base_path, keys_to_compress=["img_arrays"])
        print(f"Data saved to: {base_path}")

    # Turn off interactive mode and show final plots
    plt.ioff()
    print(f"Displaying {len(all_figures)} figure windows (close all to exit)...", flush=True)
    kpl.show()

    return all_images_array, z_positions_array


if __name__ == "__main__":
    # Example usage
    from utils.constants import CoordsKey, NVSig, VirtualLaserKey

    nv_sig = NVSig(
        name="test_z_scan_2d",
        coords={
            CoordsKey.SAMPLE: [0.0, 0.0],
            CoordsKey.Z: 0,
            CoordsKey.PIXEL: [0.0, 0.0],
        },
        pulse_durations={VirtualLaserKey.IMAGING: int(5e6)},  # 5 ms readout
    )

    # Small test scan
    results = main(
        nv_sig,
        x_range=0.2,  # 0.2V XY range
        y_range=0.2,
        num_steps=10,  # 10x10 pixels (fast)
        num_z_steps=3,  # Only 3 Z slices
        z_step_size=5,  # Move away from sample (positive = safer)
        num_averages=1,
        min_threshold=100,  # Pause if mean counts < 100
    )
