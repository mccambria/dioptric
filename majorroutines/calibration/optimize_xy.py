# -*- coding: utf-8 -*-
"""
XY optimization routine using concentric circle scan pattern.

Scans the galvo in concentric circles around the current position,
collects photon counts, and fits to find the optimal XY position.

Unlike Z optimization (piezo with hysteresis), galvos have no hysteresis
so we can use efficient streaming mode without hill-climbing.

Created on December 16th, 2025

"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, NVSig, VirtualLaserKey


def generate_concentric_circles(center_x, center_y, max_radius, num_radii, points_per_circle):
    """
    Generate coordinates for concentric circle scan pattern.

    Parameters
    ----------
    center_x : float
        X coordinate of center point
    center_y : float
        Y coordinate of center point
    max_radius : float
        Maximum radius of outer circle
    num_radii : int
        Number of concentric circles (not including center)
    points_per_circle : int
        Number of points per circle

    Returns
    -------
    coords_x : np.ndarray
        X coordinates of all scan points
    coords_y : np.ndarray
        Y coordinates of all scan points
    radii : np.ndarray
        Radius value for each point (0 for center)
    """
    # Start with center point
    coords_x = [center_x]
    coords_y = [center_y]
    radii = [0.0]

    # Generate concentric circles at increasing radii
    radius_vals = np.linspace(max_radius / num_radii, max_radius, num_radii)
    for r in radius_vals:
        cx, cy = pos.get_scan_circle_2d(center_x, center_y, r, points_per_circle)
        coords_x.extend(cx)
        coords_y.extend(cy)
        radii.extend([r] * len(cx))

    return np.array(coords_x), np.array(coords_y), np.array(radii)


def gaussian_2d(coords, amplitude, x0, y0, sigma, offset):
    """
    2D Gaussian function for fitting.

    Parameters
    ----------
    coords : tuple
        (x, y) coordinate arrays
    amplitude : float
        Peak height above offset
    x0, y0 : float
        Center coordinates
    sigma : float
        Standard deviation (same for x and y)
    offset : float
        Background offset

    Returns
    -------
    np.ndarray
        Gaussian values at each coordinate
    """
    x, y = coords
    return offset + amplitude * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    )


def main(
    nv_sig: NVSig,
    num_radii: int = 5,
    points_per_circle: int = 12,
    fit_method: str = "gaussian",
    move_to_optimal: bool = True,
    save_data: bool = True,
    num_averages: int = 1,
) -> dict:
    """
    Optimize XY position using concentric circle scan pattern.

    Scans the galvo in concentric circles, collects photon counts at each
    position, and finds the optimal XY coordinates using either 2D Gaussian
    fitting or maximum counts.

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters (pulse durations, laser settings)
    num_radii : int, optional
        Number of concentric circles to scan. Default: 5
    points_per_circle : int, optional
        Number of points per circle. Default: 12 (every 30 degrees)
    fit_method : str, optional
        Method to find optimal position: "gaussian" for 2D Gaussian fit,
        "max_counts" for maximum counts. Default: "gaussian"
    move_to_optimal : bool, optional
        Whether to move galvo to optimal position after fitting. Default: True
    save_data : bool, optional
        Whether to save data and plot. Default: True
    num_averages : int, optional
        Number of counts to average at each position. Default: 1

    Returns
    -------
    dict
        Results containing:
        - opti_x: Optimal X coordinate
        - opti_y: Optimal Y coordinate
        - opti_counts: Counts at optimal position
        - coords_x: Array of scanned X coordinates
        - coords_y: Array of scanned Y coordinates
        - counts: Array of photon counts
        - radii: Array of radius values
        - fit_params: Gaussian fit parameters (if fit_method="gaussian")
        - fit_success: Whether fit succeeded
        - initial_coords: Initial [x, y] coordinates
    """

    ### Setup

    config = common.get_config_dict()

    # Get hardware servers
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    # Get optimize range from config
    optimize_range = pos.get_positioner_optimize_range(CoordsKey.PIXEL)

    # Setup laser for imaging (matching confocal_image_sample.py - no set_filter/set_laser_power)
    laser_dict = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(laser_dict["duration"]))
    )
    readout_s = readout_ns / 1e9
    laser_name = laser_dict["physical_name"]

    # Get initial position
    initial_coords = pos.get_nv_coords(nv_sig, CoordsKey.PIXEL)
    center_x = initial_coords[0]
    center_y = initial_coords[1]

    print(f"\nXY Optimization - Concentric Circle Scan")
    print(f"="*50)
    print(f"Initial position: X={center_x:.4f}, Y={center_y:.4f}")
    print(f"Scan range: {optimize_range:.4f} V")
    print(f"Pattern: {num_radii} circles, {points_per_circle} points/circle")
    print(f"Fit method: {fit_method}")
    print(f"="*50 + "\n")

    ### Generate scan coordinates

    coords_x, coords_y, radii = generate_concentric_circles(
        center_x, center_y, optimize_range, num_radii, points_per_circle
    )
    num_points = len(coords_x)
    print(f"Total scan points: {num_points}")

    ### Setup figure for real-time display (confocal-style imshow)

    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create grid for interpolated image display
    grid_resolution = 50  # Number of pixels per side
    half_range = optimize_range * 1.1  # Slightly larger than scan area
    x_grid = np.linspace(center_x - half_range, center_x + half_range, grid_resolution)
    y_grid = np.linspace(center_y - half_range, center_y + half_range, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Initialize with NaN (will show as blank)
    img_array = np.full((grid_resolution, grid_resolution), np.nan)

    # Set up extent for imshow (left, right, bottom, top)
    half_pixel = (x_grid[1] - x_grid[0]) / 2
    extent = [
        x_grid[0] - half_pixel,
        x_grid[-1] + half_pixel,
        y_grid[-1] + half_pixel,  # Note: y is flipped for image display
        y_grid[0] - half_pixel,
    ]

    # Initial imshow display
    kpl.imshow(
        ax,
        img_array,
        title=f"XY Optimization - {laser_name}, {readout_ns/1e6:.1f} ms",
        x_label="X Voltage (V)",
        y_label="Y Voltage (V)",
        cbar_label="Counts",
        extent=extent,
    )

    # Mark initial position
    ax.plot(center_x, center_y, "r+", markersize=15, markeredgewidth=2, zorder=10)

    plt.ion()
    plt.pause(0.1)

    ### Hardware setup (exact order from confocal_image_sample.py)

    tb.reset_cfm()
    tb.init_safe_stop()
    counter.start_tag_stream()  # Start BEFORE loading sequence

    # Load pulse sequence
    seq_file = "simple_readout.py"
    positioner_dict = config["Positioning"]["Positioners"][CoordsKey.PIXEL]
    delay_ns = int(positioner_dict.get("delay", 0))

    seq_args = [delay_ns, readout_ns, laser_name, 1.0]  # Use 1.0 for power like confocal_image_sample
    pulse_gen.stream_load(seq_file, tb.encode_seq_args(seq_args))

    ### Collect counts

    counts_list = []

    # Helper function to update image display with interpolation
    def update_image_display(x_pts, y_pts, counts_pts):
        if len(counts_pts) < 3:
            return
        # Interpolate scattered points onto grid
        try:
            img_interp = griddata(
                (x_pts, y_pts),
                counts_pts,
                (X_grid, Y_grid),
                method='cubic',
                fill_value=np.nan
            )
            # Fill any remaining NaN with nearest neighbor
            mask = np.isnan(img_interp)
            if np.any(mask) and not np.all(mask):
                img_nearest = griddata(
                    (x_pts, y_pts),
                    counts_pts,
                    (X_grid, Y_grid),
                    method='nearest'
                )
                img_interp[mask] = img_nearest[mask]
            kpl.imshow_update(ax, img_interp)
        except Exception:
            pass  # Skip update if interpolation fails

    try:
        # Use STEP mode pattern (same as confocal_image_sample.py):
        # Move position -> trigger 1 pulse -> read 1 sample -> repeat
        print(f"Scanning {num_points} positions (step mode)...")
        print(f"DEBUG: delay_ns={delay_ns}, readout_ns={readout_ns}, laser={laser_name}")

        for i in range(num_points):
            if tb.safe_stop():
                print("\n[STOPPED] User interrupt")
                break

            # Move galvo to this position (same as confocal_image_sample)
            pos.set_xyz((coords_x[i], coords_y[i]), positioner=CoordsKey.PIXEL)

            # Collect samples at this position
            samples = []
            for _ in range(num_averages):
                # Trigger exactly 1 pulse sequence
                pulse_gen.stream_start(1)
                # Read exactly 1 sample (blocking)
                raw = counter.read_counter_simple(1)

                # Debug: print first few raw values
                if i < 3:
                    print(f"DEBUG point {i}: raw={raw}, type={type(raw)}")

                if raw and len(raw) > 0:
                    samples.append(int(raw[0]))

            # Average the samples
            avg_counts = np.mean(samples) if samples else 0
            counts_list.append(avg_counts)

            # Update interpolated image periodically
            if (i + 1) % 5 == 0 or i == num_points - 1:
                update_image_display(
                    coords_x[:i+1],
                    coords_y[:i+1],
                    counts_list[:i+1]
                )
                plt.pause(0.01)

            # Progress output
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_points}, last count: {avg_counts:.0f}")

    finally:
        counter.clear_buffer()
        tb.reset_cfm()
        tb.reset_safe_stop()

    counts_array = np.array(counts_list)

    # Truncate coordinates if scan was interrupted
    actual_points = len(counts_array)
    coords_x = coords_x[:actual_points]
    coords_y = coords_y[:actual_points]
    radii = radii[:actual_points]

    print(f"\nCollected {actual_points} data points")

    ### Find optimal position

    opti_x = None
    opti_y = None
    fit_params = None
    fit_success = False

    if fit_method == "gaussian" and actual_points >= 10:
        # 2D Gaussian fit
        try:
            # Initial guesses
            max_idx = np.argmax(counts_array)
            x0_guess = coords_x[max_idx]
            y0_guess = coords_y[max_idx]
            offset_guess = np.min(counts_array)
            amplitude_guess = np.max(counts_array) - offset_guess
            sigma_guess = optimize_range / 3

            guess = [amplitude_guess, x0_guess, y0_guess, sigma_guess, offset_guess]
            bounds = (
                [0, center_x - optimize_range, center_y - optimize_range, 0, 0],
                [np.inf, center_x + optimize_range, center_y + optimize_range, optimize_range, np.inf]
            )

            popt, _ = curve_fit(
                gaussian_2d,
                (coords_x, coords_y),
                counts_array,
                p0=guess,
                bounds=bounds,
                maxfev=5000
            )

            opti_x = popt[1]
            opti_y = popt[2]
            fit_params = {
                "amplitude": popt[0],
                "x0": popt[1],
                "y0": popt[2],
                "sigma": popt[3],
                "offset": popt[4],
            }
            fit_success = True

            print(f"\n2D Gaussian fit results:")
            print(f"  Optimal X: {opti_x:.4f} V")
            print(f"  Optimal Y: {opti_y:.4f} V")
            print(f"  Amplitude: {popt[0]:.0f} counts")
            print(f"  Sigma: {popt[3]:.4f} V")

        except Exception as e:
            print(f"\nGaussian fit failed: {e}")
            print("Falling back to max counts method")
            fit_method = "max_counts"

    if fit_method == "max_counts" or (not fit_success and actual_points > 0):
        # Use position of maximum counts
        max_idx = np.argmax(counts_array)
        opti_x = coords_x[max_idx]
        opti_y = coords_y[max_idx]
        print(f"\nMax counts method:")
        print(f"  Optimal X: {opti_x:.4f} V")
        print(f"  Optimal Y: {opti_y:.4f} V")
        print(f"  Max counts: {counts_array[max_idx]:.0f}")

    ### Update plot with final interpolated image

    # Final interpolated image
    img_final = None
    if actual_points >= 3:
        img_final = griddata(
            (coords_x, coords_y),
            counts_array,
            (X_grid, Y_grid),
            method='cubic',
            fill_value=np.nan
        )
        # Fill NaN with nearest neighbor
        mask = np.isnan(img_final)
        if np.any(mask) and not np.all(mask):
            img_nearest = griddata(
                (coords_x, coords_y),
                counts_array,
                (X_grid, Y_grid),
                method='nearest'
            )
            img_final[mask] = img_nearest[mask]
        kpl.imshow_update(ax, img_final)

    # Mark the scan points as small dots
    ax.scatter(coords_x, coords_y, c='white', s=3, alpha=0.5, zorder=5)

    # Mark optimal position
    if opti_x is not None and opti_y is not None:
        ax.plot(opti_x, opti_y, "g*", markersize=20, zorder=10,
                label=f"Optimal ({opti_x:.4f}, {opti_y:.4f})")

        # Draw circle showing fit sigma if available
        if fit_params is not None:
            sigma = fit_params["sigma"]
            circle = plt.Circle((opti_x, opti_y), sigma, fill=False,
                               color="lime", linestyle="--", linewidth=2,
                               label=f"1σ = {sigma:.4f}", zorder=9)
            ax.add_patch(circle)

    # Re-mark initial position
    ax.plot(center_x, center_y, "r+", markersize=15, markeredgewidth=2, zorder=10,
            label="Initial position")

    ax.legend(loc="upper right", facecolor='white', framealpha=0.8)
    plt.pause(0.1)

    ### Move to optimal position

    opti_counts = None
    if opti_x is not None and opti_y is not None and move_to_optimal:
        print(f"\nMoving to optimal position...")
        pos.set_xyz((opti_x, opti_y), positioner=CoordsKey.PIXEL)
        time.sleep(0.05)  # Settling time

        # Measure counts at optimal position
        counter.start_tag_stream()
        samples = []
        for _ in range(5):  # Average 5 samples for verification
            pulse_gen.stream_start(1)
            raw = counter.read_counter_simple(1)
            if raw:
                samples.append(int(raw[0]))
        counter.stop_tag_stream()

        opti_counts = int(np.mean(samples)) if samples else 0
        print(f"  Counts at optimal position: {opti_counts}")

    plt.ioff()
    tb.reset_cfm()

    ### Prepare results

    # Store final interpolated image if available
    img_array_final = img_final.tolist() if img_final is not None else None

    results = {
        "opti_x": float(opti_x) if opti_x is not None else None,
        "opti_y": float(opti_y) if opti_y is not None else None,
        "opti_counts": opti_counts,
        "coords_x": coords_x.tolist(),
        "coords_y": coords_y.tolist(),
        "counts": counts_array.tolist(),
        "radii": radii.tolist(),
        "fit_params": fit_params,
        "fit_success": fit_success,
        "initial_coords": initial_coords,
        "img_array": img_array_final,
        "img_extent": extent,
        "scan_params": {
            "num_radii": num_radii,
            "points_per_circle": points_per_circle,
            "optimize_range": optimize_range,
            "fit_method": fit_method,
            "num_averages": num_averages,
        },
    }

    ### Save data

    if save_data:
        timestamp = dm.get_time_stamp()
        raw_data = {
            "timestamp": timestamp,
            "nv_sig": nv_sig,
            "optimization_results": results,
        }
        nv_name = getattr(nv_sig, "name", "unknown")
        file_path = dm.get_file_path(__file__, timestamp, f"{nv_name}_xy_optimize")
        dm.save_raw_data(raw_data, file_path)
        dm.save_figure(fig, file_path)
        print(f"\nData saved to: {file_path}")

    print(f"\n{'='*50}")
    print("XY OPTIMIZATION COMPLETE")
    if opti_x is not None and opti_y is not None:
        print(f"  Optimal position: X={opti_x:.4f}, Y={opti_y:.4f}")
        if opti_counts is not None:
            print(f"  Counts at optimal: {opti_counts}")
    print(f"{'='*50}\n")

    kpl.show()

    return results


if __name__ == "__main__":
    """Example usage for testing"""
    from utils.constants import NVSig, CoordsKey, VirtualLaserKey

    # Create a minimal nv_sig for testing
    nv_sig = NVSig(
        name="test_xy_optimize",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.PIXEL: [0.0, 0.0], CoordsKey.Z: 0},
        pulse_durations={VirtualLaserKey.IMAGING: int(1e6)},  # 1 ms
    )

    # Run optimization
    results = main(
        nv_sig,
        num_radii=5,
        points_per_circle=12,
        fit_method="gaussian",  # or "max_counts"
    )
