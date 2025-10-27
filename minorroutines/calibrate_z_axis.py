# -*- coding: utf-8 -*-
"""
calubrate Z axis

Created on October 13th, 2023

@author: mccambria
"""
import numpy as np
import matplotlib.pyplot as plt
import majorroutines.confocal.confocal_stationary_count as stationary_count
from utils.constants import CoordsKey
import utils.positioning as pos
import utils.data_manager as dm
from utils import kplotlib as kpl


def process_and_plot(data):
    z_up = np.asarray(data["z_up"], float)
    counts_up = np.asarray(data["counts_up"], float)
    z_down = np.asarray(data["z_down"], float)
    counts_down = np.asarray(data["counts_down"], float)

    # Pull sweep params (for metadata / dz_nominal)
    z_start = float(data["z_start"])
    z_stop = float(data["z_stop"])
    num_steps = int(data["num_steps"])
    dz_nominal = (z_stop - z_start) / (num_steps - 1) if num_steps > 1 else np.nan

    # Linear fits (counts vs Z). Use simple OLS; if you want to be fancy, add weights.
    slope_up, intercept_up = np.polyfit(z_up, counts_up, 1)
    slope_down, intercept_down = np.polyfit(z_down, counts_down, 1)

    # Directional correction factor: scale DOWN steps so slopes match UP
    # If |slope_down| > |slope_up| then down moves "stronger" -> scale_down < 1
    if slope_down != 0:
        correction_factor = abs(slope_up / slope_down)
    else:
        correction_factor = np.nan

    # Symmetric normalization suggestion (optional)
    # Make both scales deviate equally around 1:
    # g = sqrt(|slope_down/slope_up|); z_scale_up = g, z_scale_down = 1/g
    symmetric_g = np.sqrt(abs(slope_down / slope_up)) if slope_up != 0 else np.nan
    z_scale_up_sym = symmetric_g
    z_scale_down_sym = 1.0 / symmetric_g if symmetric_g not in (0, np.nan) else np.nan

    # Plot data + fits
    fig, ax = plt.subplots()
    ax.plot(z_up, counts_up, "o-", label="Up sweep")
    ax.plot(z_down, counts_down, "s-", label="Down sweep")

    # Fit lines over the visible range for clarity
    zz = np.linspace(min(z_up.min(), z_down.min()), max(z_up.max(), z_down.max()), 100)
    ax.plot(zz, slope_up * zz + intercept_up, "--", label=f"Up fit (s={slope_up:.3g})")
    ax.plot(
        zz,
        slope_down * zz + intercept_down,
        "--",
        label=f"Down fit (s={slope_down:.3g})",
    )

    ax.set_xlabel("Z coordinate (arb.)")
    ax.set_ylabel("Average counts (kcps)")  # your stationary_count returns kcps avg
    ax.set_title(
        "Z calibration\n"
        f"dz_nominal={dz_nominal:.4g}, "
        f"corr (down←×)={correction_factor:.4f}, "
        f"sym up={z_scale_up_sym:.4f}, sym down={z_scale_down_sym:.4f}"
    )
    ax.legend()
    fig.tight_layout()

    # Hysteresis visualization: compare up vs down at same Z grid
    # Interpolate the DOWN curve onto the UP z-grid to see residuals
    try:
        counts_down_on_up = np.interp(z_up, z_down, counts_down)
        fig2, ax2 = plt.subplots()
        ax2.plot(z_up, counts_up - counts_down_on_up, "o-")
        ax2.axhline(0, lw=1)
        ax2.set_xlabel("Z coordinate (arb.)")
        ax2.set_ylabel("Up - Down (kcps)")
        ax2.set_title("Hysteresis (counts difference, up minus down)")
        fig2.tight_layout()
    except Exception:
        fig2 = None

    results = {
        "slope_up": slope_up,
        "slope_down": slope_down,
        "dz_nominal": dz_nominal,
        "correction_factor_down_only": correction_factor,  # apply to DOWN moves
        "z_scale_up_sym": z_scale_up_sym,
        "z_scale_down_sym": z_scale_down_sym,
    }
    return fig, fig2, results


def calibrate_z_axis(
    nv_sig,
    z_start=0,
    z_stop=10,
    num_steps=11,
    run_time=int(1e9),  # ns
    coords_Key=CoordsKey.Z,
):
    """
    Sweep Z in both + and - directions, measure counts, and compute correction factor.
    """
    # Generate range in both directions
    z_range_up = np.linspace(z_start, z_stop, num_steps)
    z_range_down = z_range_up[::-1]

    # Storage
    counts_up, counts_down = [], []

    # Forward sweep
    for z in z_range_up:
        nv_sig.coords[coords_Key] = float(z)
        pos.set_xyz_on_nv(nv_sig)
        avg, std = stationary_count.main(nv_sig, run_time)
        counts_up.append(avg)

    # Backward sweep
    for z in z_range_down:
        nv_sig.coords[coords_Key] = float(z)
        pos.set_xyz_on_nv(nv_sig)
        avg, std = stationary_count.main(nv_sig, run_time)
        counts_down.append(avg)

    counts_up = np.asarray(counts_up, float)
    counts_down = np.asarray(counts_down, float)

    raw_data = {
        "nv_sig": getattr(nv_sig, "name", "nv"),
        "z_start": z_start,
        "z_stop": z_stop,
        "num_steps": num_steps,
        "z_up": z_range_up,
        "counts_up": counts_up,
        "z_down": z_range_down,
        "counts_down": counts_down,
        "run_time_ns": run_time,
    }

    # Save raw
    ts = dm.get_time_stamp()
    path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, path)

    # Process + plot + save figs
    try:
        fig, fig2, results = process_and_plot(raw_data)
    except Exception:
        fig, fig2, results = None, None, {}

    if fig is not None:
        dm.save_figure(fig, path, stem="z_calibration")
    if fig2 is not None:
        dm.save_figure(fig2, path, stem="z_hysteresis")

    kpl.show(block=True)

    # Attach recommended server scales to results for convenience
    # Primary: scale DOWN only so slopes match UP
    # Optional symmetric: scales around 1.0
    if "correction_factor_down_only" in results:
        results["server_recommendation_primary"] = {
            "z_scale_up": 1.0,
            "z_scale_down": float(results["correction_factor_down_only"]),
        }
    if not np.isnan(results.get("z_scale_up_sym", np.nan)):
        results["server_recommendation_symmetric"] = {
            "z_scale_up": float(results["z_scale_up_sym"]),
            "z_scale_down": float(results["z_scale_down_sym"]),
        }

    # Save a summary JSON too (handy for logging/tuning)
    try:
        path = dm.get_file_path(__file__, ts, "z_calibration_summary")
        dm.save_raw_data(results, path)
    except Exception:
        pass

    return {"raw_path": path, **results}


if __name__ == "__main__":
    # Example: load a previous run (or call calibrate_z_axis live)
    # data = dm.get_raw_data(file_stem="z_calibration")  # if you have one saved
    pass
