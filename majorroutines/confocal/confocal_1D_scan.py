# -*- coding: utf-8 -*-
"""
1D scan over a designated area, collecting photon counts at each point.
In COUNTER mode.

Supports galvo (AOD) scanning via SEQUENCE or piezo scanning via STREAM/STEP.
"""

import time
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np

import utils.data_manager as dm
from utils import common
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import (
    CollectionMode,
    CoordsKey,
    CountFormat,
    NVSig,
    PosControlMode,
    VirtualLaserKey,
)

SEQ_FILE_SEQUENCE_SCAN = "simple_readout_laser_free_test.py"  # expects [0, readout_ns, x[], y[]]
SEQ_FILE_PIXEL_READOUT  = "simple_readout.py"                 # per-pixel readout when we move in Python

# --- replace the snake helper with a raster helper ---
def _raster_fill(vals, img, state):
    """
    Row-major fill, bottom row first, left -> right on every row.
    `state` holds a single integer: number of pixels already written.
    """
    h, w = img.shape
    if not state:        # initialize write index
        state[:] = [0]
    i = state[0]
    for v in vals:
        if i >= h * w:
            break
        row = i // w
        col = i %  w
        y = (h - 1) - row   # bottom row first (origin='lower')
        x = col             # left -> right
        img[y, x] = v
        i += 1
    state[:] = [i]

def confocal_scan_1d(nv_sig: NVSig, x_range, num_steps, nv_minus_init=False):
    """
    Perform a 1D confocal scan along the X axis only.
    
    Args:
        nv_sig: NV signature with positioning info
        x_range: Range to scan in X (voltage units)
        num_steps: Number of points along X axis
        nv_minus_init: Whether to use charge initialization
    """
    cfg = common.get_config_dict()
    count_fmt: CountFormat = cfg["count_format"]

    positioner = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
    mode = pos.get_positioner_control_mode(positioner)

    # Build 1D grid (X axis only)
    x0, y0 = pos.get_nv_coords(nv_sig, coords_key=CoordsKey.PIXEL)
    
    # Create 1D array of X positions
    x1d = np.linspace(x0 - x_range/2, x0 + x_range/2, num_steps)
    
    # Keep Y constant at current position
    y_fixed = y0
    
    total = num_steps

    # Timing / laser setup
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"])))
    readout_s = readout_ns / 1e9
    readout_laser = vld["physical_name"]

    pulse = tb.get_server_pulse_streamer()
    ctr = tb.get_server_counter()

    # Initialize 1D data array for counts
    counts_1d = np.full(num_steps, np.nan, float)
    counts_1d_kcps = np.copy(counts_1d) if count_fmt == CountFormat.KCPS else None

    # Setup plotting - line chart instead of image
    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("X Position (Voltage)")
    ax.set_ylabel("Kcps" if count_fmt == CountFormat.KCPS else "Raw Counts")
    ax.set_title(f"{readout_laser}, {readout_ns/1e6:.1f} ms readout")
    
    # Initial plot with NaN values
    line, = ax.plot(x1d, counts_1d if counts_1d_kcps is None else counts_1d_kcps, 
                    'b.-', markersize=4)
    ax.set_xlim(x1d[0], x1d[-1])
    ax.grid(True, alpha=0.3)
    
    tb.reset_cfm()
    tb.init_safe_stop()
    ctr.start_tag_stream()

    # Sequence loading
    pos_key = CoordsKey.PIXEL
    delay_ns = int(cfg["Positioning"]["Positioners"][pos_key]["delay"])
    period_ns = pulse.stream_load(
        SEQ_FILE_PIXEL_READOUT,
        tb.encode_seq_args([delay_ns, readout_ns, readout_laser, 1.0])
    )[0]

    try:
        if mode == PosControlMode.SEQUENCE:
            # SEQUENCE MODE: Build coordinate lists for 1D scan
            X_coords = list(x1d)
            Y_coords = [y_fixed] * num_steps  # Y stays constant
            
            seq_args = [0, readout_ns, X_coords, Y_coords]
            period_ns = pulse.stream_load(SEQ_FILE_SEQUENCE_SCAN, tb.encode_seq_args(seq_args))[0]
            pulse.stream_start(total)
            
            vals_collected = []
            timeout = time.time() + (period_ns * 1e-9) * total + 10.0
            
            while len(vals_collected) < total and not tb.safe_stop():
                if time.time() > timeout: break
                
                if nv_minus_init:
                    raw = ctr.read_counter_modulo_gates(2)
                    vals = [max(int(a)-int(b), 0) for (a,b) in raw]
                else:
                    raw = ctr.read_counter_simple()
                    vals = [int(v) for v in raw]
                
                vals_collected.extend(vals)
                
                # Update counts array with collected values
                n_to_update = min(len(vals_collected), num_steps)
                counts_1d[:n_to_update] = vals_collected[:n_to_update]
                
                # Update plot
                if counts_1d_kcps is not None:
                    counts_1d_kcps[:] = (counts_1d/1000.0)/readout_s
                    line.set_ydata(counts_1d_kcps)
                else:
                    line.set_ydata(counts_1d)
                    
                # Auto-scale Y axis
                valid_data = counts_1d_kcps if counts_1d_kcps is not None else counts_1d
                valid_mask = ~np.isnan(valid_data)
                if np.any(valid_mask):
                    ax.set_ylim(np.nanmin(valid_data) * 0.9, np.nanmax(valid_data) * 1.1)
                
                ax.figure.canvas.draw()
                ax.figure.canvas.flush_events()
                
        else:
            # STEP/STREAM MODE: Move positioner point by point
            for i, x_pos in enumerate(x1d):
                if tb.safe_stop(): break
                
                # Set position (x varies, y fixed)
                pos.set_xyz((x_pos, y_fixed), positioner=positioner)
                
                pulse.stream_start(1)
                
                # Read one sample
                if nv_minus_init:
                    raw = ctr.read_counter_modulo_gates(2, 1)
                    val = max(int(raw[0][0]) - int(raw[0][1]), 0)
                else:
                    raw = ctr.read_counter_simple(1)
                    val = int(raw[0])
                
                counts_1d[i] = val
                
                # Update plot
                if counts_1d_kcps is not None:
                    counts_1d_kcps[:] = (counts_1d/1000.0)/readout_s
                    line.set_ydata(counts_1d_kcps)
                else:
                    line.set_ydata(counts_1d)
                
                # Auto-scale Y axis
                valid_data = counts_1d_kcps if counts_1d_kcps is not None else counts_1d
                valid_mask = ~np.isnan(valid_data)
                if np.any(valid_mask):
                    ax.set_ylim(np.nanmin(valid_data) * 0.9, np.nanmax(valid_data) * 1.1)
                
                ax.figure.canvas.draw()
                ax.figure.canvas.flush_events()
                
    finally:
        ctr.clear_buffer()
        tb.reset_cfm()

    # Save data
    is_kcps = (count_fmt == CountFormat.KCPS)
    counts_out = (counts_1d/1000.0)/readout_s if is_kcps else counts_1d
    units_out = "kcps" if is_kcps else "counts"

    ts = dm.get_time_stamp()
    raw_data = {
        "timestamp": ts,
        "nv_sig": nv_sig,
        "mode": "confocal_scan_1d",
        "num_steps": num_steps,
        "x_range": x_range,
        "x_center": x0, "y_center": y0,
        "readout_ns": readout_ns, "readout_units": "ns",
        "counts_array": counts_out.astype(float), 
        "counts_array_units": units_out,
        "x_coords_1d": x1d,
        "y_fixed": y_fixed
    }
    
    path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_figure(fig, path)
    dm.save_raw_data(raw_data, path)
    kpl.show()
    
    return counts_out, x1d

# # ---------------------------------------------------------------------------
# Simple viewer for previously saved scan files (optional)
# ---------------------------------------------------------------------------
def get_coord(coords, key):
    # Allow either Enum or string
    if hasattr(key, "name"):  # Enum
        return coords.get(key, coords.get(key.name))
    return coords.get(key)
if __name__ == "__main__":
    file_name = "2025_10_23-22_42_56-(lovelace)"

    data = dm.get_raw_data(file_name)
    print("Top-level keys in saved file:")
    print(list(data.keys()))
    # nv_sig = data["nv_sig"]
    # z_coord = nv_sig["coords"]["z"]
    # z_coord = nv_sig.coords[CoordsKey.Z]

    # print(nv_sig)
    # z_coord = get_coord(nv_sig.coords, CoordsKey.Z)
    # print(z_coord)
    img_array = np.array(data["img_array"])
    readout_ns = data["readout_ns"]
    img_array_kcps = (img_array / 1000.0) / (readout_ns * 1e-9)
    extent = data.get("extent", None)

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    _ = kpl.imshow(ax, img_array, cbar_label="Counts", extent=extent)
    ax.set_title(data.get("title", "Saved scan"))
    plt.show(block=True)
