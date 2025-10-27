# -*- coding: utf-8 -*-
"""
2D scan over a designated area, collecting photon counts at each point.
Generates an image of the sample in COUNTER mode.

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

def confocal_scan(nv_sig: NVSig, x_range, y_range, num_steps, nv_minus_init=False):
    cfg = common.get_config_dict()
    count_fmt: CountFormat = cfg["count_format"]

    positioner = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
    mode = pos.get_positioner_control_mode(positioner)

    # Build grid (1D coordinate axes + random-access XY)
    x0, y0 = pos.get_nv_coords(nv_sig, coords_key=CoordsKey.PIXEL)
    X, Y, x1d, y1d, extent = pos.get_scan_grid_2d(x0, y0, x_range, y_range, num_steps, num_steps)
    total = len(X)

    # Readout timing/laser from virtual laser
    h = w = num_steps
    total = h * w
    # --- UI throttling: update plot every N pixels (about ~8 updates/row) ---
    UPDATE_EVERY = 1
    pixels_done = 0
    # Timing / laser
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"])))
    readout_s = readout_ns / 1e9
    readout_laser = vld["physical_name"]

    pulse = tb.get_server_pulse_streamer()
    ctr   = tb.get_server_counter()

    img = np.full((h, w), np.nan, float)
    img_kcps = np.copy(img) if count_fmt == CountFormat.KCPS else None

    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Voltage (V)")
    kpl.imshow(ax, img if img_kcps is None else img_kcps,
            #    title=f"XY={nv_sig.coords[CoordsKey.SAMPLE]}, Z={nv_sig.coords[CoordsKey.Z]}, {ccryo.get_sample_name()}",
               title=f"{readout_laser}, {readout_ns/1e6:.1f} ms",
               cbar_label=("Kcps" if img_kcps is not None else "Raw Counts"),
               extent=extent)

    tb.reset_cfm()
    tb.init_safe_stop()
    ctr.start_tag_stream()

    # ---- sequence loading (hardware delay, not Python sleep) ----
    pos_key = CoordsKey.PIXEL
    delay_ns = int(cfg["Positioning"]["Positioners"][pos_key]["delay"])  # e.g. 400e3 ns for galvo
    period_ns = pulse.stream_load(
        SEQ_FILE_PIXEL_READOUT,
        tb.encode_seq_args([delay_ns, readout_ns, readout_laser, 1.0])  # laser args ignored by your seq
    )[0]

    try:
        if mode == PosControlMode.SEQUENCE:
            # ---- RASTER ORDER COORDINATE LISTS (no serpentine) ----
            Xr = []
            Yr = []
            for row in range(h):
                y = y1d[(h - 1) - row]   # bottom row first
                for col in range(w):
                    x = x1d[col]        # left -> right always
                    Xr.append(x); Yr.append(y)

            seq_args = [0, readout_ns, list(Xr), list(Yr)]   # delay handled in pixel-readout mode only
            period_ns = pulse.stream_load(SEQ_FILE_SEQUENCE_SCAN, tb.encode_seq_args(seq_args))[0]
            pulse.stream_start(total)

            written = []  # state for _raster_fill
            done = 0
            # conservative timeout
            timeout = time.time() + (period_ns * 1e-9) * total + 10.0

            while done < total and not tb.safe_stop():
                if time.time() > timeout: break
                if nv_minus_init:
                    raw = ctr.read_counter_modulo_gates(2)
                    vals = [max(int(a)-int(b), 0) for (a,b) in raw]
                else:
                    raw = ctr.read_counter_simple()
                    vals = [int(v) for v in raw]

                if not vals:
                    continue

                _raster_fill(vals, img, written)
                done = written[0]

                if img_kcps is not None:
                    img_kcps[:] = (img/1000.0)/readout_s
                    kpl.imshow_update(ax, img_kcps)
                else:
                    kpl.imshow_update(ax, img)

        else:
            # ---- STEP/STREAM CONTROL: drive stage in raster order ----
            # pulse.stream_start(total)

            written = []  # state for _raster_fill
            for row in range(h):
                if tb.safe_stop(): break
                y = y1d[(h - 1) - row]  # bottom row first
                for col in range(w):
                    if tb.safe_stop(): break
                    x = x1d[col]        # left -> right
                    pos.set_xyz((x, y), positioner=positioner)

                    pulse.stream_start(1)  # <<< moved here
                    # read exactly ONE sample per pixel
                    if nv_minus_init:
                        raw = ctr.read_counter_modulo_gates(2, 1)  # [[a,b]]
                        vals = [max(int(raw[0][0]) - int(raw[0][1]), 0)]
                    else:
                        raw = ctr.read_counter_simple(1)            # [c]
                        vals = [int(raw[0])]

                    if not vals:
                        continue

                    _raster_fill(vals, img, written)

                    # if img_kcps is not None:
                    #     img_kcps[:] = (img/1000.0)/readout_s
                    #     kpl.imshow_update(ax, img_kcps)
                    # else:
                    #     kpl.imshow_update(ax, img)

                    # UI throttle
                    pixels_done = written[0]
                    if (pixels_done % UPDATE_EVERY) == 0:
                        if img_kcps is not None:
                            img_kcps[:] = (img / 1000.0) / readout_s
                            kpl.imshow_update(ax, img_kcps)
                        else:
                            kpl.imshow_update(ax, img)

    finally:
        ctr.clear_buffer()
        tb.reset_cfm()

    # ---- Save (units consistent with display/return) ----
    is_kcps = (count_fmt == CountFormat.KCPS)
    img_out = (img/1000.0)/readout_s if is_kcps else img
    units_out = "kcps" if is_kcps else "counts"

    ts = dm.get_time_stamp()
    # raw = dict(
    #     timestamp=ts,
    #     "nv_sig": nv_sig,
    #     mode="confocal_scan_raster",
    #     num_steps=num_steps,
    #     x_range=x_range, y_range=y_range,
    #     x_center=x0, y_center=y0,
    #     extent=extent,
    #     readout_ns=readout_ns, readout_units="ns",
    #     img_array=img_out.astype(float), img_array_units=units_out,
    #     x_coords_1d=x1d, y_coords_1d=y1d,
    #     nv_sig=nv_sig.__dict__ if hasattr(nv_sig, "__dict__") else nv_sig,
    # )
    raw = {
    "timestamp": ts,
    "nv_sig": nv_sig,
    "num_steps": num_steps,
    "x_range": x_range, "y_range": y_range,
    "x_center": x0, "y_center": y0,
    "extent": extent,
    "readout_ns": readout_ns, "readout_units": "ns",
    "img_array": img_out.astype(float), "img_array_units": units_out,
    "x_coords_1d": x1d, "y_coords_1d": y1d,
    }
    path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_figure(fig, path)
    dm.save_raw_data(raw, path)
    kpl.show(block=True)
    return img, x1d, y1d


# class ScanAxes(Enum):
#     XY = auto()
#     XZ = auto()
#     YZ = auto()


# def populate_img_array(vals_to_add, img_array, write_pos):
#     """
#     Snake-fill image array in real time as new pixel values arrive.
#     Moves the cursor BEFORE writing (avoids x==x_dim on the first write).
#     Also stops cleanly if extra samples arrive beyond the image size.
#     """
#     y_dim, x_dim = img_array.shape

#     # Initialize cursor: one past the right edge, bottom row
#     if not write_pos:
#         write_pos[:] = [x_dim, y_dim - 1]

#     x, y = write_pos
#     heading_left = (y_dim - 1 - y) % 2 == 0  # even rows go left, odd go right

#     for val in vals_to_add:
#         # Move first, then write
#         if heading_left:
#             if x == 0:
#                 # step up a row and flip direction
#                 y -= 1
#                 heading_left = False
#             else:
#                 x -= 1
#         else:
#             if x == x_dim - 1:
#                 # step up a row and flip direction
#                 y -= 1
#                 heading_left = True
#             else:
#                 x += 1

#         # If we've run out of rows (extra samples), stop gracefully
#         if y < 0:
#             break

#         img_array[y, x] = val

#     write_pos[:] = [x, y]
#     return img_array


# def main(
#     nv_sig: NVSig,
#     x_range,
#     y_range,
#     num_steps,
#     nv_minus_init: bool = False,
#     scan_axes: ScanAxes = ScanAxes.XY,
# ):
#     """
#     Perform a 2D scan and return the image array and coordinate axes.
#     """
#     # -------------------- Setup --------------------
#     config = common.get_config_dict()
#     count_format: CountFormat = config["count_format"]

#     laser_key = VirtualLaserKey.IMAGING  # imaging virtual laser
#     positioner = pos.get_laser_positioner(laser_key)
#     print(f"pistioner for {laser_key.name}: {positioner}")

#     # print(f"Using positioner: {positioner}")
#     # Get XY center in the imaging (galvo/AOD) space.
#     # If per-laser coords are missing, your modified get_nv_coords will derive from PIXEL/SAMPLE.
#     x_im, y_im = pos.get_nv_coords(nv_sig, coords_key=CoordsKey.PIXEL)

#     # Build scan grid IN THE SAME SPACE WE'LL MOVE (imaging positioner)
#     coords_1, coords_2, coords_1_1d, coords_2_1d, extent = pos.get_scan_grid_2d(
#         x_im, y_im, x_range, y_range, num_steps, num_steps
#     )
#     total_samples = len(coords_1)

#     # Reset/center hardware to NV in SAMPLE space (piezo), applying drift compensation
#     tb.reset_cfm()
#     pos.set_xyz_on_nv(nv_sig, positioner=CoordsKey.SAMPLE)

#     # Servers
#     counter = tb.get_server_counter()
#     pulse_gen = (
#         tb.get_server_pulse_streamer()
#     )  # keep this consistent with your tool_belt

#     # Laser readout settings: config default with per-NV override
#     vld = tb.get_virtual_laser_dict(
#         laser_key
#     )  # {"physical_name": "...", "duration": ...}
#     readout_laser = vld["physical_name"]
#     tb.set_filter(nv_sig, laser_key)
#     readout_power = tb.set_laser_power(nv_sig, laser_key)
#     readout_ns = int(nv_sig.pulse_durations.get(laser_key, int(vld["duration"])))
#     readout_sec = readout_ns / 1e9

#     # Image array
#     img_array = np.full((num_steps, num_steps), np.nan, dtype=float)
#     img_array_kcps = np.copy(img_array) if count_format == CountFormat.KCPS else None
#     img_write_pos = []

#     # Display setup
#     kpl.init_kplotlib(font_size=kpl.Size.SMALL)
#     cbar_label = "Kcps" if count_format == CountFormat.KCPS else "Counts"
#     title = (
#         f"{scan_axes.name} scan ({readout_laser}, {readout_ns / 1e3:.1f} μs readout)"
#     )
#     fig, ax = plt.subplots()
#     kpl.imshow(ax, img_array, title=title, cbar_label=cbar_label, extent=extent)

#     # -------------------- Sequence load --------------------
#     # Choose control mode for XY from the imaging positioner, else STEP for XZ/YZ
#     control_mode = (
#         pos.get_positioner_control_mode(positioner)
#         if scan_axes == ScanAxes.XY
#         else PosControlMode.STEP
#     )

#     if control_mode == PosControlMode.SEQUENCE and scan_axes == ScanAxes.XY:
#         # Galvo/AOD path uses a scanning sequence that takes the XY arrays
#         seq_file = (
#             "simple_readout_laser_free_test.py"  # laser-free scanning seq for counter
#         )
#         seq_args = [0, readout_ns, list(coords_1), list(coords_2)]
#     else:
#         # Piezo or non-sequence control uses simple readout timing
#         if nv_minus_init:
#             seq_file = "charge_init-simple_readout.py"
#             seq_args = [0, readout_ns, "unused", 0.0, readout_laser, readout_power]
#         else:
#             seq_file = "simple_readout.py"
#             seq_args = [0, readout_ns]

#     period = pulse_gen.stream_load(seq_file, tb.encode_seq_args(seq_args))[0]

#     # -------------------- Acquisition loop --------------------
#     tb.init_safe_stop()
#     counter.start_tag_stream()

#     def _num_samples(x):
#         import numpy as _np

#         if x is None:
#             return 0
#         if isinstance(x, _np.ndarray):
#             return x.size if x.ndim == 1 else x.shape[0]
#         # list/tuple
#         try:
#             return len(x)
#         except Exception:
#             return 0

#     def process_samples(samples):
#         # accept list or numpy array
#         if nv_minus_init:
#             # shape (N, 2) expected
#             # works for list-of-pairs or Nx2 numpy arrays
#             samples = [max(int(s[0]) - int(s[1]), 0) for s in samples]
#         populate_img_array(samples, img_array, img_write_pos)
#         if img_array_kcps is not None:
#             img_array_kcps[:] = (img_array / 1000.0) / readout_sec
#             kpl.imshow_update(ax, img_array_kcps)
#         else:
#             kpl.imshow_update(ax, img_array)

#     try:
#         pulse_gen.stream_start(total_samples)
#         timeout = time.time() + (period * 1e-9) * total_samples + 10
#         num_read = 0
#         while num_read < total_samples:
#             if time.time() > timeout or tb.safe_stop():
#                 break

#             samples = (
#                 counter.read_counter_modulo_gates(2)
#                 if nv_minus_init
#                 else counter.read_counter_simple()
#             )

#             n_new = _num_samples(samples)
#             if n_new == 0:
#                 continue

#             process_samples(samples)
#             num_read += n_new

#     finally:
#         counter.clear_buffer()
#         tb.reset_cfm()
#         # pos.set_xyz((x_samp, y_samp, z_center), positioner=CoordsKey.SAMPLE)

#     # -------------------- Save --------------------
#     timestamp = dm.get_time_stamp()
#     raw_data = dict(
#         timestamp=timestamp,
#         nv_sig=nv_sig,
#         x_center=x_im,
#         y_center=y_im,
#         # z_center=z_center,
#         x_range=x_range,
#         y_range=y_range,
#         num_steps=num_steps,
#         extent=extent,
#         scan_axes=scan_axes.name,
#         readout=readout_ns,
#         readout_units="ns",
#         title=title,
#         coords_1_1d=coords_1_1d,
#         coords_2_1d=coords_2_1d,
#         img_array=img_array.astype(int),
#         img_array_units="counts",
#         control_mode_used=control_mode.name,
#         seq_file=seq_file,
#     )
#     file_path = dm.get_file_path(
#         __file__, timestamp, nv_sig.name if hasattr(nv_sig, "name") else "nv"
#     )
#     dm.save_figure(fig, file_path)
#     dm.save_raw_data(raw_data, file_path)

#     kpl.show(block=True)
#     return img_array, coords_1_1d, coords_2_1d

# ---------------------------------------------------------------------------
    kpl.show()
    return img_out, x1d, y1d

# # ---------------------------------------------------------------------------
# Simple viewer for previously saved scan files (optional)
# ---------------------------------------------------------------------------
def get_coord(coords, key):
    # Allow either Enum or string
    if hasattr(key, "name"):  # Enum
        return coords.get(key, coords.get(key.name))
    return coords.get(key)
if __name__ == "__main__":
    file_name = "2025_10_25-11_09_07-(lovelace)"

    data = dm.get_raw_data(file_name)
    print("Top-level keys in saved file:")
    print(list(data.keys()))
    nv_sig = data["nv_sig"]
    # z_coord = nv_sig["coords"]["z"]
    # z_coord = nv_sig.coords[CoordsKey.Z]

    print(nv_sig)
    z_coord = get_coord(nv_sig.coords, CoordsKey.Z)
    print(z_coord)
    img_array = np.array(data["img_array"])
    readout_ns = data["readout_ns"]
    img_array_kcps = (img_array / 1000.0) / (readout_ns * 1e-9)
    extent = data.get("extent", None)

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    _ = kpl.imshow(ax, img_array, cbar_label="Counts", extent=extent)
    ax.set_title(data.get("title", "Saved scan"))
    plt.show(block=True)
