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


class ScanAxes(Enum):
    XY = auto()
    XZ = auto()
    YZ = auto()


def populate_img_array(vals_to_add, img_array, write_pos):
    """
    Snake-fill image array in real time as new pixel values arrive.
    Moves the cursor BEFORE writing (avoids x==x_dim on the first write).
    Also stops cleanly if extra samples arrive beyond the image size.
    """
    y_dim, x_dim = img_array.shape

    # Initialize cursor: one past the right edge, bottom row
    if not write_pos:
        write_pos[:] = [x_dim, y_dim - 1]

    x, y = write_pos
    heading_left = (y_dim - 1 - y) % 2 == 0  # even rows go left, odd go right

    for val in vals_to_add:
        # Move first, then write
        if heading_left:
            if x == 0:
                # step up a row and flip direction
                y -= 1
                heading_left = False
            else:
                x -= 1
        else:
            if x == x_dim - 1:
                # step up a row and flip direction
                y -= 1
                heading_left = True
            else:
                x += 1

        # If we've run out of rows (extra samples), stop gracefully
        if y < 0:
            break

        img_array[y, x] = val

    write_pos[:] = [x, y]
    return img_array


def main(
    nv_sig: NVSig,
    x_range,
    y_range,
    num_steps,
    nv_minus_init: bool = False,
    scan_axes: ScanAxes = ScanAxes.XY,
):
    """
    Perform a 2D scan and return the image array and coordinate axes.
    """
    # -------------------- Setup --------------------
    config = common.get_config_dict()
    count_format: CountFormat = config["count_format"]

    laser_key = VirtualLaserKey.IMAGING  # imaging virtual laser
    positioner = pos.get_laser_positioner(laser_key)
    # print(f"Using positioner: {positioner}")
    # Get XY center in the imaging (galvo/AOD) space.
    # If per-laser coords are missing, your modified get_nv_coords will derive from PIXEL/SAMPLE.
    x_im, y_im = pos.get_nv_coords(nv_sig, coords_key=CoordsKey.PIXEL)

    # Build scan grid IN THE SAME SPACE WE'LL MOVE (imaging positioner)
    coords_1, coords_2, coords_1_1d, coords_2_1d, extent = pos.get_scan_grid_2d(
        x_im, y_im, x_range, y_range, num_steps, num_steps
    )
    total_samples = len(coords_1)

    # Reset/center hardware to NV in SAMPLE space (piezo), applying drift compensation
    # tb.reset_cfm()
    pos.set_xyz_on_nv(nv_sig, positioner=CoordsKey.SAMPLE)

    # Servers
    counter = tb.get_server_counter()
    pulse_gen = (
        tb.get_server_pulse_streamer()
    )  # keep this consistent with your tool_belt

    # Laser readout settings: config default with per-NV override
    vld = tb.get_virtual_laser_dict(
        laser_key
    )  # {"physical_name": "...", "duration": ...}
    readout_laser = vld["physical_name"]
    tb.set_filter(nv_sig, laser_key)
    readout_power = tb.set_laser_power(nv_sig, laser_key)
    readout_ns = int(nv_sig.pulse_durations.get(laser_key, int(vld["duration"])))
    readout_sec = readout_ns / 1e9

    # Image array
    img_array = np.full((num_steps, num_steps), np.nan, dtype=float)
    img_array_kcps = np.copy(img_array) if count_format == CountFormat.KCPS else None
    img_write_pos = []

    # Display setup
    kpl.init_kplotlib(font_size=kpl.Size.SMALL)
    cbar_label = "Kcps" if count_format == CountFormat.KCPS else "Counts"
    title = (
        f"{scan_axes.name} scan ({readout_laser}, {readout_ns / 1e3:.1f} μs readout)"
    )
    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array, title=title, cbar_label=cbar_label, extent=extent)

    # -------------------- Sequence load --------------------
    # Choose control mode for XY from the imaging positioner, else STEP for XZ/YZ
    control_mode = (
        pos.get_positioner_control_mode(positioner)
        if scan_axes == ScanAxes.XY
        else PosControlMode.STEP
    )

    if control_mode == PosControlMode.SEQUENCE and scan_axes == ScanAxes.XY:
        # Galvo/AOD path uses a scanning sequence that takes the XY arrays
        seq_file = (
            "simple_readout_laser_free_test.py"  # laser-free scanning seq for counter
        )
        seq_args = [0, readout_ns, list(coords_1), list(coords_2)]
    else:
        # Piezo or non-sequence control uses simple readout timing
        if nv_minus_init:
            seq_file = "charge_init-simple_readout.py"
            seq_args = [0, readout_ns, "unused", 0.0, readout_laser, readout_power]
        else:
            seq_file = "simple_readout.py"
            seq_args = [0, readout_ns]

    period = pulse_gen.stream_load(seq_file, tb.encode_seq_args(seq_args))[0]

    # -------------------- Acquisition loop --------------------
    tb.init_safe_stop()
    counter.start_tag_stream()

    def _num_samples(x):
        import numpy as _np

        if x is None:
            return 0
        if isinstance(x, _np.ndarray):
            return x.size if x.ndim == 1 else x.shape[0]
        # list/tuple
        try:
            return len(x)
        except Exception:
            return 0

    def process_samples(samples):
        # accept list or numpy array
        if nv_minus_init:
            # shape (N, 2) expected
            # works for list-of-pairs or Nx2 numpy arrays
            samples = [max(int(s[0]) - int(s[1]), 0) for s in samples]
        populate_img_array(samples, img_array, img_write_pos)
        if img_array_kcps is not None:
            img_array_kcps[:] = (img_array / 1000.0) / readout_sec
            kpl.imshow_update(ax, img_array_kcps)
        else:
            kpl.imshow_update(ax, img_array)

    try:
        pulse_gen.stream_start(total_samples)
        timeout = time.time() + (period * 1e-9) * total_samples + 10
        num_read = 0
        while num_read < total_samples:
            if time.time() > timeout or tb.safe_stop():
                break

            samples = (
                counter.read_counter_modulo_gates(2)
                if nv_minus_init
                else counter.read_counter_simple()
            )

            n_new = _num_samples(samples)
            if n_new == 0:
                continue

            process_samples(samples)
            num_read += n_new

    finally:
        counter.clear_buffer()
        tb.reset_cfm()
        # pos.set_xyz((x_samp, y_samp, z_center), positioner=CoordsKey.SAMPLE)

    # -------------------- Save --------------------
    timestamp = dm.get_time_stamp()
    raw_data = dict(
        timestamp=timestamp,
        nv_sig=nv_sig,
        x_center=x_im,
        y_center=y_im,
        # z_center=z_center,
        x_range=x_range,
        y_range=y_range,
        num_steps=num_steps,
        extent=extent,
        scan_axes=scan_axes.name,
        readout=readout_ns,
        readout_units="ns",
        title=title,
        coords_1_1d=coords_1_1d,
        coords_2_1d=coords_2_1d,
        img_array=img_array.astype(int),
        img_array_units="counts",
        control_mode_used=control_mode.name,
        seq_file=seq_file,
    )
    file_path = dm.get_file_path(
        __file__, timestamp, nv_sig.name if hasattr(nv_sig, "name") else "nv"
    )
    dm.save_figure(fig, file_path)
    dm.save_raw_data(raw_data, file_path)

    kpl.show(block=True)
    return img_array, coords_1_1d, coords_2_1d


# ---------------------------------------------------------------------------
# Simple viewer for previously saved scan files (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    file_name = "2023_09_11-13_52_01-johnson-nvref"

    data = tb.get_raw_data(file_name)
    img_array = np.array(data["img_array"])
    readout_ns = data["readout"]
    img_array_kcps = (img_array / 1000.0) / (readout_ns * 1e-9)
    extent = data.get("extent", None)

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    _ = kpl.imshow(ax, img_array, cbar_label="Counts", extent=extent)
    ax.set_title(data.get("title", "Saved scan"))
    plt.show(block=True)
