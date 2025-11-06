# -*- coding: utf-8 -*-
"""
1D Z-axis scan with photon counting and real-time plotting.

Performs a scan along the Z-axis without moving X/Y position, collecting
photon counts at each Z position and displaying a line plot.

Created on November 5th, 2025

"""

import time

import matplotlib.pyplot as plt
import numpy as np

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CountFormat, CoordsKey, NVSig, VirtualLaserKey

SEQ_FILE_PIXEL_READOUT = "simple_readout.py"


def main(
    nv_sig: NVSig,
    z_start,
    z_end,
    num_steps,
    num_averages=1,
    nv_minus_init=False,
    save_data=True,
):
    """
    Perform a 1D scan along the Z-axis, collecting photon counts at each position.

    This routine:
    - Does NOT move X or Y coordinates
    - Scans only the Z-axis from z_start to z_end
    - Collects photon counts at each Z position
    - Displays real-time line plot of counts vs Z position
    - Saves data and plot

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters
    z_start : int
        Starting Z position in steps
    z_end : int
        Ending Z position in steps
    num_steps : int
        Number of Z positions to scan
    num_averages : int
        Number of photon count samples to average at each Z position (default: 1)
    nv_minus_init : bool
        Whether to use charge initialization (default: False)
    save_data : bool
        Whether to save the data and plot (default: True)

    Returns
    -------
    tuple
        (counts_array, z_positions) - counts in kcps or raw depending on config
    """

    ### Setup

    cfg = common.get_config_dict()
    count_fmt: CountFormat = cfg["count_format"]

    # Get servers
    piezo = common.get_server("pos_xyz_ATTO_piezos")
    ctr = tb.get_server_counter()
    pulse = tb.get_server_pulse_streamer()

    # Setup laser for imaging
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"]))
    )
    readout_s = readout_ns * 1e-9
    readout_laser = vld["physical_name"]

    tb.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tb.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Create Z scan array
    z_positions = np.linspace(z_start, z_end, num_steps)
    counts_1d = np.full(num_steps, np.nan, float)
    counts_1d_kcps = np.copy(counts_1d) if count_fmt == CountFormat.KCPS else None

    ### Setup figure for real-time plotting

    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Z Position (steps)")
    ax.set_ylabel("Kcps" if count_fmt == CountFormat.KCPS else "Raw Counts")
    ax.set_title(f"{readout_laser}, {readout_ns/1e6:.1f} ms readout")

    # Initial plot with NaN values
    (line,) = ax.plot(
        z_positions,
        counts_1d if counts_1d_kcps is None else counts_1d_kcps,
        "b.-",
        markersize=4,
    )
    ax.set_xlim(z_positions[0], z_positions[-1])
    ax.grid(True, alpha=0.3)

    tb.reset_cfm()
    tb.init_safe_stop()

    # Sequence loading (same as stationary_count - load once)
    delay_ns = 0  # No delay for continuous counting
    period_ns = pulse.stream_load(
        SEQ_FILE_PIXEL_READOUT,
        tb.encode_seq_args([delay_ns, readout_ns, readout_laser, readout_power]),
    )[0]

    # Start continuous streaming (same pattern as stationary_count)
    ctr.start_tag_stream()
    pulse.stream_start(-1)  # -1 means run continuously until stopped

    ### Scan through Z positions

    try:
        for i, z_pos in enumerate(z_positions):
            if tb.safe_stop():
                break

            # Move to Z position
            piezo.write_z(int(z_pos))
            time.sleep(0.01)  # Settling time

            # Clear buffer and collect fresh counts
            ctr.clear_buffer()
            time.sleep(0.01)  # Brief wait for buffer to fill

            # Read counts (collect num_averages samples)
            counts = []
            read_start = time.time()
            timeout = 2.0

            while len(counts) < num_averages:
                if tb.safe_stop():
                    break
                if time.time() - read_start > timeout:
                    break
                new_samples = ctr.read_counter_simple()
                if len(new_samples) > 0:
                    counts.extend(new_samples)

            # Calculate mean
            if nv_minus_init and len(counts) >= 2:
                # For charge init, pair samples and subtract
                val = max(int(counts[0]) - int(counts[1]), 0) if len(counts) >= 2 else 0
            else:
                val = int(np.mean(counts)) if len(counts) > 0 else 0

            counts_1d[i] = val

            # Update plot with auto-scaling
            if counts_1d_kcps is not None:
                counts_1d_kcps[:] = (counts_1d / 1000.0) / readout_s
                line.set_ydata(counts_1d_kcps)
            else:
                line.set_ydata(counts_1d)

            # Auto-scale Y axis based on valid data
            valid_data = counts_1d_kcps if counts_1d_kcps is not None else counts_1d
            valid_mask = ~np.isnan(valid_data)
            if np.any(valid_mask):
                ax.set_ylim(np.nanmin(valid_data) * 0.9, np.nanmax(valid_data) * 1.1)

            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()

    finally:
        ctr.clear_buffer()
        tb.reset_cfm()

    ### Save data

    is_kcps = count_fmt == CountFormat.KCPS
    counts_out = (counts_1d / 1000.0) / readout_s if is_kcps else counts_1d
    units_out = "kcps" if is_kcps else "counts"

    ts = dm.get_time_stamp()
    raw_data = {
        "timestamp": ts,
        "nv_sig": nv_sig,
        "mode": "z_scan_1d",
        "num_steps": num_steps,
        "num_averages": num_averages,
        "z_start": z_start,
        "z_end": z_end,
        "readout_ns": readout_ns,
        "readout_units": "ns",
        "counts_array": counts_out.astype(float).tolist(),
        "counts_array_units": units_out,
        "z_positions": z_positions.tolist(),
    }

    if save_data:
        path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
        dm.save_figure(fig, path)
        dm.save_raw_data(raw_data, path)

    kpl.show()

    return counts_out, z_positions


if __name__ == "__main__":
    # Example usage
    from utils.constants import CoordsKey, NVSig

    nv_sig = NVSig(
        name="test_z_scan",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.Z: 0},
        pulse_durations={VirtualLaserKey.IMAGING: int(5e6)},
    )

    results = main(
        nv_sig,
        z_start=100,
        z_end=-300,
        num_steps=50,
        num_averages=100,
    )
