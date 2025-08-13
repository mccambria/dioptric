# -*- coding: utf-8 -*-
"""
Sit on the passed coordinates and record counts.

Updated for NVSig dataclass and VirtualLaserKey-based optics config.
"""

import matplotlib.pyplot as plt
import numpy as np

import utils.kplotlib as kpl
import utils.positioning as pos
import utils.tool_belt as tool_belt
from utils import common
from utils.constants import CoordsKey, VirtualLaserKey


def main(
    nv_sig,
    run_time,
    disable_opt=None,
    nv_minus_init=False,
    nv_zero_init=False,
    background_subtraction=False,
    background_coords=None,  # SAMPLE-space [x, y] (or [x, y, z] — z ignored)
):
    # -------------------- Initial setup --------------------
    if disable_opt is not None:
        nv_sig.disable_opt = disable_opt

    tool_belt.reset_cfm()

    # Imaging readout duration (ns): per-NV override, otherwise config default
    vld_img = tool_belt.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld_img["duration"]))
    )
    readout_sec = readout * 1e-9

    charge_init = bool(nv_minus_init or nv_zero_init)

    pulsegen_server = (
        tool_belt.get_server_pulse_streamer()
    )  # or get_server_pulse_streamer() in your stack
    counter_server = tool_belt.get_server_counter()

    # -------------------- Background subtraction motion (optional) --------------------
    if background_subtraction:
        # Drift-adjust both NV and background target in SAMPLE XY
        nv_xy = pos.adjust_coords_for_drift(nv_sig=nv_sig, coords_key=CoordsKey.SAMPLE)
        nv_x, nv_y = nv_xy[:2] if hasattr(nv_xy, "__len__") else (nv_xy, None)

        if background_coords is None:
            raise ValueError(
                "background_subtraction=True but background_coords is None"
            )

        # Ensure bg is XY list; ignore/strip Z if present
        if hasattr(background_coords, "__len__"):
            bg_xy = list(background_coords[:2])
        else:
            raise ValueError(
                "background_coords must be a 2-element iterable [x, y] (SAMPLE space)"
            )

        bg_xy = pos.adjust_coords_for_drift(coords=bg_xy, coords_key=CoordsKey.SAMPLE)
        bg_x, bg_y = bg_xy[:2]

        x_voltages, y_voltages = pos.get_scan_two_point_2d(nv_x, nv_y, bg_x, bg_y)

        # SAMPLE positioner server (piezo)
        xy_server = pos.get_positioner_server(CoordsKey.SAMPLE)
        # Some drivers accept (x, y, loop=True); prefer 2-arg form and fall back to 3-arg if available
        try:
            xy_server.load_stream_xy(x_voltages, y_voltages)
        except TypeError:
            xy_server.load_stream_xy(x_voltages, y_voltages, True)

    # -------------------- Laser selection / power --------------------
    # Imaging laser (VirtualLaserKey.IMAGING)
    readout_laser = vld_img["physical_name"]
    tool_belt.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tool_belt.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Charge init selection
    if charge_init:
        if nv_minus_init:
            init_key = VirtualLaserKey.CHARGE_POL  # green, polarize to NV-
        elif nv_zero_init:
            init_key = VirtualLaserKey.ION  # red, ionize to NV0
        else:
            init_key = None

        vld_init = tool_belt.get_virtual_laser_dict(init_key)
        init = int(nv_sig.pulse_durations.get(init_key, int(vld_init["duration"])))
        init_laser = vld_init["physical_name"]

        tool_belt.set_filter(nv_sig, init_key)
        init_power = tool_belt.set_laser_power(nv_sig, init_key)

        seq_args = [init, readout, init_laser, init_power, readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        seq_name = "charge_init-simple_readout_background_subtraction.py"
    else:
        delay = 0
        seq_args = [delay, readout, readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        seq_name = "simple_readout.py"

    # Program pulse generator
    period = pulsegen_server.stream_load(seq_name, seq_args_string)[0]  # ns
    total_num_samples = int(run_time / period)
    run_time_s = run_time * 1e-9

    # -------------------- Figure setup --------------------
    samples = np.full(total_num_samples, np.nan, dtype=float)  # NaNs don't get plotted
    write_pos = 0
    x_vals = (np.arange(total_num_samples) + 1) * (period * 1e-9)  # elapsed time in s
    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    kpl.plot_line(ax, x_vals, samples)
    ax.set_xlim(-0.05 * run_time_s, 1.05 * run_time_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count rate (kcps)")
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass

    # -------------------- Acquisition --------------------
    counter_server.start_tag_stream()
    # stream_start(-1): run until stopped
    pulsegen_server.stream_start(-1)
    tool_belt.init_safe_stop()

    leftover_sample = None
    snr = lambda nv, bg: (nv - bg) / np.sqrt(max(nv, 1))  # avoid /0

    def _ensure_1d_counts(arr_like):
        """Flattens list/np arrays of counts to 1D ints."""
        if arr_like is None:
            return np.array([], dtype=int)
        arr = np.array(arr_like)
        if arr.ndim == 0:
            return np.array([int(arr)], dtype=int)
        if arr.dtype != np.int64 and arr.dtype != np.int32:
            arr = arr.astype(int, copy=False)
        # If modulo-gates (N,2) during charge init, we'll diff later
        return arr

    while True:
        if tool_belt.safe_stop():
            break

        # Read new samples
        if charge_init:
            new = counter_server.read_counter_modulo_gates(2)  # Nx2
        else:
            new = counter_server.read_counter_simple()  # N

        new = _ensure_1d_counts(new)

        # Convert modulo-gates to single count per gate
        if charge_init and new.size > 0:
            # Expect Nx2; if flattened, reshape safely
            new = np.array(new)
            new = new.reshape((-1, 2)) if new.ndim == 1 else new
            new = np.maximum(new[:, 0] - new[:, 1], 0)

        # Background subtraction interleave handling
        if background_subtraction and new.size > 0:
            if leftover_sample is not None:
                new = np.insert(new, 0, leftover_sample)
                leftover_sample = None
            if new.size % 2 == 1:
                leftover_sample = int(new[-1])
                new = new[:-1]
            if new.size > 0:
                # pair (NV, BG) -> SNR
                paired = [
                    snr(int(new[2 * i]), int(new[2 * i + 1]))
                    for i in range(new.size // 2)
                ]
                new = np.array(paired, dtype=float)

        n_new = new.size
        if n_new == 0:
            continue

        # Write into circular-ish buffer area: if overflow, drop earliest
        num_written = int(np.count_nonzero(~np.isnan(samples)))
        overflow = (num_written + n_new) - total_num_samples
        if overflow > 0:
            # shift left and append
            keep = total_num_samples - n_new
            keep = max(keep, 0)
            samples[:keep] = samples[num_written - keep : num_written]
            samples[keep:] = new[-n_new:]
            write_pos = total_num_samples
        else:
            samples[write_pos : write_pos + n_new] = new
            write_pos += n_new

        # Update plot in kcps
        samples_kcps = samples / (1e3 * readout_sec)
        kpl.plot_line_update(ax, x=x_vals, y=samples_kcps, relim_x=False)

    # -------------------- Cleanup + stats --------------------
    tool_belt.reset_cfm()

    if write_pos > 0:
        avg = float(np.nanmean(samples[:write_pos])) / (1e3 * readout_sec)
        std = float(np.nanstd(samples[:write_pos])) / (1e3 * readout_sec)
    else:
        avg, std = 0.0, 0.0

    print(f"Average: {avg}")
    print(f"Standard deviation: {std}")
    return avg, std


# # -*- coding: utf-8 -*-
# """
# Sit on the passed coordinates and record counts

# Created on April 12th, 2019

# @author: mccambria
# """

# import labrad
# import matplotlib.pyplot as plt
# import numpy as np

# import majorroutines.targeting as targeting
# import utils.kplotlib as kpl
# import utils.positioning as pos
# import utils.tool_belt as tool_belt


# def main(
#     nv_sig,
#     run_time,
#     disable_opt=None,
#     nv_minus_init=False,
#     nv_zero_init=False,
#     background_subtraction=False,
#     background_coords=None,
# ):
#     ### Initial setup

#     if disable_opt is not None:
#         nv_sig["disable_opt"] = disable_opt
#     tool_belt.reset_cfm()
#     readout = nv_sig["imaging_readout_dur"]
#     readout_sec = readout / 10**9
#     charge_init = nv_minus_init or nv_zero_init
#     # targeting.main_with_cxn(
#     #     nv_sig
#     # )  # Is there something wrong with this line? Let me (Matt) know and let's fix it
#     pulsegen_server = tool_belt.get_server_pulse_streamer()
#     counter_server = tool_belt.get_server_counter()

#     # Background subtraction setup

#     if background_subtraction:
#         drift = np.array(pos.get_drift())
#         adj_coords = np.array(nv_sig["coords"]) + drift
#         adj_bg_coords = np.array(background_coords) + drift
#         x_voltages, y_voltages = pos.get_scan_two_point_2d(
#             adj_coords[0], adj_coords[1], adj_bg_coords[0], adj_bg_coords[1]
#         )
#         xy_server = pos.get_server_positioner_xy()
#         xy_server.load_stream_xy(x_voltages, y_voltages, True)

#     # Imaging laser

#     laser_key = "imaging_laser"
#     readout_laser = nv_sig[laser_key]
#     tool_belt.set_filter(nv_sig, laser_key)
#     readout_power = tool_belt.set_laser_power(nv_sig, laser_key)

#     # Charge init setup and sequence processing
#     if charge_init:
#         if nv_minus_init:
#             laser_key = "nv-_prep_laser"
#         elif nv_zero_init:
#             laser_key = "nv0_prep_laser"
#         tool_belt.set_filter(nv_sig, laser_key)
#         init = nv_sig["{}_dur".format(laser_key)]
#         init_laser = nv_sig[laser_key]
#         init_power = tool_belt.set_laser_power(nv_sig, laser_key)
#         seq_args = [init, readout, init_laser, init_power, readout_laser, readout_power]
#         seq_args_string = tool_belt.encode_seq_args(seq_args)
#         seq_name = "charge_init-simple_readout_background_subtraction.py"
#     else:
#         delay = 0
#         seq_args = [delay, readout, readout_laser, readout_power]
#         seq_args_string = tool_belt.encode_seq_args(seq_args)
#         seq_name = "simple_readout.py"
#     ret_vals = pulsegen_server.stream_load(seq_name, seq_args_string)
#     period = ret_vals[0]

#     total_num_samples = int(run_time / period)
#     run_time_s = run_time * 1e-9

#     # Figure setup
#     samples = np.empty(total_num_samples)
#     samples.fill(np.nan)  # NaNs don't get plotted
#     write_pos = 0
#     x_vals = np.arange(total_num_samples) + 1
#     x_vals = x_vals / (10**9) * period  # Elapsed time in s
#     kpl.init_kplotlib()
#     fig, ax = plt.subplots()
#     kpl.plot_line(ax, x_vals, samples)
#     ax.set_xlim(-0.05 * run_time_s, 1.05 * run_time_s)
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Count rate (kcps)")
#     plt.get_current_fig_manager().window.showMaximized()  # Maximize the window

#     ### Collect the data

#     counter_server.start_tag_stream()
#     pulsegen_server.stream_start(-1)
#     tool_belt.init_safe_stop()
#     # b = 0  # If this just for the OPX, please find a way to implement that does not interfere with other expts
#     leftover_sample = None
#     snr = lambda nv, bg: (nv - bg) / np.sqrt(nv)

#     # Run until user says stop
#     while True:
#         # b = b + 1
#         # if (b % 50) == 0 and (pulsegen_server == "QM_opx"):
#         #     tool_belt.reset_cfm(cxn)
#         #     counter_server.start_tag_stream()
#         #     pulsegen_server.stream_start(-1)
#         #     print("restarting")

#         if tool_belt.safe_stop():
#             break

#         # Read the samples and update the image
#         if charge_init:
#             new_samples = counter_server.read_counter_modulo_gates(2)
#         else:
#             new_samples = counter_server.read_counter_simple()

#         # Read the samples and update the image
#         num_new_samples = len(new_samples)
#         if num_new_samples > 0:
#             # If we did charge init, subtract out the non-initialized count rate
#             if charge_init:
#                 new_samples = [max(int(el[0]) - int(el[1]), 0) for el in new_samples]

#             if background_subtraction:
#                 # Make sure we have an even number of samples
#                 new_samples = np.array(new_samples, dtype=int)
#                 # print(new_samples)
#                 if leftover_sample is not None:
#                     new_samples = np.insert(new_samples, 0, leftover_sample)
#                 if len(new_samples) % 2 == 0:
#                     leftover_sample = None
#                 else:
#                     leftover_sample = new_samples[-1]
#                     new_samples = new_samples[:-1]
#                 new_samples = [
#                     snr(new_samples[2 * ind], new_samples[2 * ind + 1])
#                     for ind in range(num_new_samples // 2)
#                 ]

#             # Update number of new samples to reflect difference-taking etc
#             num_new_net_samples = len(new_samples)
#             num_samples = np.count_nonzero(~np.isnan(samples))

#             # If we're going to overflow, shift everything over and drop earliest samples
#             overflow = (num_samples + num_new_net_samples) - total_num_samples
#             if overflow > 0:
#                 num_nans = max(total_num_samples - num_samples, 0)
#                 samples[::] = np.append(
#                     samples[
#                         num_new_net_samples - num_nans : total_num_samples - num_nans
#                     ],
#                     new_samples,
#                 )
#             else:
#                 cur_write_pos = write_pos
#                 new_write_pos = cur_write_pos + num_new_net_samples
#                 samples[cur_write_pos:new_write_pos] = new_samples
#                 write_pos = new_write_pos

#             # Update the figure in k counts per sec
#             samples_kcps = samples / (10**3 * readout_sec)
#             kpl.plot_line_update(ax, x=x_vals, y=samples_kcps, relim_x=False)

#     ### Clean up and report average and standard deviation

#     tool_belt.reset_cfm()
#     average = np.mean(samples[0:write_pos]) / (10**3 * readout_sec)
#     print(f"Average: {average}")
#     st_dev = np.std(samples[0:write_pos]) / (10**3 * readout_sec)
#     print(f"Standard deviation: {st_dev}")
#     return average, st_dev
