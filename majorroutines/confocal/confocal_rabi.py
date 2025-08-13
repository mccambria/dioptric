# -*- coding: utf-8 -*-
"""
Confocal Rabi experiment using the unified base routine.
Sweeps microwave pulse length (tau); reads signal & reference via APD tagger.

Created on Aug 2, 2025
Author: Saroj Chand (updated & hardened)
"""

import matplotlib.pyplot as plt

# majorroutines/confocal/confocal_rabi.py
import numpy as np

import majorroutines.confocal.confocal_base_routine as base
import utils.confocal_utils as confocal_utils
import utils.data_manager as dm
import utils.kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


def main(nv_sig, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list):
    pulse_streamer = tb.get_server_pulse_streamer()
    seq_file = "rabi_seq.py"

    taus = np.linspace(min_tau, max_tau, int(num_steps)).astype(np.int64)
    max_tau_ns = int(taus.max())
    mw_ind = int(uwave_ind_list[0])

    # Precompute scalars once
    pol_list, _ = confocal_utils.get_pulse_parameter_lists(
        nv_sig, VirtualLaserKey.CHARGE_POL
    )
    pol_ns = int(pol_list[0])
    readout_ns = int(
        tb.get_virtual_laser_dict(VirtualLaserKey.SPIN_READOUT)["duration"]
    )
    readout_laser = tb.get_physical_laser_name(VirtualLaserKey.SPIN_READOUT)
    readout_power = None  # set voltage if analog-modulated

    # These will be filled per run with the shuffled order
    step_args_map = {}

    def run_fn(shuffled_step_inds):
        # Build a packet per step (in the shuffled order we’ll use)
        step_args_map.clear()
        for step_ind in shuffled_step_inds:
            tau_ns = int(taus[step_ind])
            args = [
                tau_ns,
                pol_ns,
                readout_ns,
                max_tau_ns,
                mw_ind,
                readout_laser,
                readout_power,
            ]
            step_args_map[step_ind] = tb.encode_seq_args(args)

    def step_fn(step_ind):
        # Load the sequence for THIS step, then the base routine will call stream_start(num_reps)
        seq_args_string = step_args_map[step_ind]
        pulse_streamer.stream_load(seq_file, seq_args_string)

    raw = base.main(
        nv_sig,
        num_steps=int(num_steps),
        num_reps=int(num_reps),
        num_runs=int(num_runs),
        run_fn=run_fn,
        step_fn=step_fn,  # <<< stream_load happens here, per step
        uwave_ind_list=[mw_ind],
        num_exps=2,  # signal + reference (two APD gates)
        apd_indices=[0],
        load_iq=False,
        stream_load_in_run_fn=False,  # <<< we are NOT loading once per run
        charge_prep_fn=None,
    )
    return raw


# def main(nv_sig, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list):
#     """
#     Run a Rabi experiment by sweeping tau between min_tau and max_tau (ns).
#     """
#     # Pulse streamer handle (kept for parity, but run_fn will do stream_load)
#     pulse_streamer = tb.get_server_pulse_streamer()
#     seq_file = "rabi_seq.py"
#     # Constant-period sweep: pass full tau list to the seq loader via run_fn
#     taus = np.linspace(min_tau, max_tau, num_steps).astype(np.int64)

#     # Per-run loader that shuffles step order and loads the sequence for the run
#     def run_fn(shuffled_step_inds):
#         shuffled_taus = [int(taus[ind]) for ind in shuffled_step_inds]
#         # Let your helper build the base args for rabi_seq.py
#         # (should include max_tau, readout, pol, laser/power, MW state, etc.)
#         seq_args = confocal_utils.get_base_seq_args(
#             nv_sig, uwave_ind_list, shuffled_taus
#         )
#         seq_args_string = tb.encode_seq_args(seq_args)
#         pulse_streamer.stream_load(seq_file, seq_args_string)

#     # Execute unified base routine (handles repetitions/runs/tagger plumbing)
#     raw_data = confocal_base_routine.main(
#         nv_sig,
#         num_steps=num_steps,
#         num_reps=num_reps,
#         num_runs=num_runs,
#         run_fn=run_fn,
#         uwave_ind_list=uwave_ind_list,
#         num_exps=2,  # signal + reference
#         apd_indices=[0],  # adjust for your hardware
#         load_iq=False,
#         stream_load_in_run_fn=True,
#         charge_prep_fn=None,
#     )

#     # ---------- Save raw data ----------
#     timestamp = dm.get_time_stamp()
#     raw_data |= {
#         "timestamp": timestamp,
#         "taus": taus,
#         "tau-units": "ns",
#         "min_tau": int(min_tau),  # fixed: was incorrectly set to max_tau before
#         "max_tau": int(max_tau),
#     }
#     nv_name = getattr(nv_sig, "name", "nv")
#     file_path = dm.get_file_path(__file__, timestamp, nv_name)
#     dm.save_raw_data(raw_data, file_path)

#     # ---------- Optional processing & plotting ----------
#     try:
#         raw_fig = None
#         fit_fig = None

#         counts = raw_data["counts"]  # shape: (2, num_runs, num_steps, num_reps)
#         # Readout duration (ns) for SPIN_READOUT
#         vld = tb.get_virtual_laser_dict(VirtualLaserKey.SPIN_READOUT)
#         readout_ns = int(
#             nv_sig.pulse_durations.get(
#                 VirtualLaserKey.SPIN_READOUT, int(vld["duration"])
#             )
#         )

#         sig_counts = counts[0]
#         ref_counts = counts[1]
#         (
#             norm_counts,
#             norm_counts_ste,
#             sig_counts_avg_kcps,
#             ref_counts_avg_kcps,
#         ) = confocal_utils.process_counts(nv_sig, sig_counts, ref_counts, readout_ns)

#         # If you have figure builders, you can enable them here:
#         # raw_fig = create_raw_data_figure(nv_sig, taus, norm_counts, norm_counts_ste)
#         # fit_fig = create_fit_figure(nv_sig, taus, norm_counts, norm_counts_ste)

#     except Exception:
#         raw_fig = None
#         fit_fig = None

#     if raw_fig is not None:
#         dm.save_figure(raw_fig, file_path)
#     if fit_fig is not None:
#         fit_path = dm.get_file_path(__file__, timestamp, nv_name + "-fit")
#         dm.save_figure(fit_fig, fit_path)

#     # Finish cleanly
#     tb.reset_cfm()
#     kpl.show()


if __name__ == "__main__":
    pass


# def main(nv_sig, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list):
#     ### Some initial setup

#     pulse_streamer = tb.get_server_pulse_streamer()
#     seq_file = "rabi_seq.py"
#     taus = np.linspace(min_tau, max_tau, num_steps)

#     ### Collect the data
#     def run_fn(shuffled_step_inds):
#         shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
#         seq_args = confocal_utils.get_base_seq_args(
#             nv_sig, uwave_ind_list, shuffled_taus
#         )
#         seq_args_string = tb.encode_seq_args(seq_args)
#         pulse_streamer.stream_load(seq_file, seq_args_string, num_reps)

#     raw_data = confocal_base_routine.main(
#         nv_sig,
#         num_steps,
#         num_reps,
#         num_runs,
#         run_fn=run_fn,
#         uwave_ind_list=uwave_ind_list,
#     )

#     ### save the rawa data
#     timestamp = dm.get_time_stamp()
#     raw_data |= {
#         "timestamp": timestamp,
#         "taus": taus,
#         "tau-units": "ns",
#         "min_tau": max_tau,
#         "max_tau": max_tau,
#     }
#     # save the raw data
#     nv_name = nv_sig.name
#     file_path = dm.get_file_path(__file__, timestamp, nv_name)
#     dm.save_raw_data(raw_data, file_path)

#     ### Process and plot

#     try:
#         raw_fig = None
#         fit_fig = None
#         counts = raw_data["counts"]
#         readout = nv_sig["spin_readout_dur"]
#         sig_counts = counts[0]
#         ref_counts = counts[1]
#         norm_counts, norm_counts_ste, sig_counts_avg_kcps, ref_counts_avg_kcps = (
#             confocal_utils.process_counts(nv_sig, sig_counts, ref_counts, readout)
#         )

#         # raw_fig = create_raw_data_figure(nv_sig, taus, norm_counts, norm_counts_ste)
#         # fit_fig = create_fit_figure(nv_sig, taus, norm_counts, norm_counts_ste)
#     except Exception:
#         # print(traceback.format_exc())
#         raw_fig = None
#         fit_fig = None

#     if raw_fig is not None:
#         dm.save_figure(raw_fig, file_path)
#     if fit_fig is not None:
#         file_path = dm.get_file_path(__file__, timestamp, nv_name + "-fit")
#         dm.save_figure(fit_fig, file_path)
#     ### Clean up and return

#     tb.reset_cfm()
#     kpl.show()


if __name__ == "__main__":
    path = "pc_rabi/branch_master/rabi/2023_01"
    file = "2023_01_27-09_42_22-siena-nv4_2023_01_16"
    data = tb.get_raw_data(file, path)
    kpl.init_kplotlib()

    norm_avg_sig = data["norm_avg_sig"]
    uwave_time_range = data["uwave_time_range"]
    num_steps = data["num_steps"]
    uwave_freq = data["uwave_freq"]
    norm_avg_sig_ste = None

    # fit_func = tool_belt.cosexp_1_at_0

    sig_counts = data["sig_counts"]
    ref_counts = data["ref_counts"]
    num_reps = data["num_reps"]
    nv_sig = data["nv_sig"]
    readout = nv_sig["spin_readout_dur"]
    ret_vals = tb.process_counts(sig_counts, ref_counts, num_reps, readout)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals

# %%
