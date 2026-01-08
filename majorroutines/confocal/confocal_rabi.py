# -*- coding: utf-8 -*-
"""
Confocal Rabi experiment using the unified base routine.
Sweeps microwave pulse length (tau); reads signal & reference via APD tagger.

Created on Aug 2, 2025
Author: Saroj Chand (updated & hardened)

Updated on Jan 6, 2026:
- Added raw-data saving (data_manager)
- Added kplotlib-style plotting
- Added basic FFT-seeded cosine fit (optional) + figure saving
"""

# majorroutines/confocal/confocal_rabi.py

import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.confocal.confocal_base_routine as base
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import NormMode, VirtualLaserKey


def _vkey_from_arg(x):
    if x is None:
        return None
    if isinstance(x, VirtualLaserKey):
        return x
    if isinstance(x, str):
        name = x.split(".")[-1]
        return VirtualLaserKey[name]
    raise TypeError(f"Bad virtual laser key: {x!r}")


def _get_nv_name(nv_sig):
    # Works for NVSig dataclass/object or dict-like
    if hasattr(nv_sig, "name"):
        return nv_sig.name
    try:
        return nv_sig["name"]
    except Exception:
        return "confocal_nv"


def get_base_seq_args_ps(
    nv_sig,
    uwave_list,
    max_tau,
    readout_laser=None,
    readout_power=None,
):
    """
    Returns base_args for rabi_seq.get_seq:
      [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_tau]
    """
    cfg = common.get_config_dict()

    # Prefer nv_sig.pulse_durations when available (NVSig-style)
    pol_ns = None
    readout_ns = None
    if hasattr(nv_sig, "pulse_durations") and isinstance(nv_sig.pulse_durations, dict):
        pol_ns = nv_sig.pulse_durations.get(VirtualLaserKey.CHARGE_POL, None)
        readout_ns = nv_sig.pulse_durations.get(VirtualLaserKey.SPIN_READOUT, None)

    if pol_ns is None:
        pol_ns = int(
            cfg["Optics"]["VirtualLasers"][VirtualLaserKey.CHARGE_POL]["duration"]
        )
    else:
        pol_ns = int(pol_ns)

    if readout_ns is None:
        readout_ns = int(
            cfg["Optics"]["VirtualLasers"][VirtualLaserKey.SPIN_READOUT]["duration"]
        )
    else:
        readout_ns = int(readout_ns)

    readout_vkey = (
        _vkey_from_arg(readout_laser)
        if readout_laser is not None
        else VirtualLaserKey.SPIN_READOUT
    )

    base_args = [
        int(pol_ns),
        int(readout_ns),
        list(uwave_list) if isinstance(uwave_list, (list, tuple)) else [int(uwave_list)],
        str(readout_vkey.name),  # JSON-friendly
        readout_power,           # only used if ANALOG (see rabi_seq.py)
        int(max_tau),
    ]
    return base_args


def _rabi_model(tau_ns, offset, amp, freq_per_ns, decay_ns, phase_rad):
    """
    Simple decaying cosine model.
    freq_per_ns is cycles / ns (so period_ns = 1/freq_per_ns).
    """
    tau_ns = np.array(tau_ns, dtype=float)
    env = np.exp(-tau_ns / abs(decay_ns))
    return offset + env * amp * np.cos(2 * np.pi * freq_per_ns * tau_ns + phase_rad)


def fit_rabi(taus_ns, y, yerr=None):
    """
    FFT-seeded fit to _rabi_model.
    Returns dict with popt/pcov/red_chi_sq/derived period.
    """
    taus_ns = np.array(taus_ns, dtype=float)
    y = np.array(y, dtype=float)
    n = len(taus_ns)
    if n < 6:
        return {"popt": None, "pcov": None, "red_chi_sq": None, "rabi_period_ns": None}

    # Use finite errors if not provided
    if yerr is None:
        yerr = np.full_like(y, 1e-3, dtype=float)
    else:
        yerr = np.array(yerr, dtype=float)
        yerr[~np.isfinite(yerr)] = np.nan
        # avoid zeros
        yerr = np.where((yerr <= 0) | (~np.isfinite(yerr)), np.nanmedian(yerr[yerr > 0]), yerr)

    # FFT guess (remove DC)
    tau_step = float(np.median(np.diff(taus_ns)))
    y0 = y - np.nanmean(y)
    transform = np.fft.rfft(np.nan_to_num(y0))
    freqs = np.fft.rfftfreq(n, d=tau_step)  # cycles/ns

    mag = np.abs(transform)
    if len(mag) > 2:
        k = np.argmax(mag[1:]) + 1
        freq_guess = float(freqs[k])
        # phase of cosine: cos(2π f t + phase)
        phase_guess = float(np.angle(transform[k]))
    else:
        freq_guess = 1.0 / max(taus_ns.max(), 1.0)
        phase_guess = 0.0

    offset_guess = float(np.nanmean(y))
    amp_guess = float(0.5 * (np.nanmax(y) - np.nanmin(y)))
    if not np.isfinite(amp_guess) or amp_guess == 0:
        amp_guess = 0.05

    decay_guess = float(max(taus_ns.max(), 1.0))  # ns

    p0 = [offset_guess, amp_guess, max(freq_guess, 1e-6), decay_guess, phase_guess]

    # Bounds: keep things sane
    # offset: [0, 2], amp: [-1, 1], freq: [0, inf), decay: [1, inf), phase: [-2pi, 2pi]
    bounds = (
        [0.0, -1.0, 0.0, 1.0, -2 * np.pi],
        [2.0, 1.0, np.inf, np.inf, 2 * np.pi],
    )

    popt, pcov, red_chi_sq = tb.curve_fit(
        _rabi_model,
        taus_ns,
        y,
        p0=p0,
        sigma=yerr,
        bounds=bounds,
    )

    freq_fit = float(popt[2])
    rabi_period_ns = None if freq_fit <= 0 else float(1.0 / freq_fit)

    return {
        "popt": popt,
        "pcov": pcov,
        "red_chi_sq": red_chi_sq,
        "rabi_period_ns": rabi_period_ns,
    }


def create_raw_data_figure(taus_ns, norm, norm_ste, title="Confocal Rabi"):
    fig, ax = plt.subplots()
    kpl.plot_points(ax, taus_ns, norm, norm_ste, label="Data")
    ax.set_xlabel("MW pulse length τ (ns)")
    ax.set_ylabel("Normalized signal")
    ax.set_title(title)
    ax.legend()
    return fig


def create_fit_figure(taus_ns, norm, norm_ste, fit_result, title="Confocal Rabi (fit)"):
    fig, ax = plt.subplots()
    kpl.plot_points(ax, taus_ns, norm, norm_ste, label="Data")

    popt = fit_result.get("popt", None)
    if popt is not None:
        tau_lin = np.linspace(float(np.min(taus_ns)), float(np.max(taus_ns)), 1000)
        y_fit = _rabi_model(tau_lin, *popt)
        kpl.plot_line(ax, tau_lin, y_fit, label="Fit")

        period = fit_result.get("rabi_period_ns", None)
        rchi = fit_result.get("red_chi_sq", None)
        txt = []
        if period is not None:
            txt.append(f"Period ≈ {period:.2f} ns")
        if rchi is not None:
            txt.append(f"red χ² ≈ {rchi:.2f}")
        if txt:
            ax.text(
                0.02,
                0.98,
                "\n".join(txt),
                transform=ax.transAxes,
                va="top",
                ha="left",
            )

    ax.set_xlabel("MW pulse length τ (ns)")
    ax.set_ylabel("Normalized signal")
    ax.set_title(title)
    ax.legend()
    return fig


def main(
    nv_sig,
    num_steps,
    num_reps,
    num_runs,
    min_tau,
    max_tau,
    uwave_list,
    readout_laser=None,
    readout_power=None,
    do_plot=True,
    do_fit=True,
    norm_mode=NormMode.SINGLE_VALUED,
    save_figures=True,
):
    pulsegen = tb.get_server_pulse_streamer()

    taus = np.linspace(min_tau, max_tau, int(num_steps)).astype(int)
    base_args = get_base_seq_args_ps(
        nv_sig, uwave_list, max_tau, readout_laser, readout_power
    )

    def step_fn(step_ind: int):
        tau = int(taus[step_ind])
        seq_args = [base_args, tau, int(num_reps)]  # num_reps ignored by seq, ok
        pulsegen.stream_load("rabi_seq.py", tb.encode_seq_args(seq_args))

    raw_data = base.main(
        nv_sig=nv_sig,
        num_steps=int(num_steps),
        num_reps=int(num_reps),
        num_runs=int(num_runs),
        run_fn=None,
        step_fn=step_fn,
        uwave_ind_list=uwave_list,
        num_exps=2,       # signal + reference gates
        apd_indices=[0],  # update if needed
        load_iq=False,
        stream_load_in_run_fn=False,
        charge_prep_fn=None,
    )

    # ---- process counts ----
    counts = np.array(raw_data["counts"])  # (2, runs, steps, reps)
    readout_ns = int(base_args[1])

    # Sum over repetitions -> (runs, steps)
    sig_counts = counts[0].sum(axis=-1)
    ref_counts = counts[1].sum(axis=-1)

    sig_kcps, ref_kcps, norm, norm_ste = tb.process_counts(
        sig_counts, ref_counts, int(num_reps), readout_ns, norm_mode=norm_mode
    )

    raw_data |= {
        "taus_ns": taus,
        "tau-units": "ns",
        "min_tau": int(min_tau),
        "max_tau": int(max_tau),
        "uwave_list": list(uwave_list) if isinstance(uwave_list, (list, tuple)) else [int(uwave_list)],
        "readout_ns": int(readout_ns),
        "sig_counts_sum": sig_counts,
        "ref_counts_sum": ref_counts,
        "sig_kcps": sig_kcps,
        "ref_kcps": ref_kcps,
        "norm": norm,
        "norm_ste": norm_ste,
        "norm_mode": str(norm_mode),
    }

    # ---- fit (optional) ----
    fit_result = None
    if do_fit:
        try:
            fit_result = fit_rabi(taus, norm, norm_ste)
            raw_data |= {
                "fit_popt": None if fit_result["popt"] is None else np.array(fit_result["popt"]),
                "fit_red_chi_sq": fit_result["red_chi_sq"],
                "fit_rabi_period_ns": fit_result["rabi_period_ns"],
            }
        except Exception:
            print(traceback.format_exc())
            fit_result = None

    # ---- save raw data ----
    file_path = None
    timestamp = None
    timestamp = dm.get_time_stamp()
    nv_name = _get_nv_name(nv_sig)
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    raw_data |= {"timestamp": timestamp}
    dm.save_raw_data(raw_data, file_path)

    # ---- plot + save figures ----
    raw_fig = None
    fit_fig = None
    if do_plot:
        try:
            title = f"Confocal Rabi: { _get_nv_name(nv_sig) }"
            raw_fig = create_raw_data_figure(taus, norm, norm_ste, title=title)
            if do_fit and (fit_result is not None) and (fit_result.get("popt", None) is not None):
                fit_fig = create_fit_figure(taus, norm, norm_ste, fit_result, title=title + " (fit)")
        except Exception:
            print(traceback.format_exc())
            raw_fig = None
            fit_fig = None

    if save_figures and (file_path is not None):
        try:
            if raw_fig is not None:
                dm.save_figure(raw_fig, file_path)
            if fit_fig is not None:
                fit_path = dm.get_file_path(__file__, timestamp, _get_nv_name(nv_sig) + "-fit")
                dm.save_figure(fit_fig, fit_path)
        except Exception:
            print(traceback.format_exc())

    # ---- cleanup ----
    tb.reset_cfm()
    kpl.show()

    return raw_data


# ---------- Offline analysis helpers ----------
def analyze_saved(file_id):
    """
    Convenience: load a saved dataset and re-make plots.
    """
    data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    taus = data["taus_ns"]
    norm = data["norm"]
    norm_ste = data.get("norm_ste", None)

    raw_fig = create_raw_data_figure(taus, norm, norm_ste, title="Confocal Rabi (replot)")

    fit_fig = None
    if "fit_popt" in data and data["fit_popt"] is not None:
        fit_result = {
            "popt": np.array(data["fit_popt"]),
            "red_chi_sq": data.get("fit_red_chi_sq", None),
            "rabi_period_ns": data.get("fit_rabi_period_ns", None),
        }
        fit_fig = create_fit_figure(taus, norm, norm_ste, fit_result, title="Confocal Rabi (fit replot)")

    kpl.show(block=True)
    return data, raw_fig, fit_fig


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Example offline analysis:
    # data, raw_fig, fit_fig = analyze_saved(file_id=1234567890123)
    # sys.exit()

    print("This module is meant to be run from your experiment runner (calling main(...)).")
