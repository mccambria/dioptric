# -*- coding: utf-8 -*-
"""
DM narrowband lock-in search (room temperature) using widefield NV array.

- Uses XY4-N dynamical decoupling at a chosen tau (e.g. tau=7.5us for 15us revival).
- Uses 4-shot phase cycling to build quadratures:
    I+, I-, Q+, Q-
  and constructs complex signal s = I + i Q per NV per shot.
- Then computes spatial coherence matrix and dominant common-mode eigenvector.

@author: Saroj Chand (adapted for DM lock-in)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


# -----------------------------
# Processing / plotting helpers
# -----------------------------

def _flatten_shots(x):
    """
    Flatten [run, step, rep] -> [shots] (keeping order).
    Expect x shape: (num_runs, num_steps, num_reps) or similar.
    """
    return np.reshape(x, (-1,))


def compute_iq_from_counts(counts, eps=1e-12):
    """
    counts: np.ndarray with shape [exp, nv, run, step, rep]
        exp ordering: [I+, I-, Q+, Q-, (optional ref)]
    Returns:
        s: complex array [nv, shots]
        I, Q: float arrays [nv, shots]
    """
    counts = np.asarray(counts)
    assert counts.ndim == 5, f"Expected 5D counts, got shape {counts.shape}"

    # Exps
    Ip = counts[0]  # [nv, run, step, rep]
    Im = counts[1]
    Qp = counts[2]
    Qm = counts[3]

    num_nvs = Ip.shape[0]
    # Flatten run/step/rep -> shots
    Ip_f = np.stack([_flatten_shots(Ip[nv]) for nv in range(num_nvs)], axis=0)
    Im_f = np.stack([_flatten_shots(Im[nv]) for nv in range(num_nvs)], axis=0)
    Qp_f = np.stack([_flatten_shots(Qp[nv]) for nv in range(num_nvs)], axis=0)
    Qm_f = np.stack([_flatten_shots(Qm[nv]) for nv in range(num_nvs)], axis=0)

    I = (Ip_f - Im_f) / (Ip_f + Im_f + eps)
    Q = (Qp_f - Qm_f) / (Qp_f + Qm_f + eps)
    s = I + 1j * Q
    return s, I, Q


def complex_coherence_matrix(s, remove_mean=True, eps=1e-18):
    """
    s: complex array [nv, shots]
    Returns:
        coh: complex coherence matrix [nv, nv] with diag ~ 1
        C: complex covariance matrix [nv, nv]
    """
    s = np.asarray(s)
    if remove_mean:
        s0 = s - np.mean(s, axis=1, keepdims=True)
    else:
        s0 = s

    # Covariance-like (not unbiased; we want stable estimator)
    C = (s0 @ np.conjugate(s0.T)) / s0.shape[1]

    # Normalize to coherence
    p = np.real(np.diag(C))
    p = np.maximum(p, eps)
    denom = np.sqrt(p[:, None] * p[None, :])
    coh = C / denom
    return coh, C


def dominant_mode_stats(C):
    """
    C: Hermitian-ish covariance matrix [nv, nv]
    Returns:
        evals_sorted, evecs_sorted, frac_power_first
    """
    # Force Hermitian (numerical)
    Ch = 0.5 * (C + np.conjugate(C.T))
    evals, evecs = np.linalg.eigh(Ch)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    frac = float(evals[-1] / (np.sum(evals) + 1e-18))
    return evals, evecs, frac


def process_and_plot_dm_lockin(raw_data, show=True):
    """
    raw_data: from base_routine.main(...)
    Produces:
      - coherence heatmap (Re and |.|)
      - dominant eigenmode amplitude vs NV index
      - eigenvalue spectrum (optional)
    """
    nv_list = raw_data["nv_list"]
    counts = np.array(raw_data["counts"])
    num_nvs = len(nv_list)

    # Build complex lock-in output per NV per shot
    s, I, Q = compute_iq_from_counts(counts)

    # Coherence across NVs
    coh, C = complex_coherence_matrix(s, remove_mean=True)
    evals, evecs, frac = dominant_mode_stats(C)
    v = evecs[:, -1]  # dominant spatial mode

    print(f"[DM lock-in] NVs: {num_nvs}")
    print(f"[DM lock-in] shots per NV: {s.shape[1]}")
    print(f"[DM lock-in] dominant mode power fraction: {frac:.3f}")

    # Plot coherence matrices
    figs = []

    # 1) Real(coherence)
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    mat1 = np.real(coh)
    np.fill_diagonal(mat1, np.nan)
    kpl.imshow(
        ax1,
        mat1,
        title="Re[Coherence] of lock-in signal s = I + iQ",
        cbar_label="Re(coh)",
        cmap="RdBu_r",
        # vmin=-1,
        # vmax=1,
        nan_color=kpl.KplColors.GRAY,
    )
    ax1.set_xlabel("NV index")
    ax1.set_ylabel("NV index")
    figs.append(fig1)

    # 2) |coherence|
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    mat2 = np.abs(coh)
    np.fill_diagonal(mat2, np.nan)
    kpl.imshow(
        ax2,
        mat2,
        title="|Coherence| of lock-in signal",
        cbar_label="|coh|",
        cmap="viridis",
        # vmin=0,
        # vmax=1,
        nan_color=kpl.KplColors.GRAY,
    )
    ax2.set_xlabel("NV index")
    ax2.set_ylabel("NV index")
    figs.append(fig2)

    # 3) Dominant mode amplitude/phase vs NV index
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    ax3.plot(np.arange(num_nvs), np.abs(v), marker=".", linewidth=1)
    ax3.set_title(f"Dominant common-mode spatial eigenvector |v|  (power frac={frac:.3f})")
    ax3.set_xlabel("NV index")
    ax3.set_ylabel("|v|")
    figs.append(fig3)

    # 4) Eigenvalue spectrum (optional quick check)
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    ax4.plot(np.arange(len(evals)), evals, marker=".", linewidth=1)
    ax4.set_title("Eigenvalue spectrum of covariance(C)")
    ax4.set_xlabel("mode index")
    ax4.set_ylabel("eigenvalue")
    figs.append(fig4)

    if show:
        kpl.show(block=True)

    return figs, {"s": s, "I": I, "Q": Q, "coh": coh, "C": C, "evals": evals, "evecs": evecs, "frac": frac}


# -----------------------------
# Main experiment
# -----------------------------

def main(
    nv_list,
    num_reps,
    num_runs,
    tau_ns,
    n_xy4_blocks=1,
    include_ref=False,
    uwave_ind_list=(0, 1),
):
    """
    DM lock-in experiment runner.

    tau_ns:
      This is the "Hahn tau" (the half-period in your earlier notation).
      If you set 2*tau = 15us (revival), use tau_ns = 7.5e3.
      The lock-in center frequency is approximately f0 ~ 1/(2*tau).

    n_xy4_blocks:
      XY4-N. Total pi pulses = 4 * n_xy4_blocks.
      Total evolution time ~ 8*tau*n_xy4_blocks (with this implementation).
    """
    seq_file = "dm_xy4_iq_lockin_correlation.py"  # must match the QUA seq file name
    num_steps = 1

    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(_shuffled_step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, list(uwave_ind_list)),
            float(tau_ns)
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        num_exps= 4,
        run_fn=run_fn,
        uwave_ind_list=list(uwave_ind_list),
        load_iq=True,
    )

    # Process + plot
    figs = None
    try:
        figs, proc = process_and_plot_dm_lockin(raw_data, show=False)
    except Exception as exc:
        print(f"[WARN] processing failed: {exc}")
        proc = None

    tb.reset_cfm()
    kpl.show()

    # Save
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "tau_ns": tau_ns,
        "n_xy4_blocks": n_xy4_blocks,
        "include_ref": include_ref,
        "seq_file": seq_file,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    if figs is not None:
        for ind, fig in enumerate(figs):
            fig_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-fig{ind}")
            dm.save_figure(fig, fig_path)

    # Optionally save processed arrays (small enough)
    # if proc is not None:
    #     npz_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-proc")
    #     dm.save_npz(
    #         {
    #             "s": proc["s"],
    #             "I": proc["I"],
    #             "Q": proc["Q"],
    #             "coh": proc["coh"],
    #             "frac": np.array([proc["frac"]]),
    #         },
    #         npz_path,
    #     )

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()
    raw_data = dm.get_raw_data(file_stem="2025_12_23-16_10_36-johnson-nv0_2025_10_21", load_npz=True)
    figs, proc = process_and_plot_dm_lockin(raw_data, show=False)
    kpl.show(block=True)
