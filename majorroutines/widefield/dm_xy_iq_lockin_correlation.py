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


def compute_iq_from_counts(counts, eps=1e-12, return_masks=False):
    counts = np.asarray(counts)

    # If counts is object (None inside), force to float -> None becomes nan
    if counts.dtype == object:
        counts = counts.astype(float)

    assert counts.ndim == 5, f"Expected 5D counts, got shape {counts.shape}"

    Ip = np.asarray(counts[0], dtype=float)
    Im = np.asarray(counts[1], dtype=float)
    Qp = np.asarray(counts[2], dtype=float)
    Qm = np.asarray(counts[3], dtype=float)

    num_nvs = Ip.shape[0]
    Ip_f = np.stack([_flatten_shots(Ip[nv]) for nv in range(num_nvs)], axis=0)
    Im_f = np.stack([_flatten_shots(Im[nv]) for nv in range(num_nvs)], axis=0)
    Qp_f = np.stack([_flatten_shots(Qp[nv]) for nv in range(num_nvs)], axis=0)
    Qm_f = np.stack([_flatten_shots(Qm[nv]) for nv in range(num_nvs)], axis=0)

    denI = Ip_f + Im_f
    denQ = Qp_f + Qm_f

    I = np.full_like(denI, np.nan, dtype=float)
    Q = np.full_like(denQ, np.nan, dtype=float)

    mI = np.isfinite(Ip_f) & np.isfinite(Im_f) & np.isfinite(denI) & (denI > 0)
    mQ = np.isfinite(Qp_f) & np.isfinite(Qm_f) & np.isfinite(denQ) & (denQ > 0)

    I[mI] = (Ip_f[mI] - Im_f[mI]) / (denI[mI] + eps)
    Q[mQ] = (Qp_f[mQ] - Qm_f[mQ]) / (denQ[mQ] + eps)

    s = I + 1j * Q

    if return_masks:
        return s, I, Q, mI, mQ
    return s, I, Q


def nan_pairwise_covariance(s, remove_mean=True, min_pairs=1000, eps=1e-18):
    """
    s: complex array [M, N] with possible NaNs.
    Returns:
      C: complex covariance-like matrix [M, M] using only pairwise-valid shots
      cnt: number of valid (i,j) pairs used per entry
      keep: boolean mask of NVs kept after basic validity checks
    """
    s = np.asarray(s, dtype=np.complex64)
    M, N = s.shape

    finite = np.isfinite(s)
    valid_frac = finite.mean(axis=1)

    # Keep NVs that have at least some decent fraction of valid samples
    # (start lenient; tighten later)
    keep = valid_frac > 0.5
    s = s[keep]
    finite = finite[keep]
    M2 = s.shape[0]

    if M2 < 2:
        raise RuntimeError(f"After keep mask, only {M2} NVs remain (valid_frac>0.5).")

    # Mean per NV over valid samples only
    if remove_mean:
        denom = np.maximum(finite.sum(axis=1), 1)
        mu = (np.where(finite, s, 0).sum(axis=1) / denom).astype(np.complex64)
        s0 = s - mu[:, None]
    else:
        s0 = s.copy()

    # Zero-out invalid entries (they won't contribute)
    s0z = np.where(finite, s0, 0).astype(np.complex64)

    # Numerator and pair-counts
    num = s0z @ s0z.conj().T                           # [M2,M2]
    cnt = finite.astype(np.int32) @ finite.astype(np.int32).T  # [M2,M2]

    # Divide only where we have enough pairs
    C = np.full_like(num, np.nan, dtype=np.complex64)
    ok = cnt >= min_pairs
    C[ok] = num[ok] / (cnt[ok].astype(np.float32) + eps)

    # Symmetrize
    C = 0.5 * (C + C.conj().T)

    return C, cnt, keep

def coherence_from_cov(C, eps=1e-18, min_var=1e-12):
    C = np.asarray(C)
    p = np.real(np.diag(C))

    keep = np.isfinite(p) & (p > min_var)
    C2 = C[keep][:, keep]

    if C2.shape[0] < 2:
        raise RuntimeError(f"After var-cut, only {C2.shape[0]} NVs remain.")

    p2 = np.real(np.diag(C2))
    denom = np.sqrt(np.maximum(p2[:, None] * p2[None, :], eps))
    coh = C2 / denom
    return coh, C2, keep


def dominant_mode_stats(C, eps=1e-18):
    C = np.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square, got shape {C.shape}")
    if C.shape[0] < 2:
        raise RuntimeError(f"Not enough NVs left to eigendecompose: C is {C.shape}")

    Ch = 0.5 * (C + np.conjugate(C.T))
    evals, evecs = np.linalg.eigh(Ch)
    order = np.argsort(np.real(evals))
    evals = np.real(evals[order])
    evecs = evecs[:, order]

    tr = float(np.sum(evals))
    frac = float(evals[-1] / (tr + eps))
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

    M0, N = s.shape
    print(f"[DM lock-in] raw: M={M0}, N={N}")
    print(f"[DM lock-in] I NaN frac median={np.nanmedian(~np.isfinite(I)):.3g}")
    print(f"[DM lock-in] Q NaN frac median={np.nanmedian(~np.isfinite(Q)):.3g}")

    C, cnt, keep1 = nan_pairwise_covariance(s, remove_mean=True, min_pairs=1000)
    coh, C2, keep2 = coherence_from_cov(C)

    print(f"[DM lock-in] kept NVs after validity: {keep1.sum()} / {M0}")
    print(f"[DM lock-in] kept NVs after var-cut: {keep2.sum()} / {keep1.sum()}")
    print(f"[DM lock-in] median pair-count: {np.nanmedian(cnt[cnt>0])}")

    evals, evecs, frac = dominant_mode_stats(C2)
    v = evecs[:, -1]


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
    n_pi=1,
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
    seq_file = "dm_xy_iq_lockin_correlation.py"  # must match the QUA seq file name
    num_steps = 1

    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(_shuffled_step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, list(uwave_ind_list)),
            float(tau_ns),
            n_pi,
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
        "n_pi_pulses": n_pi,
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
    raw_data = dm.get_raw_data(file_stem="2025_12_24-09_32_29-johnson-nv0_2025_10_21", load_npz=True)
    figs, proc = process_and_plot_dm_lockin(raw_data, show=False)
    kpl.show(block=True)
