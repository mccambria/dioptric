import numpy as np
import matplotlib.pyplot as plt
from utils import kplotlib as kpl
from utils import data_manager as dm


def y_hahn(t, tau):
    # Hahn: +1 for [0,tau), -1 for [tau,2tau)
    return np.where(t < tau, 1.0, -1.0)


def y_xy4_standard(t, tau):
    # XY4-1 with endcaps tau and 2tau between pi pulses:
    # pi pulses at t = tau, 3tau, 5tau, 7tau; total T = 8tau.
    flip_times = np.array([tau, 3 * tau, 5 * tau, 7 * tau])
    flips = np.sum(t[:, None] >= flip_times[None, :], axis=1)
    return np.where(flips % 2 == 0, 1.0, -1.0)


def Y_from_y(t, y, w):
    dt = t[1] - t[0]
    phase = np.exp(1j * w[:, None] * t[None, :])
    return (phase @ y) * dt


def filter_plot(tau_hahn=15e-6, tau_xy4=3.75e-6, fmax=200e3):
    f = np.linspace(1e3, fmax, 5000)
    w = 2 * np.pi * f

    # Hahn
    T_h = 2 * tau_hahn
    t_h = np.linspace(0, T_h, 60000, endpoint=False)
    y_h = y_hahn(t_h, tau_hahn)
    F_h = np.abs(Y_from_y(t_h, y_h, w)) ** 2
    F_h /= F_h.max()

    # XY4-1
    T_x = 8 * tau_xy4
    t_x = np.linspace(0, T_x, 60000, endpoint=False)
    y_x = y_xy4_standard(t_x, tau_xy4)
    F_x = np.abs(Y_from_y(t_x, y_x, w)) ** 2
    F_x /= F_x.max()

    plt.figure()
    plt.plot(
        f / 1e3,
        F_h,
        label=f"Hahn tau={tau_hahn*1e6:.2f} us (T={2*tau_hahn*1e6:.1f} us)",
    )
    plt.plot(
        f / 1e3, F_x, label=f"XY4-1 tau={tau_xy4*1e6:.2f} us (T={8*tau_xy4*1e6:.1f} us)"
    )
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized |Y(ω)|²")
    plt.title("Filter functions (numerical)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("filter_functions.pdf")


def tau_mapping_plot():
    tau = np.linspace(1e-6, 30e-6, 400)
    f0_h = 1 / (2 * tau)  # Hahn approx
    f0_x = 1 / (4 * tau)  # XY4 approx (standard timing)

    plt.figure()
    plt.plot(tau * 1e6, f0_h / 1e3, label="Hahn: f0 ≈ 1/(2τ)")
    plt.plot(tau * 1e6, f0_x / 1e3, label="XY4: f0 ≈ 1/(4τ)")
    plt.axvline(15, linestyle="--", label="Hahn revival τ = 15 µs")
    plt.xlabel("τ (µs)")
    plt.ylabel("f0 (kHz)")
    plt.title("Approx. lock-in center frequency vs τ")
    plt.ylim(0, 200)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tau_mapping.pdf")


import numpy as np
import matplotlib.pyplot as plt


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


def whiten_lockin(s, remove_mean=True, eps=1e-18):
    """Return whitened complex lock-in series sw and correlation-like matrix R."""
    s = np.asarray(s)
    if remove_mean:
        s0 = s - np.mean(s, axis=1, keepdims=True)
    else:
        s0 = s
    sigma = np.sqrt(np.mean(np.abs(s0) ** 2, axis=1, keepdims=True)) + eps
    sw = s0 / sigma
    R = (sw @ np.conjugate(sw.T)) / sw.shape[1]
    # enforce Hermitian numerically
    R = 0.5 * (R + np.conjugate(R.T))
    return sw, R


def whiten_per_nv_complex_power(s_mn, eps=1e-12):
    """
    s_mn: (M, N) complex
    Returns z where each NV has E[|z|^2] ~ 1.
    """
    mu = np.mean(s_mn, axis=1, keepdims=True)
    x = s_mn - mu
    power = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)  # complex power
    scale = np.sqrt(np.maximum(power, eps))
    return x / scale


def mp_bounds(M, N):
    q = M / N
    lam_minus = (1 - np.sqrt(q)) ** 2
    lam_plus = (1 + np.sqrt(q)) ** 2
    return lam_minus, lam_plus, q


def top_mode_fraction_from_matrix(R, eps=1e-18):
    evals = np.linalg.eigvalsh(R)
    evals = np.sort(np.real(evals))
    frac = float(evals[-1] / (np.sum(evals) + eps))
    return evals, frac


def null_frac_distribution(sw, n_null=300, method="roll", seed=0):
    """
    sw: whitened series [M, N] (complex)
    method:
      - "roll": circularly shift each row by random offset (preserves per-NV spectrum)
      - "perm": permute shots within each row (stronger destruction)
    Returns array of frac_null, and eigenvalues of one representative null trial.
    """
    rng = np.random.default_rng(seed)
    M, N = sw.shape
    fracs = np.empty(n_null, dtype=float)

    for t in range(n_null):
        if method == "roll":
            shifts = rng.integers(0, N, size=M)
            sw_null = np.empty_like(sw)
            for i in range(M):
                sw_null[i] = np.roll(sw[i], int(shifts[i]))
        elif method == "perm":
            sw_null = np.empty_like(sw)
            for i in range(M):
                sw_null[i] = sw[i, rng.permutation(N)]
        else:
            raise ValueError("method must be 'roll' or 'perm'")

        Rn = (sw_null @ sw_null.conj().T) / N
        Rn = 0.5 * (Rn + Rn.conj().T)
        evals_n, frac_n = top_mode_fraction_from_matrix(Rn)
        fracs[t] = frac_n

    return fracs


def null_top_frac(sw, n_null=500, method="roll", seed=0):
    rng = np.random.default_rng(seed)
    M, N = sw.shape
    fracs = np.empty(n_null)

    for t in range(n_null):
        if method == "roll":
            shifts = rng.integers(0, N, size=M)
            sw_null = np.empty_like(sw)
            for i in range(M):
                sw_null[i] = np.roll(sw[i], int(shifts[i]))
        else:  # "perm"
            sw_null = np.empty_like(sw)
            for i in range(M):
                sw_null[i] = sw[i, rng.permutation(N)]

        Rn = (sw_null @ sw_null.conj().T) / N
        Rn = 0.5 * (Rn + Rn.conj().T)
        evals = np.linalg.eigvalsh(Rn)
        fracs[t] = float(np.max(evals) / np.sum(evals))

    return fracs


def top_mode_fraction_from_matrix(R, eps=1e-18):
    """
    R: (M,M) Hermitian-ish matrix
    Returns:
      evals (ascending), evecs (columns aligned with evals), frac = max(evals)/sum(evals)
    """
    Rh = 0.5 * (R + R.conj().T)  # enforce Hermitian numerically
    evals, evecs = np.linalg.eigh(Rh)  # evals ascending
    evals = np.real(evals)
    frac = float(evals[-1] / (np.sum(evals) + eps))
    return evals, evecs, frac


def null_frac_distribution(sw, n_null=300, method="roll", seed=0, eps=1e-18):
    """
    sw: whitened series [M, N] (complex), ideally mean-removed and var-normalized per NV.

    method:
      - "roll": circularly shift each row by random offset (preserves per-NV spectrum)
      - "perm": permute shots within each row (stronger destruction)

    Returns:
      fracs: array of top-mode fractions under the null
    """
    rng = np.random.default_rng(seed)
    M, N = sw.shape
    fracs = np.empty(n_null, dtype=float)

    for t in range(n_null):
        sw_null = np.empty_like(sw)

        if method == "roll":
            shifts = rng.integers(0, N, size=M)
            for i in range(M):
                sw_null[i] = np.roll(sw[i], int(shifts[i]))
        elif method == "perm":
            for i in range(M):
                sw_null[i] = sw[i, rng.permutation(N)]
        else:
            raise ValueError("method must be 'roll' or 'perm'")

        Rn = (sw_null @ sw_null.conj().T) / N
        Rn = 0.5 * (Rn + Rn.conj().T)

        evals_n = np.linalg.eigvalsh(Rn)
        evals_n = np.real(evals_n)
        fracs[t] = float(np.max(evals_n) / (np.sum(evals_n) + eps))

    return fracs


def process_and_plot_dm_lockin(raw_data, show=True, n_null=300):
    nv_list = raw_data["nv_list"]
    counts = np.array(raw_data["counts"])
    M = len(nv_list)

    s, I, Q = compute_iq_from_counts(counts)
    M, N = s.shape

    # whiten + correlation-like matrix (best for common-mode searches)
    sw, R = whiten_lockin(s, remove_mean=True)

    evals_R, evecs_R, frac_R = top_mode_fraction_from_matrix(R)

    lam_minus, lam_plus, q = mp_bounds(M, N)
    frac_noise_mp = lam_plus / M

    print(f"[DM lock-in] NVs: {M}")
    print(f"[DM lock-in] shots per NV: {N}")
    print(f"[DM lock-in] top-mode frac (data, whitened): {frac_R:.4f}")
    print(
        f"[DM lock-in] MP noise frac ~ {frac_noise_mp:.4f}   (q={q:.3f}, lam+={lam_plus:.3f})"
    )

    # ---- Null distribution of frac ----
    fracs_null = null_frac_distribution(sw, n_null=n_null, method="roll", seed=1)
    pval = (np.sum(fracs_null >= frac_R) + 1) / (len(fracs_null) + 1)

    figs = []

    # A) Eigenvalue spectrum with MP bounds
    figA, axA = plt.subplots(figsize=(7, 3.8))
    axA.plot(
        np.arange(M),
        evals_R,
        marker=".",
        linewidth=1,
        label="Data eigvals (whitened R)",
    )
    axA.axhline(lam_minus, linestyle="--", linewidth=1, label=r"MP $\lambda_-$")
    axA.axhline(lam_plus, linestyle="--", linewidth=1, label=r"MP $\lambda_+$")
    axA.set_title("Eigenvalue spectrum (whitened correlation matrix)")
    axA.set_xlabel("mode index")
    axA.set_ylabel("eigenvalue")
    axA.legend()
    figs.append(figA)

    # B) Null histogram for top-mode fraction
    figB, axB = plt.subplots(figsize=(6.5, 3.5))
    axB.hist(fracs_null, bins=35, alpha=0.7, density=True, label="Null (roll each NV)")
    axB.axvline(frac_R, linewidth=2, label=f"Data frac={frac_R:.4f}")
    axB.axvline(
        frac_noise_mp,
        linestyle="--",
        linewidth=1.5,
        label=f"MP est={frac_noise_mp:.4f}",
    )
    axB.set_title(f"Top-mode fraction null test (p ≈ {pval:.3f})")
    axB.set_xlabel(r"$\lambda_1 / \mathrm{Tr}$")
    axB.set_ylabel("density")
    axB.legend()
    figs.append(figB)

    # C) Heatmaps of whitened R
    figC, axC = plt.subplots(figsize=(7, 6))
    matC = np.real(0.5 * (R + R.conj().T))
    np.fill_diagonal(matC, np.nan)
    kpl.imshow(
        axC,
        matC,
        title="Re[R] (whitened)",
        cbar_label="Re",
        cmap="RdBu_r",
        nan_color=kpl.KplColors.GRAY,
    )
    figs.append(figC)

    figD, axD = plt.subplots(figsize=(7, 6))
    matD = np.abs(R)
    np.fill_diagonal(matD, np.nan)
    kpl.imshow(
        axD,
        matD,
        title="|R| (whitened)",
        cbar_label="|.|",
        cmap="viridis",
        nan_color=kpl.KplColors.GRAY,
    )
    figs.append(figD)

    # D) Dominant eigenvector |v| (reuse evecs_R)
    v = evecs_R[:, -1]
    figE, axE = plt.subplots(figsize=(8, 3.5))
    axE.plot(np.arange(M), np.abs(v), marker=".", linewidth=1)
    axE.set_title(f"Dominant eigenvector |v| (whitened)   frac={frac_R:.4f}")
    axE.set_xlabel("NV index")
    axE.set_ylabel("|v|")
    figs.append(figE)

    if show:
        kpl.show(block=True)

    return figs, {
        "s": s,
        "sw": sw,
        "R": R,
        "evals_R": evals_R,
        "frac_R": frac_R,
        "fracs_null": fracs_null,
        "pval": pval,
        "lam_minus": lam_minus,
        "lam_plus": lam_plus,
        "frac_noise_mp": frac_noise_mp,
    }


import numpy as np


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

# dm_lockin_goodshot_analysis.py
#
# End-to-end “good-shot” analysis for DM lock-in widefield data:
#   1) load raw_data
#   2) find global-dropout runs/shots
#   3) drop bad runs (and optionally any remaining global-bad shots)
#   4) run coherence/covariance + plots + per-NV table
#
# Assumes raw_data["counts"] has shape (4, M, R, S, P)
#   4 phase-cycles: (Ip, Im, Qp, Qm)

import numpy as np
import matplotlib.pyplot as plt

from utils import data_manager as dm
from utils import kplotlib as kpl


# -----------------------------
# Core helpers
# -----------------------------


def compute_iq_from_counts(counts, eps=1e-12, return_masks=False):
    """
    counts: array-like shape (4, M, R, S, P)
    Returns:
      s: complex [M, Nshots]
      I: float   [M, Nshots]
      Q: float   [M, Nshots]
      (optional) mI, mQ masks [M, Nshots] showing where I/Q are valid
    """
    counts = np.asarray(counts)

    # If object (None inside), force to float -> None becomes nan
    if counts.dtype == object:
        counts = counts.astype(float)

    assert counts.ndim == 5, f"Expected 5D counts, got shape {counts.shape}"
    assert (
        counts.shape[0] == 4
    ), f"Expected first dim=4 (Ip,Im,Qp,Qm), got {counts.shape[0]}"

    Ip = np.asarray(counts[0], dtype=float)  # (M,R,S,P)
    Im = np.asarray(counts[1], dtype=float)
    Qp = np.asarray(counts[2], dtype=float)
    Qm = np.asarray(counts[3], dtype=float)

    M = Ip.shape[0]
    Ip_f = Ip.reshape(M, -1)
    Im_f = Im.reshape(M, -1)
    Qp_f = Qp.reshape(M, -1)
    Qm_f = Qm.reshape(M, -1)

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


def nan_pairwise_covariance(
    s, remove_mean=True, min_pairs=1000, eps=1e-18, min_valid_frac=0.5
):
    """
    s: complex array [M, N] with possible NaNs.
    Returns:
      C: complex covariance-like matrix [M2, M2] using only pairwise-valid shots
      cnt: number of valid (i,j) pairs used per entry [M2,M2]
      keep: boolean mask [M] of NVs kept after validity check
    """
    s = np.asarray(s, dtype=np.complex64)
    M, N = s.shape

    finite = np.isfinite(s)
    valid_frac = finite.mean(axis=1)

    keep = valid_frac > float(min_valid_frac)
    s = s[keep]
    finite = finite[keep]
    M2 = s.shape[0]

    if M2 < 2:
        raise RuntimeError(
            f"After keep mask (valid_frac>{min_valid_frac}), only {M2} NVs remain."
        )

    if remove_mean:
        denom = np.maximum(finite.sum(axis=1), 1)
        mu = (np.where(finite, s, 0).sum(axis=1) / denom).astype(np.complex64)
        s0 = s - mu[:, None]
    else:
        s0 = s.copy()

    s0z = np.where(finite, s0, 0).astype(np.complex64)

    num = s0z @ s0z.conj().T
    cnt = finite.astype(np.int32) @ finite.astype(np.int32).T

    C = np.full_like(num, np.nan, dtype=np.complex64)
    ok = cnt >= int(min_pairs)
    C[ok] = num[ok] / (cnt[ok].astype(np.float32) + eps)

    C = 0.5 * (C + C.conj().T)
    return C, cnt, keep


def coherence_from_cov(C, eps=1e-18, min_var=1e-12):
    """
    Normalized coherence matrix from covariance:
      coh_ij = C_ij / sqrt(C_ii * C_jj)
    Also applies a variance cut.
    """
    C = np.asarray(C)
    p = np.real(np.diag(C))

    keep = np.isfinite(p) & (p > float(min_var))
    C2 = C[keep][:, keep]

    if C2.shape[0] < 2:
        raise RuntimeError(
            f"After var-cut (min_var={min_var}), only {C2.shape[0]} NVs remain."
        )

    p2 = np.real(np.diag(C2))
    denom = np.sqrt(np.maximum(p2[:, None] * p2[None, :], eps))
    coh = C2 / denom
    return coh, C2, keep


def dominant_mode_stats(C, eps=1e-18):
    """
    Eigen-decompose Hermitian C. Returns evals/evecs sorted ascending and
    fraction of total power in the top mode.
    """
    C = np.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square, got shape {C.shape}")
    if C.shape[0] < 2:
        raise RuntimeError(f"Not enough NVs left: C is {C.shape}")

    Ch = 0.5 * (C + np.conjugate(C.T))
    evals, evecs = np.linalg.eigh(Ch)
    order = np.argsort(np.real(evals))
    evals = np.real(evals[order])
    evecs = evecs[:, order]

    tr = float(np.sum(evals))
    frac = float(evals[-1] / (tr + eps))
    return evals, evecs, frac


# -----------------------------
# Good-shot / dropout detection + filtering
# -----------------------------


def summarize_global_bad_runs(raw_data, global_bad_thresh=0.9):
    """
    Returns info about "global bad" shots and which runs are dominated by them.
    A shot is "global bad" if >global_bad_thresh of NVs are invalid in that shot.
    A run is "bad" if >global_bad_thresh of its reps are global-bad.
    """
    counts = np.asarray(raw_data["counts"])
    _, M, R, S, P = counts.shape
    N = R * S * P

    s, I, Q, mI, mQ = compute_iq_from_counts(counts, return_masks=True)
    bad = (~mI) | (~mQ)  # [M, Nshots]
    frac_bad_per_shot = bad.mean(axis=0)

    global_bad_shots = np.where(frac_bad_per_shot > float(global_bad_thresh))[0]

    out = {
        "counts_shape": counts.shape,
        "M": M,
        "R": R,
        "S": S,
        "P": P,
        "Nshots": N,
        "frac_bad_per_shot": frac_bad_per_shot,
        "global_bad_shots": global_bad_shots,
    }

    if len(global_bad_shots) == 0:
        return out

    # Map first/last global bad shot -> (run, step, rep)
    r0, s0, p0 = np.unravel_index(global_bad_shots[0], (R, S, P))
    r1, s1, p1 = np.unravel_index(global_bad_shots[-1], (R, S, P))
    out["first_global_bad"] = (int(global_bad_shots[0]), int(r0), int(s0), int(p0))
    out["last_global_bad"] = (int(global_bad_shots[-1]), int(r1), int(s1), int(p1))

    # Run-wise fraction of global-bad reps
    global_bad_mask_3d = (frac_bad_per_shot > float(global_bad_thresh)).reshape(R, S, P)
    global_bad_frac_per_run = global_bad_mask_3d.mean(axis=(1, 2))  # [R]
    bad_runs = np.where(global_bad_frac_per_run > float(global_bad_thresh))[0]

    out["global_bad_frac_per_run"] = global_bad_frac_per_run
    out["bad_runs"] = bad_runs
    if len(bad_runs) > 0:
        out["bad_run_range"] = (int(bad_runs[0]), int(bad_runs[-1]))
        out["trailing_block"] = bool(np.all(bad_runs == np.arange(bad_runs[0], R)))

    return out


def make_goodshot_raw_data(
    raw_data,
    global_bad_thresh=0.9,
    drop_bad_runs=True,
    drop_remaining_global_bad_shots=True,
):
    """
    Produces a filtered raw_data_good that contains only good runs/shots.

    - First, identifies bad runs (runs dominated by global-bad shots) and removes them.
    - Then (optional), removes any remaining global-bad shots (not necessarily whole runs)
      by physically rebuilding counts with those shots removed (only supported when S=1).
      If your S>1, it will skip shot-level rebuilding and rely on NaN-handling.

    Returns: (raw_good, info)
    """
    counts = np.asarray(raw_data["counts"])
    raw_good = dict(raw_data)  # shallow copy
    info0 = summarize_global_bad_runs(raw_data, global_bad_thresh=global_bad_thresh)

    M, R, S, P = info0["M"], info0["R"], info0["S"], info0["P"]

    bad_runs = info0.get("bad_runs", np.array([], dtype=int))
    keep_runs = np.arange(R)

    if drop_bad_runs and len(bad_runs) > 0:
        keep_runs = np.setdiff1d(keep_runs, bad_runs)
        counts = counts[:, :, keep_runs, :, :]
        raw_good["counts"] = counts
    else:
        raw_good["counts"] = counts

    # Recompute on run-trimmed data to see if any global-bad shots remain
    info1 = summarize_global_bad_runs(raw_good, global_bad_thresh=global_bad_thresh)

    # Optionally drop remaining global-bad shots by rebuilding counts
    # This is easiest when S=1 because shots map cleanly to (run, rep).
    if drop_remaining_global_bad_shots and len(info1["global_bad_shots"]) > 0:
        _, M2, R2, S2, P2 = raw_good["counts"].shape
        if S2 != 1:
            # Fall back: don't rebuild counts; NaN-handling will ignore invalid shots
            info1["note"] = (
                "S != 1, skipping shot-level rebuild; relying on NaN-handling."
            )
            return raw_good, {"before": info0, "after_runs": info1}

        # Build a boolean keep mask in shot space, then reshape to (R, P)
        frac_bad = info1["frac_bad_per_shot"]
        keep_shots = frac_bad <= float(global_bad_thresh)  # keep non-global-bad shots
        keep_shots_2d = keep_shots.reshape(R2, P2)  # since S2=1

        # Keep only reps that are not global-bad; this makes P vary per run,
        # so we will compress by selecting run/rep pairs and creating a new "rep axis".
        # For lock-in analysis you only need the shot-stream, so we store as (Rnew,1,Pnew)
        # by flattening kept shots and setting Rnew=1, Pnew=Nkept.
        counts2 = raw_good["counts"]  # (4,M,R,1,P)
        flat = counts2.reshape(4, M2, R2 * P2)  # (4,M,Nshots)
        flat = flat[:, :, keep_shots]  # keep good shots
        # rebuild to (4,M,R=1,S=1,P=Nkept)
        flat = flat.reshape(4, M2, 1, 1, flat.shape[-1])
        raw_good["counts"] = flat
        raw_good["goodshot_repacked"] = True

        info2 = summarize_global_bad_runs(raw_good, global_bad_thresh=global_bad_thresh)
        return raw_good, {"before": info0, "after_runs": info1, "after_shots": info2}

    return raw_good, {"before": info0, "after_runs": info1}


# -----------------------------
# Plotting / reporting
# -----------------------------


def plot_run_health(raw_data, title="Run health"):
    counts = np.asarray(raw_data["counts"], dtype=float)  # (4,M,R,S,P)
    _, M, R, S, P = counts.shape
    Ip, Im, Qp, Qm = counts

    denI = Ip + Im
    denQ = Qp + Qm

    mean_denI_per_run = np.nanmean(denI, axis=(0, 2, 3))  # [R]
    mean_denQ_per_run = np.nanmean(denQ, axis=(0, 2, 3))  # [R]
    frac_denI_pos = np.mean(denI > 0, axis=(0, 2, 3))
    frac_denQ_pos = np.mean(denQ > 0, axis=(0, 2, 3))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(mean_denI_per_run, marker=".", linewidth=1, label="mean(Ip+Im) per run")
    ax.plot(mean_denQ_per_run, marker=".", linewidth=1, label="mean(Qp+Qm) per run")
    ax.set_xlabel("run index")
    ax.set_ylabel("mean denominator (counts)")
    ax.legend()
    ax.set_title(title)
    return fig


def _nv_label(nv_obj, idx):
    for attr in ["name", "id", "sig", "label"]:
        if hasattr(nv_obj, attr):
            try:
                return str(getattr(nv_obj, attr))
            except Exception:
                pass
    return f"nv{idx}"


def print_dm_lockin_per_nv(raw_data, min_pairs=1000, min_var=1e-12, min_valid_frac=0.5):
    nv_list = raw_data["nv_list"]
    counts = np.asarray(raw_data["counts"])
    num_nvs = len(nv_list)

    s, I, Q = compute_iq_from_counts(counts)
    M, N = s.shape
    assert M == num_nvs, f"M mismatch: s has {M}, nv_list has {num_nvs}"

    nanI = np.mean(~np.isfinite(I), axis=1)
    nanQ = np.mean(~np.isfinite(Q), axis=1)

    I_mean = np.nanmean(I, axis=1)
    Q_mean = np.nanmean(Q, axis=1)
    I_std = np.nanstd(I, axis=1)
    Q_std = np.nanstd(Q, axis=1)
    s_abs_mean = np.nanmean(np.abs(s), axis=1)
    s_abs_std = np.nanstd(np.abs(s), axis=1)

    C, cnt, keep1 = nan_pairwise_covariance(
        s, remove_mean=True, min_pairs=min_pairs, min_valid_frac=min_valid_frac
    )
    coh, C2, keep2 = coherence_from_cov(C, min_var=min_var)
    evals, evecs, frac = dominant_mode_stats(C2)
    v = evecs[:, -1]

    keep1_inds = np.where(keep1)[0]
    keep2_inds_rel = np.where(keep2)[0]
    keep_final_inds = keep1_inds[keep2_inds_rel]

    var_full = np.full(num_nvs, np.nan, dtype=float)
    var_full[keep1_inds] = np.real(np.diag(C))

    v_full = np.full(num_nvs, np.nan + 1j * np.nan, dtype=np.complex128)
    v_full[keep_final_inds] = v

    # Fix global phase for readability (largest component -> phase 0)
    i0 = np.nanargmax(np.abs(v_full))
    v_full = v_full * np.exp(-1j * np.angle(v_full[i0]))

    median_pair = np.nanmedian(cnt[cnt > 0]) if np.any(cnt > 0) else np.nan

    print(f"[DM lock-in] M={M}, Nshots={N}")
    print(f"[DM lock-in] I NaN frac median={np.nanmedian(nanI):.6g}")
    print(f"[DM lock-in] Q NaN frac median={np.nanmedian(nanQ):.6g}")
    print(f"[DM lock-in] kept NVs after validity: {int(np.sum(keep1))} / {M}")
    print(
        f"[DM lock-in] kept NVs after var-cut: {int(np.sum(keep2))} / {int(np.sum(keep1))}"
    )
    print(f"[DM lock-in] median pair-count: {median_pair}")
    print(f"[DM lock-in] dominant mode power fraction: {frac:.6g}")
    print()

    header = (
        "idx  NV                nanI    nanQ     <I>      <Q>     stdI    stdQ    "
        "<|s|>   std|s|    var(s)     |v|     arg(v)[deg]"
    )
    print(header)
    print("-" * len(header))

    for i in range(num_nvs):
        name = _nv_label(nv_list[i], i)[:16].ljust(16)

        var_i = var_full[i]
        vi = v_full[i]
        vi_abs = np.abs(vi) if np.isfinite(vi.real) and np.isfinite(vi.imag) else np.nan
        vi_ang = (
            np.degrees(np.angle(vi))
            if np.isfinite(vi.real) and np.isfinite(vi.imag)
            else np.nan
        )

        print(
            f"{i:3d}  {name}  "
            f"{nanI[i]:.4f}  {nanQ[i]:.4f}  "
            f"{I_mean[i]: .5f}  {Q_mean[i]: .5f}  "
            f"{I_std[i]:.5f}  {Q_std[i]:.5f}  "
            f"{s_abs_mean[i]:.5f}  {s_abs_std[i]:.5f}  "
            f"{var_i: .3e}  "
            f"{vi_abs: .3e}  {vi_ang: .2f}"
        )

    return {
        "nanI": nanI,
        "nanQ": nanQ,
        "var_full": var_full,
        "v_full": v_full,
        "frac": frac,
        "evals": evals,
        "keep_final_inds": keep_final_inds,
        "coh": coh,
    }


def process_and_plot_dm_lockin(
    raw_data, min_pairs=1000, min_var=1e-12, min_valid_frac=0.5, show=True
):
    counts = np.asarray(raw_data["counts"])
    s, I, Q = compute_iq_from_counts(counts)

    C, cnt, keep1 = nan_pairwise_covariance(
        s, remove_mean=True, min_pairs=min_pairs, min_valid_frac=min_valid_frac
    )
    coh, C2, keep2 = coherence_from_cov(C, min_var=min_var)
    evals, evecs, frac = dominant_mode_stats(C2)
    v = evecs[:, -1]

    # Plots
    figs = []

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    mat1 = np.real(coh).copy()
    np.fill_diagonal(mat1, np.nan)
    kpl.imshow(
        ax1,
        mat1,
        title="Re[Coherence] of lock-in signal s = I + iQ",
        cbar_label="Re(coh)",
        cmap="RdBu_r",
        nan_color=kpl.KplColors.GRAY,
    )
    ax1.set_xlabel("NV index (kept)")
    ax1.set_ylabel("NV index (kept)")
    figs.append(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    mat2 = np.abs(coh).copy()
    np.fill_diagonal(mat2, np.nan)
    kpl.imshow(
        ax2,
        mat2,
        title="|Coherence| of lock-in signal",
        cbar_label="|coh|",
        cmap="viridis",
        nan_color=kpl.KplColors.GRAY,
    )
    ax2.set_xlabel("NV index (kept)")
    ax2.set_ylabel("NV index (kept)")
    figs.append(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    ax3.plot(np.arange(len(v)), np.abs(v), marker=".", linewidth=1)
    ax3.set_title(f"Dominant eigenvector |v| (cov)  (power frac={frac:.4f})")
    ax3.set_xlabel("NV index (kept)")
    ax3.set_ylabel("|v|")
    figs.append(fig3)

    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    ax4.plot(np.arange(len(evals)), evals, marker=".", linewidth=1)
    ax4.set_title("Eigenvalue spectrum (cov)")
    ax4.set_xlabel("mode index")
    ax4.set_ylabel("eigenvalue")
    figs.append(fig4)

    if show:
        kpl.show(block=True)

    return figs, {"coh": coh, "C2": C2, "evals": evals, "evecs": evecs, "frac": frac}


def project_common_mode(raw_good, min_pairs=1000, min_var=1e-12, min_valid_frac=0.5):
    counts = np.asarray(raw_good["counts"])
    s, I, Q = compute_iq_from_counts(counts)  # [M,N]
    C, cnt, keep1 = nan_pairwise_covariance(
        s, remove_mean=True, min_pairs=min_pairs, min_valid_frac=min_valid_frac
    )

    coh, C2, keep2 = coherence_from_cov(C, min_var=min_var)
    evals, evecs, frac = dominant_mode_stats(C2)
    v = evecs[:, -1]  # [M_kept]

    # Fix global phase for readability
    v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))

    # Rebuild s on the final kept set
    keep1_inds = np.where(keep1)[0]
    keep2_inds = np.where(keep2)[0]
    keep_final = keep1_inds[keep2_inds]
    s_keep = s[keep_final]  # [M_kept, N]

    # Common-mode time series a[k] = v^H s(:,k)
    a = np.conjugate(v) @ s_keep  # [N]

    return {
        "s_keep": s_keep,
        "keep_final": keep_final,
        "v": v,
        "evals": evals,
        "frac": frac,
        "a": a,
        "coh": coh,
        "C2": C2,
    }


def plot_common_mode_timeseries_and_psd(a, fs_shots=1.0, title_prefix="Common mode"):
    """
    a: complex [N] common-mode amplitude per shot.
    fs_shots: sampling rate in "shots per unit time". If you know real timing, set it.
    """
    a = np.asarray(a)
    N = a.size

    # time series (magnitude and phase)
    fig1, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(np.abs(a), linewidth=1)
    ax.set_title(f"{title_prefix}: |a[k]| over shots (N={N})")
    ax.set_xlabel("shot index k")
    ax.set_ylabel("|a|")

    # PSD of real part (or magnitude). Use rFFT.
    x = np.real(a - np.mean(a))
    win = np.hanning(N)
    X = np.fft.rfft(x * win)
    psd = (np.abs(X) ** 2) / (np.sum(win**2) + 1e-18)
    f = np.fft.rfftfreq(N, d=1.0 / fs_shots)

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(f, psd, linewidth=1)
    ax2.set_title(f"{title_prefix}: PSD of Re[a[k]] (Hann window)")
    ax2.set_xlabel(
        "frequency [cycles per shot]" if fs_shots == 1.0 else "frequency [Hz]"
    )
    ax2.set_ylabel("PSD (arb.)")
    ax2.set_yscale("log")
    ax2.set_xscale("log")

    return fig1, fig2


def subtract_rank1_common_mode(s_keep, v, a):
    """
    s_keep: [M,N], v:[M], a:[N]
    Removes rank-1 contribution v a^T (with a = v^H s).
    """
    return s_keep - v[:, None] * a[None, :]


def recheck_after_common_mode_removal(
    raw_good, min_pairs=1000, min_var=1e-12, min_valid_frac=0.5
):
    proj = project_common_mode(
        raw_good, min_pairs=min_pairs, min_var=min_var, min_valid_frac=min_valid_frac
    )
    s_keep = proj["s_keep"]
    v = proj["v"]
    a = proj["a"]

    s_res = subtract_rank1_common_mode(s_keep, v, a)

    # Recompute covariance/coherence on residuals
    C_res, cnt_res, keep_res = nan_pairwise_covariance(
        s_res, remove_mean=True, min_pairs=min_pairs, min_valid_frac=min_valid_frac
    )
    coh_res, C2_res, keep2_res = coherence_from_cov(C_res, min_var=min_var)
    evals_res, evecs_res, frac_res = dominant_mode_stats(C2_res)

    print(f"[Residual] dominant mode power fraction: {frac_res:.6g}")

    return {
        "proj": proj,
        "s_res": s_res,
        "coh_res": coh_res,
        "evals_res": evals_res,
        "frac_res": frac_res,
    }


def print_mode_spectrum_summary(evals, M=None, top=10):
    evals = np.asarray(evals, dtype=float)
    lam = evals[::-1]  # descending
    tr = lam.sum()
    if M is None:
        M = len(evals)
    baseline = 1.0 / M

    print(f"[spectrum] M={M}, baseline~1/M={baseline:.6f}")
    print(f"[spectrum] trace={tr:.6g}")
    print("[spectrum] top modes:")
    for k in range(min(top, len(lam))):
        print(f"  k={k+1:2d}: λ={lam[k]:.6g}, frac={lam[k]/tr:.6g}")

    if len(lam) >= 2:
        pred_after_rank1 = lam[1] / max(tr - lam[0], 1e-18)
        print(
            f"[spectrum] predicted frac after removing top mode (λ2/(tr-λ1)) = {pred_after_rank1:.6g}"
        )


def subtract_top_k_modes_from_cov_evecs(s_keep, evecs, k):
    """
    s_keep: [M,N] complex, already on the kept NV set and mean-removed (recommended).
    evecs : [M,M] eigenvectors of covariance (columns), orthonormal.
    k     : number of top modes to remove.
    """
    V = evecs[:, -k:]  # top-k eigenvectors
    A = V.conj().T @ s_keep  # [k,N] mode time series
    s_res = s_keep - V @ A  # remove rank-k subspace
    return s_res


def rank_k_removal_scan(
    raw_good, k_list=(1, 2, 3, 5, 8, 12), min_pairs=1000, min_var=1e-12
):
    # Build s on kept set (and mean-remove consistently)
    counts = np.asarray(raw_good["counts"])
    s, _, _ = compute_iq_from_counts(counts)  # [M,N], already good shots
    # Mean remove per NV
    s0 = s - np.mean(s, axis=1, keepdims=True)

    # Covariance on full set (no NaNs now)
    C = (s0 @ s0.conj().T) / s0.shape[1]
    C = 0.5 * (C + C.conj().T)

    evals, evecs = np.linalg.eigh(C)
    tr = np.sum(np.real(evals))
    frac0 = float(np.real(evals[-1]) / (tr + 1e-18))
    print(f"[rank-K] K=0: frac={frac0:.6g}")

    for k in k_list:
        s_res = subtract_top_k_modes_from_cov_evecs(s0, evecs, k)
        C_res = (s_res @ s_res.conj().T) / s_res.shape[1]
        C_res = 0.5 * (C_res + C_res.conj().T)
        evals_res = np.linalg.eigh(C_res)[0]
        tr_res = np.sum(np.real(evals_res))
        frac_res = float(np.real(evals_res[-1]) / (tr_res + 1e-18))
        print(f"[rank-K] K={k:2d}: frac={frac_res:.6g}")


import numpy as np
import matplotlib.pyplot as plt

from utils import data_manager as dm
from utils import kplotlib as kpl


# -----------------------------
# Core helpers (vectorized)
# -----------------------------


def compute_iq_from_counts_keep_shape(counts, eps=1e-12, return_masks=False):
    """
    counts: (4, M, R, S, P)  with quadratures [Ip, Im, Qp, Qm]
    returns I,Q,s with shape (M, R, S, P)
    """
    counts = np.asarray(counts)
    if counts.dtype == object:
        counts = counts.astype(float)

    Ip = counts[0].astype(float)
    Im = counts[1].astype(float)
    Qp = counts[2].astype(float)
    Qm = counts[3].astype(float)

    denI = Ip + Im
    denQ = Qp + Qm

    mI = np.isfinite(Ip) & np.isfinite(Im) & np.isfinite(denI) & (denI > 0)
    mQ = np.isfinite(Qp) & np.isfinite(Qm) & np.isfinite(denQ) & (denQ > 0)

    I = np.full_like(denI, np.nan, dtype=float)
    Q = np.full_like(denQ, np.nan, dtype=float)

    I[mI] = (Ip[mI] - Im[mI]) / (denI[mI] + eps)
    Q[mQ] = (Qp[mQ] - Qm[mQ]) / (denQ[mQ] + eps)

    s = I + 1j * Q

    if return_masks:
        return s, I, Q, mI, mQ
    return s, I, Q


def flatten_shots(x_mrsp):
    """Flatten (M,R,S,P) -> (M, Nshots) with time order preserved."""
    M = x_mrsp.shape[0]
    return x_mrsp.reshape(M, -1)


def find_global_bad_runs(raw_data, global_bad_thresh=0.9):
    counts = np.asarray(raw_data["counts"])
    _, M, R, S, P = counts.shape

    s, I, Q, mI, mQ = compute_iq_from_counts_keep_shape(counts, return_masks=True)
    bad = (~mI) | (~mQ)  # (M,R,S,P)
    bad_flat = flatten_shots(bad.astype(float))  # (M,N)
    frac_bad_per_shot = bad_flat.mean(axis=0)  # (N,)

    global_bad_shots = np.where(frac_bad_per_shot > global_bad_thresh)[0]
    global_bad_mask_3d = (frac_bad_per_shot > global_bad_thresh).reshape(R, S, P)

    global_bad_frac_per_run = global_bad_mask_3d.mean(axis=(1, 2))  # (R,)
    bad_runs = np.where(global_bad_frac_per_run > global_bad_thresh)[0]

    return {
        "bad_runs": bad_runs,
        "global_bad_shots": global_bad_shots,
        "global_bad_frac_per_run": global_bad_frac_per_run,
        "R": R,
        "S": S,
        "P": P,
        "M": M,
    }


def filter_runs_in_raw_data(raw_data, keep_runs):
    """Return a shallow copy of raw_data with counts filtered along run axis."""
    out = dict(raw_data)
    counts = np.asarray(raw_data["counts"])
    out["counts"] = counts[:, :, keep_runs, :, :]
    out["kept_run_inds"] = np.asarray(keep_runs, dtype=int)
    return out


def detrend_per_run(s_mrsp, mode="none"):
    """
    s_mrsp: (M,R,S,P)
    mode:
      - "none": no detrend
      - "runmean": subtract per-NV, per-run mean over reps (and steps)
    """
    if mode == "none":
        return s_mrsp

    if mode == "runmean":
        # mean over (S,P) within each run, per NV
        mu = np.nanmean(s_mrsp, axis=(2, 3), keepdims=True)  # (M,R,1,1)
        return s_mrsp - mu

    raise ValueError(f"Unknown detrend mode: {mode}")


def whiten_per_nv(s_mn, eps=1e-12):
    """Whiten complex data per NV: subtract mean, divide by std(|.|) or std of real/imag."""
    s = s_mn.copy()
    mu = np.mean(s, axis=1, keepdims=True)
    s = s - mu
    # Use std of complex magnitude as a robust-ish scale
    scale = np.std(np.abs(s), axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    return s / scale


def corr_matrix_from_samples(z_mn):
    """Correlation-ish matrix from whitened complex samples z."""
    N = z_mn.shape[1]
    C = (z_mn @ z_mn.conj().T) / max(N, 1)
    return 0.5 * (C + C.conj().T)


def eig_spectrum(C):
    w, V = np.linalg.eigh(0.5 * (C + C.conj().T))
    w = np.real(w)
    order = np.argsort(w)
    return w[order], V[:, order]


def participation_ratio(v):
    """For normalized v: PR = 1 / sum(|v|^4).  Range ~[1, M]."""
    a2 = np.abs(v) ** 2
    s4 = np.sum(a2**2)
    return 1.0 / max(s4, 1e-18)


def marchenko_pastur_edge(M, N):
    """
    For i.i.d. noise on correlation matrix, lambda_max ~ (1+sqrt(q))^2 with q=M/N.
    """
    q = M / max(N, 1)
    return (1.0 + np.sqrt(q)) ** 2


def subtract_top_k(z_mn, V, k):
    """Project out top-k eigenvectors (columns of V) from whitened data."""
    Vk = V[:, -k:]  # top-k
    A = Vk.conj().T @ z_mn  # (k,N)
    return z_mn - Vk @ A


# -----------------------------
# Main analysis pipeline
# -----------------------------
def top_participants(v, nv_list=None, top_k=10):
    amps = np.abs(v)
    order = np.argsort(amps)[::-1][:top_k]
    i0 = order[0]
    v0 = v * np.exp(-1j * np.angle(v[i0]))  # set global phase

    print("Top participants:")
    for i in order:
        name = (
            getattr(nv_list[i], "name", f"nv{i}") if nv_list is not None else f"nv{i}"
        )
        print(
            f"  {i:3d}  {str(name)[:22].ljust(22)}  |v|={np.abs(v0[i]):.3e}  phase={np.degrees(np.angle(v0[i])):7.2f} deg"
        )


def analyze_good_shot_data(
    file_stem,
    global_bad_thresh=0.9,
    detrend="runmean",  # "none" or "runmean"
    do_plots=True,
    k_list=(1, 2, 3, 5, 8, 12, 20, 30),
):
    raw = dm.get_raw_data(file_stem=file_stem, load_npz=True)
    counts = np.asarray(raw["counts"])
    _, M, R, S, P = counts.shape

    # --- detect + filter bad runs
    diag = find_global_bad_runs(raw, global_bad_thresh=global_bad_thresh)
    bad_runs = diag["bad_runs"]

    print("=== BEFORE filtering ===")
    print(f"counts shape: {counts.shape}")
    print(
        f"global-bad shots: {len(diag['global_bad_shots'])} ({len(diag['global_bad_shots'])/(R*S*P):.3%})"
    )
    if len(bad_runs) > 0:
        print(
            f"bad runs: {len(bad_runs)}  range={bad_runs[0]}..{bad_runs[-1]}  trailing={np.all(bad_runs==np.arange(bad_runs[0], R))}"
        )
    else:
        print("bad runs: none")

    if len(bad_runs) > 0:
        keep_runs = np.setdiff1d(np.arange(R), bad_runs)
        raw_good = filter_runs_in_raw_data(raw, keep_runs)
    else:
        raw_good = raw

    counts_g = np.asarray(raw_good["counts"])
    _, M2, R2, S2, P2 = counts_g.shape

    diag2 = find_global_bad_runs(raw_good, global_bad_thresh=global_bad_thresh)
    print("\n=== AFTER filtering ===")
    print(f"counts shape: {counts_g.shape}")
    print(
        f"global-bad shots: {len(diag2['global_bad_shots'])} ({len(diag2['global_bad_shots'])/(R2*S2*P2):.3%})"
    )
    if len(diag2["bad_runs"]) > 0:
        print(f"bad runs: {len(diag2['bad_runs'])}")
    else:
        print("bad runs: none")

    # --- build good-shot lock-in data
    s_mrsp, I_mrsp, Q_mrsp = compute_iq_from_counts_keep_shape(counts_g)
    # sanity: after filtering you expect no NaNs
    s_mn = flatten_shots(s_mrsp)
    nan_frac = np.mean(~np.isfinite(s_mn))
    if nan_frac > 0:
        print(f"[warn] still have NaNs in s: frac={nan_frac:.3e}")

    # --- detrend then whiten
    s_dt = detrend_per_run(s_mrsp, mode=detrend)
    s_dt_mn = flatten_shots(s_dt)

    z = whiten_per_nv(s_dt_mn)  # (M, N)
    N = z.shape[1]
    z = whiten_per_nv_complex_power(s_dt_mn)
    C = (z @ z.conj().T) / z.shape[1]
    C = 0.5 * (C + C.conj().T)
    print("trace(C) =", np.trace(C).real)  # should be ~M
    print("median diag(C) =", np.median(np.diag(C).real))  # should be ~1

    # --- correlation PCA
    # C = corr_matrix_from_samples(z)
    w, V = eig_spectrum(C)
    v = V[:, -1]
    top_participants(v, raw_good.get("nv_list", None), top_k=15)

    tr = np.sum(w)
    frac0 = float(w[-1] / (tr + 1e-18))
    lam_max = float(w[-1])
    mp_edge = marchenko_pastur_edge(M2, N)
    pr0 = participation_ratio(V[:, -1])

    print(f"\n[coh-PCA/whitened] M={M2}, N={N}, detrend={detrend}")
    print(f"[coh-PCA/whitened] trace={tr:.6g} (should be ~M if fully whitened)")
    print(
        f"[coh-PCA/whitened] λmax={lam_max:.6g}, frac={frac0:.6g}, baseline~1/M={1/M2:.6g}"
    )
    print(
        f"[coh-PCA/whitened] MP noise λmax~(1+sqrt(M/N))^2 ≈ {mp_edge:.6g}  (if i.i.d. noise)"
    )
    print(
        f"[coh-PCA/whitened] dominant-mode participation ratio PR≈{pr0:.2f} (uniform would be ~{M2})"
    )

    # --- rank-K removal scan on whitened data
    print("\n[rank-K / whitened]")
    print(f"K= 0: frac={frac0:.6g}")
    fracKs = [frac0]
    for k in k_list:
        z_res = subtract_top_k(z, V, k)
        C_res = corr_matrix_from_samples(z_res)
        w_res, _ = eig_spectrum(C_res)
        frac = float(w_res[-1] / (np.sum(w_res) + 1e-18))
        fracKs.append(frac)
        print(f"K={k:2d}: frac={frac:.6g}   (baseline 1/(M-K)≈{1/(M2-k):.6g})")

    # --- plots
    if do_plots:
        # eigenvalues (top 30)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(w[::-1][:40], marker=".", linewidth=1)
        ax.set_title("Whitened correlation eigenvalues (top)")
        ax.set_xlabel("mode rank (descending)")
        ax.set_ylabel("eigenvalue")
        plt.show()

        # frac vs K
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        Ks = [0] + list(k_list)
        ax2.plot(Ks, fracKs, marker=".", linewidth=1)
        ax2.set_title("Dominant-mode fraction after removing top-K modes (whitened)")
        ax2.set_xlabel("K removed")
        ax2.set_ylabel("λmax / trace")
        plt.show()

        # dominant mode weights
        v = V[:, -1]
        fig3, ax3 = plt.subplots(figsize=(7, 3.5))
        ax3.plot(np.abs(v), marker=".", linewidth=0.8)
        ax3.set_title(f"Dominant eigenvector |v| (PR≈{pr0:.1f})")
        ax3.set_xlabel("NV index")
        ax3.set_ylabel("|v|")
        plt.show()

    return {
        "raw_good": raw_good,
        "z": z,
        "C": C,
        "evals": w,
        "evecs": V,
        "frac0": frac0,
        "mp_edge": mp_edge,
        "PR0": pr0,
        "s_dt_mn": s_dt_mn,  # detrended (but not whitened) complex data
        "v": v,
    }


def per_nv_power(s_mn):
    mu = np.mean(s_mn, axis=1, keepdims=True)
    x = s_mn - mu
    return np.mean(np.abs(x) ** 2, axis=1)


def frac_excess(fracs, M, Ks):
    out = []
    for frac, K in zip(fracs, Ks):
        base = 1.0 / (M - K)
        out.append((K, frac, base, frac - base))
    return out


def iterative_remove_top_modes(z, Kmax):
    z_res = z.copy()
    fracs = []
    lams = []
    prs = []

    for k in range(Kmax + 1):
        C = (z_res @ z_res.conj().T) / z_res.shape[1]
        C = 0.5 * (C + C.conj().T)
        w, V = eig_spectrum(C)
        frac = float(w[-1] / (np.sum(w) + 1e-18))
        v = V[:, -1]
        pr = participation_ratio(v)

        fracs.append(frac)
        lams.append(float(w[-1]))
        prs.append(float(pr))

        if k < Kmax:
            # remove current top mode
            z_res = subtract_top_k(z_res, V, 1)

    return np.array(fracs), np.array(lams), np.array(prs)


def mode_timeseries(z_mn, v_m):
    # z_mn: (M,N), v_m: (M,)
    return np.conj(v_m) @ z_mn  # (N,) complex


import numpy as np
import matplotlib.pyplot as plt


def _mp_edge(M, N_eff):
    # Marchenko–Pastur upper edge for sample covariance of i.i.d. noise
    q = M / max(N_eff, 1.0)
    return (1.0 + np.sqrt(q)) ** 2


def _autocorr(x, max_lag=2000):
    """
    Normalized autocorrelation r[k] for k=0..max_lag.
    Uses FFT for speed. x must be 1D real.
    """
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    x = np.where(np.isfinite(x), x, 0.0)

    n = len(x)
    max_lag = min(max_lag, n - 1)
    # zero-pad to power of 2
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(X * np.conj(X), n=nfft)[:n]
    ac = ac / (ac[0] + 1e-18)
    return ac[: max_lag + 1]


def _effective_sample_size_from_autocorr(r, N):
    """
    Neff ≈ N / (1 + 2 * sum_{k=1..K} r[k])  (truncate when r becomes negative)
    """
    r = np.asarray(r, float)
    # truncate sum when autocorr drops below 0 (common practice for Neff)
    ks = np.arange(1, len(r))
    pos = r[1:] > 0
    if not np.any(pos):
        return float(N)
    kmax = np.max(ks[pos])
    tau = 1.0 + 2.0 * np.sum(r[1 : kmax + 1])
    return float(N / max(tau, 1e-12))


def _mode_timeseries(z_mn, v_m):
    # a[n] = v^H z[:,n]
    return np.conj(v_m) @ z_mn


def _brightness_proxy_from_counts(counts):
    """
    counts: (4, M, R, S, P)
    returns bright_shot: (Nshots,) mean total denominator proxy per shot
    """
    counts = np.asarray(counts, float)
    Ip, Im, Qp, Qm = counts
    den = Ip + Im + Qp + Qm  # (M,R,S,P)
    bright = den.mean(axis=0).reshape(-1)  # mean over NVs -> (R*S*P,)
    return bright


def _global_iq_proxies(I_mn, Q_mn):
    # mean across NVs (per shot), and RMS across NVs
    gI = np.mean(I_mn, axis=0)
    gQ = np.mean(Q_mn, axis=0)
    rI = np.sqrt(np.mean(I_mn**2, axis=0))
    rQ = np.sqrt(np.mean(Q_mn**2, axis=0))
    return gI, gQ, rI, rQ


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    return float((aa @ bb) / (np.sqrt((aa @ aa) * (bb @ bb)) + 1e-18))


def diagnose_mode_temporal(raw_good, out, max_lag=2000, show_plots=True):
    """
    raw_good: filtered raw_data dict (no global-bad runs)
    out: dict containing at least:
        out["z"] : (M,N) whitened data
        out["v"] : (M,) dominant eigenvector for whitened C
        optionally out["lam"] or out["evals"] if you want
    """

    counts = np.asarray(raw_good["counts"], float)
    _, M, R, S, P = counts.shape
    N = R * S * P

    # recompute I/Q so we can make proxies (robust even if out doesn't store them)
    s, I, Q = compute_iq_from_counts(counts)  # your function; I/Q are (M,N)

    z = out["z"]
    v = out["v"]
    assert z.shape == (M, N), f"z shape {z.shape} != (M,N)=({M},{N})"
    assert v.shape == (M,), f"v shape {v.shape} != (M,)"

    a = _mode_timeseries(z, v)  # complex (N,)
    a_re = np.real(a)
    a_im = np.imag(a)

    bright = _brightness_proxy_from_counts(counts)
    gI, gQ, rI, rQ = _global_iq_proxies(I, Q)

    print("\n[mode-temporal]")
    print(f"M={M}, N={N}, runs={R}, steps={S}, reps={P}")

    # correlations to common proxies
    print(f"corr(Re[a], brightness) = {_corr(a_re, bright):.6g}")
    print(f"corr(Re[a], mean(I))    = {_corr(a_re, gI):.6g}")
    print(f"corr(Re[a], mean(Q))    = {_corr(a_re, gQ):.6g}")
    print(f"corr(Re[a], rms(I))     = {_corr(a_re, rI):.6g}")
    print(f"corr(Re[a], rms(Q))     = {_corr(a_re, rQ):.6g}")
    print(f"corr(Im[a], brightness) = {_corr(a_im, bright):.6g}")

    # Neff estimate from autocorr of the mode time series
    r = _autocorr(a_re, max_lag=5000)
    Neff = effective_sample_size_from_autocorr_first_zero(r, N)
    print("Neff =", Neff)
    print("MP edge =", (1 + np.sqrt(M / Neff)) ** 2)

    # r = _autocorr(a_re, max_lag=max_lag)
    # Neff = _effective_sample_size_from_autocorr(r, N)
    mp_edge = _mp_edge(M, Neff)
    mp_edge_naive = _mp_edge(M, N)

    print(
        f"\nNeff estimate from autocorr(Re[a]) with max_lag={len(r)-1}: Neff≈{Neff:.1f} (N={N})"
    )
    print(f"MP edge using N:    λmax≈{mp_edge_naive:.6g}")
    print(f"MP edge using Neff: λmax≈{mp_edge:.6g}")

    if show_plots:
        # 1) autocorr
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        ax.plot(r, marker=".", linewidth=1)
        ax.set_title("Autocorr of Re[a] (dominant-mode time series)")
        ax.set_xlabel("lag [shots]")
        ax.set_ylabel("r[lag]")
        ax.axhline(0, linewidth=1)
        plt.tight_layout()

        # 2) time series snippets
        fig2, ax2 = plt.subplots(figsize=(9, 3.2))
        nshow = min(5000, N)
        ax2.plot(a_re[:nshow], linewidth=1, label="Re[a]")
        ax2.plot(
            (bright[:nshow] - np.mean(bright[:nshow]))
            / (np.std(bright[:nshow]) + 1e-18),
            linewidth=1,
            label="brightness (z-scored)",
        )
        ax2.set_title("First shots: mode vs brightness proxy")
        ax2.set_xlabel("shot index")
        ax2.legend()
        plt.tight_layout()

        # 3) per-run mean of mode
        a3 = a_re.reshape(R, S, P).mean(axis=(1, 2))
        fig3, ax3 = plt.subplots(figsize=(9, 3.2))
        ax3.plot(a3, marker=".", linewidth=1)
        ax3.set_title("Per-run mean of Re[a] (after your detrend)")
        ax3.set_xlabel("run index")
        ax3.set_ylabel("mean Re[a]")
        plt.tight_layout()

        plt.show()

    return {
        "a": a,
        "a_re": a_re,
        "a_im": a_im,
        "bright": bright,
        "gI": gI,
        "gQ": gQ,
        "r": r,
        "Neff": Neff,
        "mp_edge_N": mp_edge_naive,
        "mp_edge_Neff": mp_edge,
    }


def effective_sample_size_from_autocorr_first_zero(r, N, eps=1e-12):
    """
    r: autocorr array with r[0]=1
    Neff = N / (1 + 2*sum_{k=1..K} r[k]), where K is the last positive lag
           before first non-positive crossing.
    Caps Neff at N.
    """
    r = np.asarray(r, float)
    if len(r) < 2:
        return float(N)

    # Find first lag where r <= 0 (after lag 0). Stop before that.
    nz = np.where(r[1:] <= 0)[0]
    if len(nz) == 0:
        K = len(r) - 1
    else:
        K = int(nz[0])  # index into r[1:], so r[1:][K] <= 0, sum up to r[1+K-1]
        K = max(K, 1)

    # sum r[1]..r[K] where K is last strictly-positive block endpoint (exclusive of crossing)
    # If crossing at r[1] already, K=1 => empty sum.
    s = np.sum(r[1:K]) if K > 1 else 0.0
    tau = 1.0 + 2.0 * s
    Neff = float(N / max(tau, eps))
    return float(min(Neff, N))


def regress_out_global_IQ(I, Q, use_Q=True):
    """
    Remove common-mode regressors from each NV time series:
      regressors: gI = mean over NVs per shot, (and optionally gQ)
    Returns I_res, Q_res, betas
    """
    I = np.asarray(I, float)
    Q = np.asarray(Q, float)
    M, N = I.shape

    gI = np.mean(I, axis=0)
    gQ = np.mean(Q, axis=0)

    # build design matrix X: (K,N)
    if use_Q:
        X = np.vstack([gI, gQ])  # K=2
    else:
        X = np.vstack([gI])  # K=1

    # center regressors (important)
    X = X - X.mean(axis=1, keepdims=True)

    # precompute (X X^T)^{-1} X for speed: (K,K) and (K,N)
    XXT = X @ X.T
    XXT_inv = np.linalg.pinv(XXT)
    P = XXT_inv @ X  # (K,N)

    # For each NV: beta = y X^T (X X^T)^{-1}  => beta = y @ X.T @ inv(XXT)
    # Equivalent using P: beta = y @ P.T
    beta_I = I @ P.T  # (M,K)
    I_fit = beta_I @ X  # (M,N)
    I_res = I - I_fit

    beta_Q = Q @ P.T
    Q_fit = beta_Q @ X
    Q_res = Q - Q_fit

    return I_res, Q_res, {"gI": gI, "gQ": gQ, "beta_I": beta_I, "beta_Q": beta_Q}


import numpy as np

# ----------------------------
# Core utilities
# ----------------------------


def _flatten_shots(x):
    return np.reshape(x, (-1,))


def compute_iq_from_counts(counts, eps=1e-12, return_masks=False):
    """
    counts shape: (4, M, R, S, P)
    Returns:
      s, I, Q with shape (M, N=R*S*P)
      masks mI/mQ (same shape) if return_masks
    """
    counts = np.asarray(counts)
    if counts.dtype == object:
        counts = counts.astype(float)

    assert counts.ndim == 5, f"Expected 5D counts, got {counts.shape}"
    Ip = np.asarray(counts[0], float)
    Im = np.asarray(counts[1], float)
    Qp = np.asarray(counts[2], float)
    Qm = np.asarray(counts[3], float)

    M = Ip.shape[0]
    Ip_f = np.stack([_flatten_shots(Ip[nv]) for nv in range(M)], axis=0)
    Im_f = np.stack([_flatten_shots(Im[nv]) for nv in range(M)], axis=0)
    Qp_f = np.stack([_flatten_shots(Qp[nv]) for nv in range(M)], axis=0)
    Qm_f = np.stack([_flatten_shots(Qm[nv]) for nv in range(M)], axis=0)

    denI = Ip_f + Im_f
    denQ = Qp_f + Qm_f

    I = np.full_like(denI, np.nan, float)
    Q = np.full_like(denQ, np.nan, float)

    mI = np.isfinite(Ip_f) & np.isfinite(Im_f) & np.isfinite(denI) & (denI > 0)
    mQ = np.isfinite(Qp_f) & np.isfinite(Qm_f) & np.isfinite(denQ) & (denQ > 0)

    I[mI] = (Ip_f[mI] - Im_f[mI]) / (denI[mI] + eps)
    Q[mQ] = (Qp_f[mQ] - Qm_f[mQ]) / (denQ[mQ] + eps)

    s = I + 1j * Q
    if return_masks:
        return s, I, Q, mI, mQ
    return s, I, Q


def find_trailing_global_bad_run_cut(counts, global_bad_thresh=0.9):
    """
    Identify trailing run block where shots are globally invalid.
    Returns R_keep (number of runs to keep).
    """
    counts = np.asarray(counts)
    _, M, R, S, P = counts.shape
    _, _, _, mI, mQ = compute_iq_from_counts(counts, return_masks=True)
    bad = (~mI) | (~mQ)  # (M, N)
    frac_bad_shot = bad.mean(axis=0)  # (N,)
    global_bad = (frac_bad_shot > global_bad_thresh).reshape(R, S, P)
    frac_bad_run = global_bad.mean(axis=(1, 2))  # (R,)

    bad_runs = np.where(frac_bad_run > global_bad_thresh)[0]
    if len(bad_runs) == 0:
        return R
    # if it's trailing contiguous, cut at first bad run
    r0 = int(bad_runs[0])
    trailing = np.all(bad_runs == np.arange(r0, R))
    return r0 if trailing else R


def detrend_runmean(x, R, S, P):
    """
    x: (M,N) where N=R*S*P
    subtract per-run mean across reps (and steps)
    """
    M, N = x.shape
    x4 = x.reshape(M, R, S, P)
    mu = np.mean(x4, axis=(2, 3), keepdims=True)  # mean over (S,P) per run
    x4_dt = x4 - mu
    return x4_dt.reshape(M, N)


def regress_out_global_IQ(I, Q, use_Q=True):
    """
    Regress out gI(t)=mean_nvs I_nv(t) and optionally gQ(t).
    """
    I = np.asarray(I, float)
    Q = np.asarray(Q, float)
    M, N = I.shape

    gI = np.mean(I, axis=0)
    gQ = np.mean(Q, axis=0)

    if use_Q:
        X = np.vstack([gI, gQ])  # (K,N), K=2
    else:
        X = np.vstack([gI])  # (1,N)

    X = X - X.mean(axis=1, keepdims=True)
    XXT = X @ X.T
    XXT_inv = np.linalg.pinv(XXT)
    P = XXT_inv @ X  # (K,N)

    betaI = I @ P.T  # (M,K)
    betaQ = Q @ P.T
    I_res = I - (betaI @ X)
    Q_res = Q - (betaQ @ X)

    return I_res, Q_res, {"gI": gI, "gQ": gQ, "betaI": betaI, "betaQ": betaQ}


def whiten_per_nv(s, eps=1e-12):
    """
    s: complex (M,N). Return s_whitened so that mean power per NV is 1.
    """
    p = np.mean(np.abs(s) ** 2, axis=1)  # (M,)
    scale = 1.0 / np.sqrt(np.maximum(p, eps))  # (M,)
    return s * scale[:, None], scale, p


def coherence_matrix(s_w):
    """
    s_w: complex (M,N) whitened so E|s|^2=1 per NV.
    Returns C = (s_w s_w^H)/N so diag ~ 1.
    """
    M, N = s_w.shape
    C = (s_w @ s_w.conj().T) / float(N)
    C = 0.5 * (C + C.conj().T)
    return C


def eig_hermitian(C):
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(np.real(evals))[::-1]
    evals = np.real(evals[order])
    evecs = evecs[:, order]
    return evals, evecs


def participation_ratio(v):
    # v normalized by eigh; PR = 1 / sum |v|^4
    w = np.abs(v) ** 2
    return float(1.0 / np.sum(w**2))


def overlap_with_uniform(v):
    v = np.asarray(v)
    M = v.size
    u = np.ones(M) / np.sqrt(M)
    return float(np.abs(np.vdot(u, v)) / (np.linalg.norm(v) + 1e-18))


def _autocorr_fft(x, max_lag=5000):
    """
    Autocorr of 1D real signal using FFT, normalized so r[0]=1.
    """
    x = np.asarray(x, float)
    x = x - np.mean(x)
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, n=nfft)
    S = X * np.conj(X)
    r = np.fft.irfft(S, n=nfft)[: max_lag + 1]
    r = r / max(r[0], 1e-18)
    return r


def effective_sample_size_first_zero(r, N, eps=1e-12):
    """
    Neff = N / (1 + 2*sum_{k=1..K} r[k]) with truncation at first r<=0.
    Always returns Neff <= N.
    """
    r = np.asarray(r, float)
    if len(r) < 2:
        return float(N)
    nz = np.where(r[1:] <= 0)[0]
    if len(nz) == 0:
        K = len(r) - 1
    else:
        K = max(int(nz[0]), 1)
    s = np.sum(r[1:K]) if K > 1 else 0.0
    tau = 1.0 + 2.0 * s
    Neff = float(N / max(tau, eps))
    return float(min(Neff, N))


def _nv_label(nv_obj, idx):
    for attr in ["name", "id", "sig", "label"]:
        if hasattr(nv_obj, attr):
            try:
                return str(getattr(nv_obj, attr))
            except Exception:
                pass
    return f"nv{idx}"


# ----------------------------
# Main analysis
# ----------------------------


def analyze_good_shots(
    raw_data,
    global_bad_thresh=0.9,
    detrend="runmean",
    regress_global=True,
    regress_use_Q=True,
    top_k=15,
    neff_max_lag=5000,
):
    counts = np.asarray(raw_data["counts"])
    nv_list = raw_data.get("nv_list", None)
    _, M, R, S, P = counts.shape
    N0 = R * S * P

    # ---- filter trailing bad runs (your case) ----
    R_keep = find_trailing_global_bad_run_cut(
        counts, global_bad_thresh=global_bad_thresh
    )
    if R_keep < R:
        counts = counts[:, :, :R_keep, :, :]
        R = R_keep
    N = R * S * P

    print("=== AFTER filtering ===")
    print(f"counts shape: {counts.shape}")

    # ---- lock-in ----
    s, I, Q, mI, mQ = compute_iq_from_counts(counts, return_masks=True)
    if np.any(~np.isfinite(I)) or np.any(~np.isfinite(Q)):
        raise RuntimeError(
            "Still have NaNs after filtering; expected 'good-shot only' here."
        )

    # ---- detrend ----
    if detrend == "runmean":
        I_dt = detrend_runmean(I, R, S, P)
        Q_dt = detrend_runmean(Q, R, S, P)
    elif detrend in [None, "none"]:
        I_dt, Q_dt = I, Q
    else:
        raise ValueError(f"Unknown detrend={detrend}")

    # ---- regress out global mean(I/Q) ----
    if regress_global:
        I_dt, Q_dt, reg = regress_out_global_IQ(I_dt, Q_dt, use_Q=regress_use_Q)
    else:
        reg = None

    s_dt = I_dt + 1j * Q_dt

    # ---- whiten ----
    s_w, scale, p = whiten_per_nv(s_dt)
    C = coherence_matrix(s_w)

    print(f"trace(C) = {np.trace(C).real:.6g}")
    print(f"median diag(C) = {np.median(np.real(np.diag(C))):.6g}")

    # ---- eig ----
    evals, evecs = eig_hermitian(C)
    lam1 = float(evals[0])
    frac = lam1 / float(np.sum(evals) + 1e-18)
    baseline = 1.0 / M
    v1 = evecs[:, 0]
    PR = participation_ratio(v1)
    ov = overlap_with_uniform(v1)

    print()
    tag = "coh-PCA/whitened"
    if regress_global:
        tag += "+regress(gI,gQ)" if regress_use_Q else "+regress(gI)"
    print(f"[{tag}] M={M}, N={N}, detrend={detrend}")
    print(f"[{tag}] λmax={lam1:.6f}, frac={frac:.8f}, baseline~1/M={baseline:.8f}")
    print(f"[{tag}] PR≈{PR:.2f} (uniform would be ~{M})")
    print(f"[{tag}] overlap with uniform ≈ {ov:.3f}")

    # MP edge (use N; then optionally Neff from mode amplitude below)
    mp_N = (1 + np.sqrt(M / N)) ** 2
    print(f"[{tag}] MP noise edge using N: λmax≈{mp_N:.6f}")

    # ---- top participants ----
    # Fix global phase for readability
    i0 = int(np.argmax(np.abs(v1)))
    v1p = v1 * np.exp(-1j * np.angle(v1[i0]))

    print("Top participants:")
    inds = np.argsort(np.abs(v1p))[::-1][:top_k]
    for i in inds:
        name = _nv_label(nv_list[i], i) if nv_list is not None else f"nv{i}"
        amp = np.abs(v1p[i])
        ph = np.degrees(np.angle(v1p[i]))
        print(
            f"  {i:3d}  {str(name)[:20].ljust(20)}  |v|={amp:.3e}  phase={ph:7.2f} deg"
        )

    # ---- temporal mode amplitude a(t) and Neff ----
    # a(t) = v^H s_w(t)
    a = (v1.conj().T @ s_w).astype(np.complex128)  # (N,)
    a_re = np.real(a)
    r = _autocorr_fft(a_re, max_lag=neff_max_lag)
    Neff = effective_sample_size_first_zero(r, N)
    mp_Neff = (1 + np.sqrt(M / Neff)) ** 2
    print()
    print(f"[mode-temporal] Neff(first-zero)≈{Neff:.1f} (N={N})")
    print(f"[mode-temporal] MP edge using Neff: λmax≈{mp_Neff:.6f}")

    # helpful nuisance correlations (computed on detrended/regressed series)
    gI = np.mean(I_dt, axis=0)
    gQ = np.mean(Q_dt, axis=0)
    print(f"[mode-temporal] corr(Re[a], mean(I_dt)) = {np.corrcoef(a_re, gI)[0,1]:.6g}")
    print(f"[mode-temporal] corr(Re[a], mean(Q_dt)) = {np.corrcoef(a_re, gQ)[0,1]:.6g}")

    out = {
        "counts_filtered": counts,
        "I_dt": I_dt,
        "Q_dt": Q_dt,
        "s_w": s_w,
        "C": C,
        "evals": evals,
        "evecs": evecs,
        "v1": v1p,
        "a": a,
        "Neff": Neff,
        "mp_edge_N": mp_N,
        "mp_edge_Neff": mp_Neff,
        "reg": reg,
    }
    return out


import numpy as np
import matplotlib.pyplot as plt


def _get_nv_xy(nv, i):
    """
    Best-effort: return (x,y) in pixels (or whatever the NV stores).
    Edit this once you know the exact attribute names in your NV objects.
    """
    # Common patterns I've seen in widefield NV codebases:
    cand = [
        "pixel_coords",
        "pixel_coord",
        "coords_px",
        "coord_px",
        "coords",
        "coord",
        "xy",
        "pos",
        "position",
        "img_coords",
        "camera_coords",
    ]
    for a in cand:
        if hasattr(nv, a):
            val = getattr(nv, a)
            try:
                val = np.asarray(val).astype(float).ravel()
                if val.size >= 2:
                    return float(val[0]), float(val[1])
            except Exception:
                pass
    # fallback: put them on a line if coords missing
    return float(i), 0.0


def plot_mode_spatial(raw_data, out, top_n=40, title="Top mode spatial map"):
    """
    raw_data: original raw_data (for nv_list)
    out: dict returned by analyze_good_shots(...)
         uses out["v1"] = phase-fixed dominant eigenvector on NVs
    """
    nv_list = raw_data["nv_list"]
    v = out["v1"]
    M = len(nv_list)

    xy = np.array([_get_nv_xy(nv_list[i], i) for i in range(M)], float)
    x, y = xy[:, 0], xy[:, 1]
    amp = np.abs(v)
    ph = np.angle(v)

    # plot: size ~ amp^2, color ~ phase
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x, y, s=20 + 4000 * (amp**2), c=ph, cmap="twilight", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    plt.colorbar(sc, ax=ax, label="phase(v) [rad]")

    # annotate top-N
    top = np.argsort(amp)[::-1][:top_n]
    for i in top:
        ax.text(x[i], y[i], str(i), fontsize=7)

    plt.tight_layout()
    plt.show()


def nearest_neighbor_stats(xy, idx):
    """
    Mean nearest-neighbor distance among a set of indices.
    """
    pts = xy[idx]
    if len(pts) < 2:
        return np.nan
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return float(np.mean(np.min(d, axis=1)))


def test_spatial_clustering(raw_data, out, top_n=30, n_perm=2000, seed=0):
    nv_list = raw_data["nv_list"]
    v = out["v1"]
    M = len(nv_list)
    xy = np.array([_get_nv_xy(nv_list[i], i) for i in range(M)], float)

    amp = np.abs(v)
    top = np.argsort(amp)[::-1][:top_n]

    d_obs = nearest_neighbor_stats(xy, top)

    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_perm):
        idx = rng.choice(M, size=top_n, replace=False)
        ds.append(nearest_neighbor_stats(xy, idx))
    ds = np.array(ds)

    p = float(np.mean(ds <= d_obs))  # small distance => clustered
    print(
        f"[spatial] top_n={top_n}, mean NN dist(obs)={d_obs:.3g}, perm p={p:.4f} (small => clustered)"
    )


def _get_orientation(nv):
    for a in ["orientation", "ori", "nv_orientation", "axis", "family"]:
        if hasattr(nv, a):
            try:
                return getattr(nv, a)
            except Exception:
                pass
    return None


def orientation_mode_summary(raw_data, out):
    nv_list = raw_data["nv_list"]
    v = out["v1"]
    amp = np.abs(v)

    oris = [_get_orientation(nv) for nv in nv_list]
    uniq = sorted(set([o for o in oris if o is not None]), key=lambda x: str(x))

    if len(uniq) == 0:
        print("[orientation] Could not find orientation attribute on NV objects.")
        return

    print("[orientation] mean(|v|) and sum(|v|^2) by orientation:")
    for o in uniq:
        idx = [i for i, oo in enumerate(oris) if oo == o]
        a = amp[idx]
        print(
            f"  {o}: n={len(idx):3d}, mean|v|={np.mean(a):.4f}, sum|v|^2={np.sum(a*a):.4f}"
        )


def mode_stability_split(raw_data, analyze_fn, split_run=855, **kwargs):
    """
    Re-run analyze_good_shots on first half / second half and compare eigenvectors.
    analyze_fn should be analyze_good_shots.
    """
    counts = np.asarray(raw_data["counts"])
    _, M, R, S, P = counts.shape
    split_run = min(max(1, split_run), R - 1)

    rd1 = dict(raw_data)
    rd2 = dict(raw_data)
    rd1["counts"] = counts[:, :, :split_run, :, :]
    rd2["counts"] = counts[:, :, split_run:, :, :]

    out1 = analyze_fn(rd1, **kwargs)
    out2 = analyze_fn(rd2, **kwargs)

    v1 = out1["v1"]
    v2 = out2["v1"]

    # phase-align by maximizing inner product
    c = np.vdot(v1, v2)
    v2a = v2 * np.exp(-1j * np.angle(c))

    sim = float(
        np.abs(np.vdot(v1, v2a)) / (np.linalg.norm(v1) * np.linalg.norm(v2a) + 1e-18)
    )
    print(f"[stability] |<v_half1|v_half2>| = {sim:.4f}  (1=stable, ~0=changes)")


def shuffle_null_test(raw_data, out, n_shuffles=50, seed=0):
    """
    Shuffle time within each NV on s_w to destroy cross-NV correlation but keep per-NV marginals.
    Then recompute λmax distribution. Should sit near MP edge.
    """
    s_w = out["s_w"]
    M, N = s_w.shape
    rng = np.random.default_rng(seed)

    lams = []
    for _ in range(n_shuffles):
        s_shuf = s_w.copy()
        for i in range(M):
            perm = rng.permutation(N)
            s_shuf[i] = s_shuf[i, perm]
        C = (s_shuf @ s_shuf.conj().T) / float(N)
        C = 0.5 * (C + C.conj().T)
        evals = np.linalg.eigvalsh(C)
        lams.append(float(np.max(evals)))
    lams = np.array(lams)
    print(
        f"[shuffle-null] λmax: mean={lams.mean():.6f}, std={lams.std():.6f}, min={lams.min():.6f}, max={lams.max():.6f}"
    )


def remove_topk_modes(out, K=1):
    """
    Project out top-K eigenmodes from s_w, then recompute whitened C and λmax.
    """
    s_w = out["s_w"]
    evals = out["evals"]
    evecs = out["evecs"]  # columns are modes (sorted descending in analyze_good_shots)
    M, N = s_w.shape

    Vk = evecs[:, :K]  # (M,K)
    proj = Vk @ (Vk.conj().T @ s_w)  # (M,N)
    s_clean = s_w - proj

    # re-whiten after subtraction
    p = np.mean(np.abs(s_clean) ** 2, axis=1)
    s_clean = s_clean / np.sqrt(np.maximum(p, 1e-12))[:, None]

    C = (s_clean @ s_clean.conj().T) / float(N)
    C = 0.5 * (C + C.conj().T)
    e = np.linalg.eigvalsh(C)
    lam = float(np.max(e))
    frac = lam / float(np.sum(e) + 1e-18)
    print(f"[remove-topK] K={K}: λmax={lam:.6f}, frac={frac:.8f}")
    return {"s_clean": s_clean, "C_clean": C, "lam": lam, "frac": frac}


# ----------------------------
# Suggested calls (after RUN 2)
# ----------------------------
if __name__ == "__main__":
    from utils import data_manager as dm

    raw_data = dm.get_raw_data(
        file_stem="2025_12_24-09_32_29-johnson-nv0_2025_10_21", load_npz=True
    )

    print("=== RUN 1: baseline (detrend only, no regress) ===")
    out0 = analyze_good_shots(raw_data, detrend="runmean", regress_global=False)

    print("\n=== RUN 2: detrend + regress out global mean(I,Q) ===")
    out1 = analyze_good_shots(
        raw_data, detrend="runmean", regress_global=True, regress_use_Q=True
    )

    # 1) spatial map + clustering p-value
    plot_mode_spatial(
        raw_data, out1, top_n=40, title="Mode after regress(gI,gQ): spatial"
    )
    test_spatial_clustering(raw_data, out1, top_n=30, n_perm=2000)

    # 2) orientation check (if attribute exists)
    orientation_mode_summary(raw_data, out1)

    # 3) stability across time (half/half)
    mode_stability_split(
        raw_data,
        analyze_good_shots,
        split_run=855,
        detrend="runmean",
        regress_global=True,
        regress_use_Q=True,
    )

    # 4) shuffle-null sanity check
    shuffle_null_test(raw_data, out1, n_shuffles=30)

    # 5) remove top K modes and see if you go to MP edge
    for K in [1, 2, 3, 5, 8]:
        remove_topk_modes(out1, K=K)
    kpl.show(block=True)

# if __name__ == "__main__":
#     kpl.init_kplotlib()

#     file_stem = "2025_12_24-09_32_29-johnson-nv0_2025_10_21"
#     raw_data = dm.get_raw_data(file_stem=file_stem, load_npz=True)

#     # print("\n=== BEFORE filtering ===")
#     # info_before = summarize_global_bad_runs(raw_data, global_bad_thresh=0.9)
#     # print(f"counts shape: {info_before['counts_shape']}")
#     # print(
#     #     f"global-bad shots: {len(info_before['global_bad_shots'])} ({len(info_before['global_bad_shots'])/info_before['Nshots']:.3%})"
#     # )
#     # if len(info_before.get("bad_runs", [])) > 0:
#     #     br = info_before["bad_run_range"]
#     #     print(
#     #         f"bad runs: {len(info_before['bad_runs'])}  range={br[0]}..{br[1]}  trailing={info_before.get('trailing_block', False)}"
#     #     )

#     # figA = plot_run_health(raw_data, title="Run health (before filtering)")

#     # # Build good-shot dataset
#     raw_good, filt_info = make_goodshot_raw_data(
#         raw_data,
#         global_bad_thresh=0.9,
#         drop_bad_runs=True,
#         drop_remaining_global_bad_shots=True,
#     )

#     # print("\n=== AFTER filtering ===")
#     # info_after = summarize_global_bad_runs(raw_good, global_bad_thresh=0.9)
#     # print(f"counts shape: {info_after['counts_shape']}")
#     # print(
#     #     f"global-bad shots: {len(info_after['global_bad_shots'])} ({len(info_after['global_bad_shots'])/info_after['Nshots']:.3%})"
#     # )
#     # if len(info_after.get("bad_runs", [])) > 0:
#     #     br = info_after["bad_run_range"]
#     #     print(
#     #         f"bad runs: {len(info_after['bad_runs'])}  range={br[0]}..{br[1]}  trailing={info_after.get('trailing_block', False)}"
#     #     )
#     # else:
#     #     print("bad runs: none")

#     # figB = plot_run_health(raw_good, title="Run health (after filtering)")

#     # Per-NV table (sanity: NaN fractions should drop a lot after filtering)
#     # stats = print_dm_lockin_per_nv(
#     #     raw_good, min_pairs=1000, min_var=1e-12, min_valid_frac=0.5
#     # )

#     # # Coherence / covariance plots
#     # figs, proc = process_and_plot_dm_lockin(
#     #     raw_good, min_pairs=1000, min_var=1e-12, min_valid_frac=0.5, show=False
#     # )
#     # proj = project_common_mode(raw_good)
#     # fig_ts, fig_psd = plot_common_mode_timeseries_and_psd(proj["a"], fs_shots=1.0)
#     # res = recheck_after_common_mode_removal(raw_good)

#     # print_mode_spectrum_summary(proj["evals"], M=204, top=8)
#     # print_mode_spectrum_summary(res["evals_res"], M=204, top=8)
#     # rank_k_removal_scan(raw_good, k_list=(1, 2, 3, 5, 8, 12, 20, 30))

#     out = analyze_good_shot_data(
#         file_stem="2025_12_24-09_32_29-johnson-nv0_2025_10_21",
#         detrend="runmean",  # try "none" vs "runmean"
#         do_plots=True,
#         k_list=(1, 2, 3, 5, 8, 12, 20, 30, 50),
#     )
#     z = out["z"]
#     s_dt_mn = out["s_dt_mn"]
#     pow_nv = per_nv_power(s_dt_mn)
#     hi = np.argsort(pow_nv)[::-1][:10]
#     print("Top power NVs:")
#     for i in hi:
#         print(i, pow_nv[i])

#     fracs, lams, prs = iterative_remove_top_modes(z, Kmax=10)
#     for k in range(0, 11):
#         print(f"k={k:2d}: frac={fracs[k]:.6g}, lam={lams[k]:.6g}, PR={prs[k]:.2f}")

#     # Example use after your rank-K print:
#     Ks = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50]
#     fracs = [
#         0.00579392,
#         0.00559656,
#         0.00558203,
#         0.00558384,
#         0.00559589,
#         0.00563252,
#         0.00569423,
#         0.00587641,
#         0.0061555,
#         0.00688837,
#     ]
#     for K, frac, base, d in frac_excess(fracs, 204, Ks):
#         print(f"K={K:2d}: frac={frac:.6g}, base={base:.6g}, excess={d:.6g}")

#     # after you have z and v:
#     a = mode_timeseries(out["z"], out["v"])  # complex
#     a_re = np.real(a)

#     # brightness proxy: mean denominator per shot from *filtered* counts
#     counts = np.asarray(raw_good["counts"], float)  # (4,M,R,S,P)
#     Ip, Im, Qp, Qm = counts
#     den = Ip + Im + Qp + Qm  # (M,R,S,P)
#     bright = den.mean(axis=0).reshape(-1)  # (N,) mean over NVs

#     # correlate
#     corr = np.corrcoef(a_re, bright)[0, 1]
#     print("corr(Re[a], brightness) =", corr)
#     diag2 = diagnose_mode_temporal(raw_good, out, max_lag=5000, show_plots=True)
#     kpl.show(block=True)
