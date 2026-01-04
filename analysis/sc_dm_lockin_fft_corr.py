# dm_lockin_fft_corr_clean.py
# Clean minimal utilities for:
#   (1) build complex lock-in series s_nv[t] = I + iQ from counts
#   (2) trim / drop bad runs + bad shots
#   (3) detrend (per-run mean), regress out global mean(I,Q), whiten
#   (4) correlate across NV index n (time-domain covariance/coherence)
#   (5) FFT + frequency-domain cross-NV coherence in a chosen band
import sys
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Build complex lock-in s = I + iQ
# ----------------------------
def lockin_iq_from_counts(counts, eps=1e-12, return_masks=False):
    """
    counts: array (4, M, R, S, P) with ordering [Ip, Im, Qp, Qm]
    Returns:
      I, Q, s with shape (M, N) where N=R*S*P
      (optional) mI, mQ with shape (M, N)
    """
    counts = np.asarray(counts)
    if counts.dtype == object:
        counts = counts.astype(float)

    assert counts.ndim == 5 and counts.shape[0] == 4, f"counts shape must be (4,M,R,S,P), got {counts.shape}"
    Ip = counts[0].astype(float)
    Im = counts[1].astype(float)
    Qp = counts[2].astype(float)
    Qm = counts[3].astype(float)

    M = Ip.shape[0]
    Ip_f = Ip.reshape(M, -1)
    Im_f = Im.reshape(M, -1)
    Qp_f = Qp.reshape(M, -1)
    Qm_f = Qm.reshape(M, -1)

    denI = Ip_f + Im_f
    denQ = Qp_f + Qm_f

    mI = np.isfinite(Ip_f) & np.isfinite(Im_f) & np.isfinite(denI) & (denI > 0)
    mQ = np.isfinite(Qp_f) & np.isfinite(Qm_f) & np.isfinite(denQ) & (denQ > 0)

    I = np.full_like(denI, np.nan, dtype=float)
    Q = np.full_like(denQ, np.nan, dtype=float)

    I[mI] = (Ip_f[mI] - Im_f[mI]) / (denI[mI] + eps)
    Q[mQ] = (Qp_f[mQ] - Qm_f[mQ]) / (denQ[mQ] + eps)

    s = I + 1j * Q
    if return_masks:
        return I, Q, s, mI, mQ
    return I, Q, s


# ----------------------------
# 2) Bad-run / bad-shot handling
# ----------------------------
def _shot_bad_fraction(counts, eps=1e-12):
    """
    counts: (4,M,R,S,P)
    Returns frac_bad_per_shot: (N,) where N=R*S*P,
    where "bad" means invalid I or invalid Q for that NV+shot.
    """
    _, _, s, mI, mQ = lockin_iq_from_counts(counts, eps=eps, return_masks=True)
    bad = (~mI) | (~mQ)  # (M,N)
    return bad.mean(axis=0)  # (N,)


def find_bad_runs(counts, global_bad_thresh=0.9, eps=1e-12):
    """
    A shot is "global-bad" if >global_bad_thresh fraction of NVs invalid.
    A run is "bad" if >global_bad_thresh fraction of its shots are global-bad.
    Returns bad_run_indices (np.ndarray).
    """
    counts = np.asarray(counts)
    _, M, R, S, P = counts.shape
    frac_bad_shot = _shot_bad_fraction(counts, eps=eps)  # (N,)
    global_bad = (frac_bad_shot > global_bad_thresh).reshape(R, S, P)
    frac_global_bad_per_run = global_bad.mean(axis=(1, 2))  # (R,)
    bad_runs = np.where(frac_global_bad_per_run > global_bad_thresh)[0]
    return bad_runs


def trim_trailing_bad_runs(counts, global_bad_thresh=0.9, eps=1e-12):
    """
    If bad runs form a trailing contiguous block, cut them off.
    Returns (counts_trim, R_keep).
    """
    counts = np.asarray(counts)
    _, _, R, _, _ = counts.shape
    bad_runs = find_bad_runs(counts, global_bad_thresh=global_bad_thresh, eps=eps)
    if bad_runs.size == 0:
        return counts, R
    r0 = int(bad_runs[0])
    trailing = np.all(bad_runs == np.arange(r0, R))
    if trailing:
        return counts[:, :, :r0, :, :], r0
    return counts, R


def drop_bad_runs_anywhere(counts, global_bad_thresh=0.9, eps=1e-12):
    """
    Drops any run classified as bad (not just trailing).
    Preserves run order.
    Returns (counts_keep, keep_runs).
    """
    counts = np.asarray(counts)
    _, _, R, _, _ = counts.shape
    bad_runs = find_bad_runs(counts, global_bad_thresh=global_bad_thresh, eps=eps)
    if bad_runs.size == 0:
        keep = np.arange(R, dtype=int)
        return counts, keep
    keep = np.setdiff1d(np.arange(R), bad_runs)
    return counts[:, :, keep, :, :], keep


def drop_shots_with_any_nan(s_mn):
    """
    Keep only shots where all NVs have finite complex s.
    s_mn: (M,N)
    Returns s_keep, keep_mask (N,)
    """
    s_mn = np.asarray(s_mn)
    keep = np.all(np.isfinite(s_mn), axis=0)
    return s_mn[:, keep], keep


# ----------------------------
# 3) Detrend / regress / whiten
# ----------------------------
def detrend_per_run_mean(x_mn, R, S, P):
    """
    Subtract per-NV, per-run mean over shots within that run.
    x_mn: (M, N), N=R*S*P
    """
    M, N = x_mn.shape
    assert N == R * S * P, "x_mn length must match R*S*P"
    x = x_mn.reshape(M, R, S, P)
    mu = np.mean(x, axis=(2, 3), keepdims=True)  # (M,R,1,1)
    return (x - mu).reshape(M, N)


def regress_out_global_IQ(I_mn, Q_mn, use_Q=True):
    """
    Regress out global mean across NVs per shot:
      gI[t]=mean_n I[n,t], optionally gQ[t]
    Returns (I_res, Q_res, reg_dict)
    """
    I = np.asarray(I_mn, float)
    Q = np.asarray(Q_mn, float)
    M, N = I.shape

    gI = np.mean(I, axis=0)
    gQ = np.mean(Q, axis=0)

    if use_Q:
        X = np.vstack([gI, gQ])  # (K,N)
    else:
        X = np.vstack([gI])      # (1,N)

    X = X - X.mean(axis=1, keepdims=True)
    XXT_inv = np.linalg.pinv(X @ X.T)
    P = XXT_inv @ X  # (K,N)

    betaI = I @ P.T
    betaQ = Q @ P.T

    I_res = I - (betaI @ X)
    Q_res = Q - (betaQ @ X)

    return I_res, Q_res, {"gI": gI, "gQ": gQ, "betaI": betaI, "betaQ": betaQ}


def whiten_per_nv(s_mn, eps=1e-12):
    """
    Mean-remove per NV, then scale so mean power per NV is 1:
      z = (s - mean(s)) / sqrt(mean(|s-mean|^2))
    """
    s = np.asarray(s_mn, np.complex128)
    mu = np.mean(s, axis=1, keepdims=True)
    x = s - mu
    p = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)
    z = x / np.sqrt(np.maximum(p, eps))
    return z


# ----------------------------
# 4) Correlate across NV index n (time-domain)
# ----------------------------
def coherence_matrix_time(z_mn):
    """
    z_mn: (M,N) whitened (diag ~ 1)
    Returns C: (M,M) complex, Hermitian, diag ~ 1
    """
    z = np.asarray(z_mn, np.complex128)
    M, N = z.shape
    C = (z @ z.conj().T) / float(max(N, 1))
    return 0.5 * (C + C.conj().T)


def eig_hermitian(C):
    """
    Returns evals (descending), evecs columns aligned to evals.
    """
    C = 0.5 * (C + C.conj().T)
    w, V = np.linalg.eigh(C)  # ascending
    order = np.argsort(np.real(w))[::-1]
    return np.real(w[order]), V[:, order]


def common_mode_timeseries(z_mn, v_m):
    """
    a[t] = v^H z[:,t]
    """
    return (v_m.conj().T @ z_mn).astype(np.complex128)


# ----------------------------
# 5) FFT + frequency-domain coherence across NVs
# ----------------------------
def fft_complex(x, dt=1.0, window="hann"):
    """
    x: (N,) complex or real time series
    dt: sampling interval (seconds). If unknown, dt=1 => units are cycles/shot.
    window: 'hann' or None
    Returns (f_pos, X_pos) for positive frequencies only.
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    if window == "hann":
        w = np.hanning(N)
        xw = x * w
    else:
        w = None
        xw = x

    X = np.fft.fft(xw)
    f = np.fft.fftfreq(N, d=dt)

    pos = f >= 0
    f_pos = f[pos]
    X_pos = X[pos]
    return f_pos, X_pos


def fft_all_nvs(z_mn, dt=1.0, window="hann"):
    """
    z_mn: (M,N) complex
    Returns:
      f_pos: (F,)
      Zf: (M,F) complex FFT values (positive freqs)
    """
    z = np.asarray(z_mn, np.complex128)
    M, N = z.shape
    if window == "hann":
        w = np.hanning(N)[None, :]
        zw = z * w
    else:
        zw = z

    Z = np.fft.fft(zw, axis=1)
    f = np.fft.fftfreq(N, d=dt)
    pos = f >= 0
    return f[pos], Z[:, pos]


def spectral_covariance(Zf_mf, f, fmin=None, fmax=None):
    """
    Build a cross-NV covariance matrix by averaging over frequency bins in [fmin,fmax].
    Zf_mf: (M,F) complex FFT values
    f: (F,) positive frequencies
    Returns Cspec: (M,M) complex (Hermitian)
    """
    Zf = np.asarray(Zf_mf, np.complex128)
    f = np.asarray(f, float)

    sel = np.ones_like(f, dtype=bool)
    if fmin is not None:
        sel &= (f >= float(fmin))
    if fmax is not None:
        sel &= (f <= float(fmax))

    Zb = Zf[:, sel]
    if Zb.shape[1] < 1:
        raise ValueError("No FFT bins selected in the requested band.")

    C = (Zb @ Zb.conj().T) / float(Zb.shape[1])
    return 0.5 * (C + C.conj().T)


def coherence_from_cov(C, eps=1e-18):
    """
    Normalize C -> coh_ij = C_ij / sqrt(C_ii C_jj)
    """
    C = np.asarray(C, np.complex128)
    p = np.real(np.diag(C))
    p = np.maximum(p, eps)
    denom = np.sqrt(p[:, None] * p[None, :])
    return C / denom


# ----------------------------
# One-shot pipeline (minimal)
# ----------------------------
def analyze_counts_fft_corr(
    counts,
    dt=1.0,
    global_bad_thresh=0.9,
    trim_trailing=True,
    drop_all_bad_runs=False,
    detrend=True,
    regress_global=True,
    regress_use_Q=True,
    fband=None,  # (fmin,fmax) or None
    eps=1e-12,
):
    """
    counts: (4,M,R,S,P) with ordering [Ip, Im, Qp, Qm]
    dt: seconds per shot (or dt=1 for cycles/shot)
    fband: None or tuple (fmin,fmax) for spectral coherence matrix

    Returns dict with:
      - z: (M,N) whitened complex lock-in series
      - C_time: (M,M) time-domain coherence/covariance across NV index
      - evals_time, v1_time, a_time: eigendecomp + top-mode timeseries
      - f_pos, Zf: per-NV FFT
      - optional: C_spec, coh_spec, evals_spec, v1_spec (band-averaged)
      - diagnostics: bad-run stats, keep masks, regress info, intermediates
    """
    counts = np.asarray(counts)

    # ----------------------------
    # A) Pre-diagnostics on raw counts (before trimming/dropping)
    # ----------------------------
    frac_bad_shot0 = _shot_bad_fraction(counts, eps=eps)  # (N_total,) where N_total=R*S*P
    _, M0, R0, S0, P0 = counts.shape
    global_bad0 = (frac_bad_shot0 > global_bad_thresh).reshape(R0, S0, P0)
    frac_global_bad_per_run0 = global_bad0.mean(axis=(1, 2))  # (R0,)

    # ----------------------------
    # B) Optionally trim/drop runs
    # ----------------------------
    keep_runs = None
    if trim_trailing:
        counts, R_keep = trim_trailing_bad_runs(counts, global_bad_thresh=global_bad_thresh, eps=eps)
        # if we trimmed trailing runs, keep_runs is the prefix
        keep_runs = np.arange(R_keep, dtype=int)

    if drop_all_bad_runs:
        counts, keep_runs2 = drop_bad_runs_anywhere(counts, global_bad_thresh=global_bad_thresh, eps=eps)
        keep_runs = keep_runs2  # overrides (or refines) keep_runs

    # updated dims after run filtering
    _, M, R, S, P = counts.shape

    # ----------------------------
    # C) Build I0,Q0 on the FULL (R,S,P) grid, then detrend/regress,
    #    then drop bad shots. (This avoids reshape issues.)
    # ----------------------------
    I0, Q0, _, mI, mQ = lockin_iq_from_counts(counts, eps=eps, return_masks=True)  # (M, R*S*P)

    # Detrend per-run mean (requires exact R,S,P structure)
    if detrend:
        I0 = detrend_per_run_mean(I0, R, S, P)
        Q0 = detrend_per_run_mean(Q0, R, S, P)

    # Regress out global mean across NVs per shot
    if regress_global:
        I0, Q0, reg = regress_out_global_IQ(I0, Q0, use_Q=regress_use_Q)
    else:
        reg = None

    # Complex series
    s0_full = I0 + 1j * Q0

    # Drop any shots that have any NaN across NVs
    s0, keep_shots = drop_shots_with_any_nan(s0_full)  # keep_shots length = R*S*P (after run dropping)
    # For convenience, also keep the post-processed I0,Q0 after shot dropping
    I0_keep = np.real(s0)
    Q0_keep = np.imag(s0)

    # ----------------------------
    # D) Whiten per NV
    # ----------------------------
    z = whiten_per_nv(s0, eps=eps)  # (M, Nshots)

    # ----------------------------
    # E) Time-domain coherence across NV index
    # ----------------------------
    C_time = coherence_matrix_time(z)
    evals_t, evecs_t = eig_hermitian(C_time)
    v1_t = evecs_t[:, 0]
    a_t = common_mode_timeseries(z, v1_t)

    # ----------------------------
    # F) FFT per NV (positive freqs)
    # ----------------------------
    f_pos, Zf = fft_all_nvs(z, dt=dt, window="hann")

    # ----------------------------
    # G) Optional spectral band coherence
    # ----------------------------
    out = {
        # shapes / bookkeeping
        "counts_used_shape": counts.shape,
        "M": M, "R": R, "S": S, "P": P,
        "Nshots": z.shape[1],
        "dt": dt,

        # keep masks
        "keep_runs": keep_runs,       # None or array of kept run indices (relative to original)
        "keep_shots": keep_shots,     # boolean mask relative to flattened shots of kept runs

        # raw diagnostics (before trimming/dropping)
        "frac_bad_shot0": frac_bad_shot0,
        "frac_global_bad_per_run0": frac_global_bad_per_run0,

        # intermediates (post-detrend/regress; BEFORE whitening)
        "I0_full": I0,                # (M, R*S*P) after run filtering, before shot drop
        "Q0_full": Q0,
        "s0_full": s0_full,
        "I0": I0_keep,                # (M, Nshots) after shot drop
        "Q0": Q0_keep,
        "s0": s0,                     # (M, Nshots)
        "reg": reg,

        # main outputs
        "z": z,
        "C_time": C_time,
        "evals_time": evals_t,
        "v1_time": v1_t,
        "a_time": a_t,

        # FFT outputs
        "f_pos": f_pos,
        "Zf": Zf,
    }

    if fband is not None:
        fmin, fmax = fband
        C_spec = spectral_covariance(Zf, f_pos, fmin=fmin, fmax=fmax)
        coh_spec = coherence_from_cov(C_spec)
        evals_s, evecs_s = eig_hermitian(coh_spec)
        out.update({
            "fband": (fmin, fmax),
            "C_spec": C_spec,
            "coh_spec": coh_spec,
            "evals_spec": evals_s,
            "v1_spec": evecs_s[:, 0],
        })

    return out


def plot_bad_run_stats(out, global_bad_thresh=0.9):
    frac_run = out.get("frac_global_bad_per_run0", None)
    if frac_run is None:
        return
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(frac_run, "o-")
    ax.axhline(global_bad_thresh, ls="--")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Frac(global-bad shots)")
    ax.set_title("Bad-run diagnostic")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

def plot_lockin_hist(out, max_points=200000):
    I0 = out.get("I0", None)
    Q0 = out.get("Q0", None)
    if I0 is None or Q0 is None:
        return

    # downsample for speed
    I = I0.ravel()
    Q = Q0.ravel()
    m = np.isfinite(I) & np.isfinite(Q)
    I = I[m]; Q = Q[m]
    if I.size > max_points:
        idx = np.random.choice(I.size, size=max_points, replace=False)
        I = I[idx]; Q = Q[idx]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(I, Q, ".", ms=1, alpha=0.3)
    ax.set_xlabel("I (after detrend/regress)")
    ax.set_ylabel("Q (after detrend/regress)")
    ax.set_title("IQ cloud (sanity check)")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

def plot_example_timeseries(out, nv_indices=(0), nshots=6000):
    z = out["z"]
    N = z.shape[1]
    nshots = min(nshots, N)
    t = np.arange(nshots)

    fig, ax = plt.subplots(figsize=(7,4))
    for i in nv_indices:
        if i < 0 or i >= z.shape[0]:
            continue
        ax.plot(t, np.real(z[i,:nshots]), label=f"NV {i} Re(z)")
    ax.set_xlabel("Shot index")
    ax.set_ylabel("Re(z) (whitened)")
    ax.set_title("Example whitened traces")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()



def make_orientation_permutation(M, ORI_11m1, ORI_m111, keep_rest=True):
    """
    Returns:
      perm: indices length M (new order)
      groups: dict with group slices for plotting boundaries
    """
    ORI_11m1 = [i for i in ORI_11m1 if 0 <= i < M]
    ORI_m111 = [i for i in ORI_m111 if 0 <= i < M]

    # remove overlaps (just in case)
    set11 = set(ORI_11m1)
    setm1 = [i for i in ORI_m111 if i not in set11]

    perm = ORI_11m1 + setm1

    if keep_rest:
        rest = [i for i in range(M) if i not in set(perm)]
        perm = perm + rest

    perm = np.array(perm, dtype=int)

    n1 = len(ORI_11m1)
    n2 = len(setm1)
    groups = {
        "ORI_11m1": (0, n1),
        "ORI_m111": (n1, n1+n2),
        "rest": (n1+n2, len(perm))
    }
    return perm, groups


def plot_matrix_with_boundaries(C, perm, groups, title="C", abs_phase="abs",
                                prc=(1,99), diag_color="0.85"):
    """
    abs_phase: "abs" or "phase"
    """
    C = np.asarray(C)
    Cp = C[np.ix_(perm, perm)]

    # choose data to show
    if abs_phase == "abs":
        A = np.abs(Cp)
        ttl = title + " |abs|"
    else:
        A = np.angle(Cp)
        ttl = title + " phase"

    # mask diagonal (display gray) + off-diag autoscale
    mask = np.eye(A.shape[0], dtype=bool)
    Am = np.ma.array(A, mask=mask)

    # off-diagonal limits
    A_tmp = np.array(A, float, copy=True)
    np.fill_diagonal(A_tmp, np.nan)
    vmin, vmax = np.nanpercentile(A_tmp, prc)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(diag_color)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(Am, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(ttl + " (sorted by orientation)")
    ax.set_xlabel("NV index (sorted)")
    ax.set_ylabel("NV index (sorted)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # draw boundaries + labels
    for name, (a, b) in groups.items():
        if name == "rest" and (b - a) == 0:
            continue
        if a > 0:
            ax.axhline(a - 0.5, color="w", lw=1.2)
            ax.axvline(a - 0.5, color="w", lw=1.2)

    # label group centers on axes
    for name, (a, b) in groups.items():
        if (b - a) <= 0:
            continue
        c = 0.5 * (a + b - 1)
        ax.text(-0.02, c, name, va="center", ha="right", transform=ax.get_yaxis_transform())
        ax.text(c, -0.03, name, va="top", ha="center", transform=ax.get_xaxis_transform())

    plt.tight_layout()
    return fig, ax, Cp

def _offdiag_limits(A, prc=(1, 99)):
    A = np.array(A, float, copy=True)
    np.fill_diagonal(A, np.nan)
    vmin, vmax = np.nanpercentile(A, prc)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(A), np.nanmax(A)
    return vmin, vmax

def plot_matrix(
    C,
    title="Matrix",
    show_phase=True,
    prc=(1, 99),
    diag_color="lightgray",   # <- diagonal displayed as this color
):
    C = np.asarray(C)

    def _imshow_with_gray_diag(ax, A, vmin=None, vmax=None):
        A = np.asarray(A)
        mask = np.eye(A.shape[0], dtype=bool)
        Am = np.ma.array(A, mask=mask)  # mask diagonal

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(diag_color)        # masked values -> gray

        im = ax.imshow(Am, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        return im

    # --- |C| ---
    A = np.abs(C)
    vmin, vmax = _offdiag_limits(A, prc=prc)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = _imshow_with_gray_diag(ax, A, vmin=vmin, vmax=vmax)
    ax.set_title(title + " |abs| (diag gray, off-diag autoscale)")
    ax.set_xlabel("NV index")
    ax.set_ylabel("NV index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # --- phase ---
    if show_phase:
        P = np.angle(C)
        vmin, vmax = _offdiag_limits(P, prc=prc)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = _imshow_with_gray_diag(ax, P, vmin=vmin, vmax=vmax)
        ax.set_title(title + " phase (diag gray, off-diag autoscale)")
        ax.set_xlabel("NV index")
        ax.set_ylabel("NV index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

def plot_eigs(evals, title="Eigenvalues"):
    evals = np.asarray(evals, float)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(evals, "o-")
    ax.set_xlabel("mode index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

def plot_common_mode(out, dt=1.0, fmax=None):
    a = out["a_time"]
    N = a.size
    t = np.arange(N) * dt

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(t, np.real(a), lw=1, label="Re(a)")
    ax.plot(t, np.imag(a), lw=1, label="Im(a)")
    ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
    ax.set_ylabel("a(t)")
    ax.set_title("Top-mode time series a(t)")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend()
    plt.tight_layout()

    # spectrum of a(t)
    A = np.fft.rfft(a * np.hanning(N))
    f = np.fft.rfftfreq(N, d=dt)
    psd = np.abs(A)**2

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(f, psd)
    ax.set_xlabel("frequency (Hz)" if dt != 1.0 else "cycles/shot")
    ax.set_ylabel("|A(f)|^2")
    ax.set_title("Spectrum of a(t)")
    ax.grid(True, ls="--", lw=0.5)
    if fmax is not None:
        ax.set_xlim(0, fmax)
    plt.tight_layout()

def plot_nv_spectrum_median(out, fmax=None):
    f = out["f_pos"]
    Zf = out["Zf"]
    psd_med = np.median(np.abs(Zf)**2, axis=0)

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(f, psd_med)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("median |Z(f)|^2 across NVs")
    ax.set_title("Median per-NV spectrum")
    ax.grid(True, ls="--", lw=0.5)
    if fmax is not None:
        ax.set_xlim(0, fmax)
    plt.tight_layout()

def make_all_plots(out, dt=1.0, global_bad_thresh=0.9, fmax=None):
    plot_bad_run_stats(out, global_bad_thresh=global_bad_thresh)
    plot_lockin_hist(out)
    plot_example_timeseries(out, nv_indices=(0,1,2,3), nshots=600)

    plot_matrix(out["C_time"], title="Time-domain coherence (NV×NV)", show_phase=False)
    plot_eigs(out["evals_time"], title="Eigenvalues of time-domain coherence")

    plot_common_mode(out, dt=dt, fmax=fmax)
    plot_nv_spectrum_median(out, fmax=fmax)

    if "coh_spec" in out:
        plot_matrix(out["coh_spec"], title="Spectral coherence (band-averaged)", show_phase=False)
        plot_eigs(out["evals_spec"], title="Eigenvalues of spectral coherence")


import numpy as np

def block_contrast_abs(C, g1, g2):
    """Contrast = mean(|C| within groups) - mean(|C| cross), excluding diagonal."""
    C = np.asarray(C)
    idx1 = np.array(g1, int); idx2 = np.array(g2, int)

    C11 = np.abs(C[np.ix_(idx1, idx1)])
    C22 = np.abs(C[np.ix_(idx2, idx2)])
    C12 = np.abs(C[np.ix_(idx1, idx2)])

    # drop diagonal for within blocks
    np.fill_diagonal(C11, np.nan)
    np.fill_diagonal(C22, np.nan)

    within = np.nanmean(np.r_[C11.ravel(), C22.ravel()])
    cross  = np.nanmean(C12.ravel())
    return float(within - cross), float(within), float(cross)

def perm_test_orientation(C, ORI_11m1, ORI_m111, nperm=2000, seed=0):
    rng = np.random.default_rng(seed)
    C = np.asarray(C)
    M = C.shape[0]

    # make clean groups with in-range indices
    g1 = [i for i in ORI_11m1 if 0 <= i < M]
    g2 = [i for i in ORI_m111 if 0 <= i < M]
    n1, n2 = len(g1), len(g2)

    obs, within, cross = block_contrast_abs(C, g1, g2)

    all_idx = np.arange(M)
    stats = np.empty(nperm, float)
    for k in range(nperm):
        perm = rng.permutation(all_idx)
        g1p = perm[:n1]
        g2p = perm[n1:n1+n2]
        stats[k], _, _ = block_contrast_abs(C, g1p, g2p)

    # one-sided p-value: how often perm >= observed
    p = (np.sum(stats >= obs) + 1) / (nperm + 1)
    return {"obs": obs, "within": within, "cross": cross, "p": p, "null": stats}

# Example:import numpy as np

def shuffle_null_eigs(z_mn, nperm=300, seed=1, mode="time_shuffle"):
    """
    Build a null distribution for the top eigenvalue of the NV×NV coherence matrix.

    Parameters
    ----------
    z_mn : (M,N) complex
        Whitened complex series (output out["z"]).
    nperm : int
        Number of permutations.
    seed : int
        RNG seed.
    mode : str
        "time_shuffle"  : independently permute time samples within each NV (destroys cross-NV corr)
        "phase_random"  : randomize phases in Fourier domain per NV, preserve PSD, destroy cross-NV corr

    Returns
    -------
    emax : (nperm,) float
        Null distribution of largest eigenvalue.
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(z_mn, np.complex128)
    M, N = z.shape

    emax = np.empty(nperm, dtype=float)

    for k in range(nperm):
        if mode == "time_shuffle":
            zs = z.copy()
            for i in range(M):
                rng.shuffle(zs[i])  # in-place along time axis
        elif mode == "phase_random":
            # preserves per-NV power spectrum, removes cross-NV coherence
            zs = np.empty_like(z)
            for i in range(M):
                X = np.fft.rfft(z[i])
                phase = rng.uniform(0, 2*np.pi, size=X.shape)
                Xr = np.abs(X) * np.exp(1j*phase)
                zs[i] = np.fft.irfft(Xr, n=N)
        else:
            raise ValueError("mode must be 'time_shuffle' or 'phase_random'")

        C = coherence_matrix_time(zs)
        w, _ = eig_hermitian(C)
        emax[k] = float(w[0])

    return emax


# def extract_xy_from_nv_list(nv_list, which="pixel"):
#     """
#     which: "pixel" (coords['pixel']) or "sample" (coords['sample']) or "laser_INTE_520_aod" etc.
#     Returns xy: (M,2) float, mask_valid: (M,) bool
#     """
#     M = len(nv_list)
#     xy = np.full((M, 2), np.nan, float)

#     for i, nv in enumerate(nv_list):
#         c = getattr(nv, "coords", None)
#         if c is None:
#             continue
#         if which in c and c[which] is not None and len(c[which]) >= 2:
#             xy[i] = np.array(c[which][:2], float)

#     valid = np.all(np.isfinite(xy), axis=1)
#     return xy, valid

def correlated_participation_score(C, K=6):
    """
    Score_i = sum_{k=1..K} |v_k(i)|^2 where v_k are eigenvectors of C.
    """
    w, V = eig_hermitian(C)  # V[:,k]
    K = min(K, V.shape[1])
    score = np.sum(np.abs(V[:, :K])**2, axis=1)
    return score, w[:K], V[:, :K]

def plot_nv_map_scores(xy, score, valid=None, top_n=40, title="NV map (score)", invert_y=True):
    xy = np.asarray(xy)
    score = np.asarray(score)
    if valid is None:
        valid = np.all(np.isfinite(xy), axis=1)

    idx_valid = np.where(valid)[0]
    s_valid = score[idx_valid]
    order = idx_valid[np.argsort(s_valid)[::-1]]  # descending score
    top = order[:min(top_n, len(order))]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xy[valid,0], xy[valid,1], s=12, alpha=0.25)
    ax.scatter(xy[top,0], xy[top,1], s=45, alpha=0.9)

    for k, i in enumerate(top[:15]):  # label first ~15 to avoid clutter
        ax.text(xy[i,0], xy[i,1], str(i), fontsize=8)

    ax.set_title(title + f" | top {len(top)} highlighted")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, ls="--", lw=0.5)
    ax.axis("equal")
    if invert_y:
        ax.invert_yaxis()  # pixel coords often have y downward
    plt.tight_layout()
    return fig, ax, top


def corr_abs_vs_distance(C, xy, valid=None, nbins=25, subset=None):
    """
    Uses |C_ij| for i<j. Excludes diagonal.
    If subset is provided (list of indices), only uses those indices.
    Returns bin_centers, mean_absC
    """
    C = np.asarray(C)
    xy = np.asarray(xy)
    M = C.shape[0]
    if valid is None:
        valid = np.all(np.isfinite(xy), axis=1)

    idx = np.where(valid)[0]
    if subset is not None:
        subset = np.array([i for i in subset if i in set(idx)], dtype=int)
        idx = subset

    d_list, c_list = [], []
    for a in range(len(idx)):
        i = idx[a]
        for b in range(a+1, len(idx)):
            j = idx[b]
            d = np.linalg.norm(xy[i] - xy[j])
            d_list.append(d)
            c_list.append(np.abs(C[i,j]))

    d_list = np.array(d_list)
    c_list = np.array(c_list)

    bins = np.linspace(d_list.min(), d_list.max(), nbins+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    m = np.full(nbins, np.nan)
    for k in range(nbins):
        sel = (d_list >= bins[k]) & (d_list < bins[k+1])
        if np.any(sel):
            m[k] = np.mean(c_list[sel])
    return centers, m

def plot_corr_vs_r(r, m, title="|C| vs distance"):
    plt.figure(figsize=(6,3.2))
    plt.plot(r, m, "o-")
    plt.xlabel("distance (same units as xy)")
    plt.ylabel("mean |C_ij|")
    plt.title(title)
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


def extract_xy_from_nv_list(nv_list, key="pixel"):
    M = len(nv_list)
    xy = np.full((M,2), np.nan, float)
    for i, nv in enumerate(nv_list):
        c = getattr(nv, "coords", None)
        if c is None: 
            continue
        v = c.get(key, None)
        if v is not None and len(v) >= 2:
            xy[i] = np.array(v[:2], float)
    valid = np.all(np.isfinite(xy), axis=1)
    return xy, valid

def plot_highlight(xy, highlight, title="", invert_y=True, label_first=20):
    xy = np.asarray(xy)
    highlight = np.asarray(highlight, int)

    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.scatter(xy[:,0], xy[:,1], s=10, alpha=0.2)
    ax.scatter(xy[highlight,0], xy[highlight,1], s=60, alpha=0.95)
    for i in highlight[:label_first]:
        ax.text(xy[i,0], xy[i,1], str(i), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.axis("equal"); ax.grid(True, ls="--", lw=0.5)
    if invert_y: ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def cluster_stat_radius(xy, idx):
    """Radius of gyration of selected points."""
    pts = xy[np.asarray(idx, int)]
    ctr = np.mean(pts, axis=0)
    return float(np.sqrt(np.mean(np.sum((pts-ctr)**2, axis=1))))

def perm_test_spatial_cluster(xy, idx, nperm=5000, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, float)
    idx = np.asarray(idx, int)
    n = len(idx)

    obs = cluster_stat_radius(xy, idx)
    all_idx = np.arange(xy.shape[0])
    null = np.empty(nperm, float)
    for k in range(nperm):
        pick = rng.choice(all_idx, size=n, replace=False)
        null[k] = cluster_stat_radius(xy, pick)

    p = (np.sum(null <= obs) + 1) / (nperm + 1)   # smaller radius = more clustered
    return obs, null, p


def close_pairs(xy, idx, rmax=8.0):
    """
    Return list of (i,j,dist) among idx with dist <= rmax.
    rmax in same units as xy (pixels if xy is pixel coords).
    """
    idx = np.asarray(idx, int)
    pts = xy[idx]
    pairs = []
    for a in range(len(idx)):
        for b in range(a+1, len(idx)):
            d = float(np.linalg.norm(pts[a]-pts[b]))
            if d <= rmax:
                pairs.append((int(idx[a]), int(idx[b]), d))
    pairs.sort(key=lambda x: x[2])
    return pairs


# ----------------------------
# Example usage (edit for your loader)
# ----------------------------
if __name__ == "__main__":
    # Example: load from an npz that contains 'counts'
    from utils import data_manager as dm
    raw_data = dm.get_raw_data(
        file_stem="2025_12_24-09_32_29-johnson-nv0_2025_10_21", load_npz=True
    )
    nv_list =  raw_data["nv_list"] 
    counts = raw_data["counts"]
    
    out = analyze_counts_fft_corr(
    counts,
    dt=0.06,                 # set to real seconds/shot if you know it
    global_bad_thresh=0.9,
    trim_trailing=True,
    drop_all_bad_runs=False,
    detrend=True,
    regress_global=True,
    regress_use_Q=True,
    fband=(0.05, 0.15),      # example band if dt=1.0 cycles/shot; otherwise Hz
)


    # ORI_11m1 = [...]   # your list
    # ORI_m111 = [...]   # your list
    # #fmt:off 
    ORI_11m1 = [0, 1, 3, 5, 6, 7, 9, 10, 13, 18, 19, 21, 24, 25, 27, 28, 30, 32, 34, 36, 40, 41, 43, 44, 46, 48, 49, 51, 52, 53, 56, 57, 64, 65, 66, 67, 68, 69, 73, 75, 77, 80, 82, 84, 86, 88, 91, 98, 100, 101, 102, 103, 106, 107, 109, 110, 111, 113, 115, 116, 118, 119, 120, 121, 123, 124, 127, 129, 130, 131, 132, 133, 134, 135, 141, 142, 146, 149, 150, 152, 153, 156, 157, 158, 162, 163, 165, 167, 168, 171, 174, 177, 179, 184, 185, 186, 187, 189, 190, 191, 192, 193, 195, 198, 201, 203]
    ORI_m111 = [2, 4, 8, 11, 12, 14, 15, 16, 17, 20, 22, 23, 26, 29, 31, 33, 35, 37, 38, 39, 42, 45, 47, 50, 54, 55, 58, 59, 60, 61, 62, 63, 70, 71, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 90, 92, 93, 94, 95, 96, 97, 99, 104, 105, 108, 112, 114, 117, 122, 125, 126, 128, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 151, 154, 155, 159, 160, 161, 164, 166, 169, 170, 172, 173, 175, 176, 178, 180, 181, 182, 183, 188, 194, 196, 197, 199, 200, 202] 
    # # #fmt:on
    # M = out["M"]
    # perm, groups = make_orientation_permutation(M, ORI_11m1, ORI_m111, keep_rest=True)

    # # time-domain matrix
    # plot_matrix_with_boundaries(out["C_time"], perm, groups, title="C_time", abs_phase="abs")
    # plot_matrix_with_boundaries(out["C_time"], perm, groups, title="C_time", abs_phase="phase")

    # # spectral matrix (if you computed fband)
    # if "coh_spec" in out:
    #     plot_matrix_with_boundaries(out["coh_spec"], perm, groups, title="coh_spec", abs_phase="abs", prc=(1,99))


    # C = out["C_time"]  # or out["coh_spec"]
    # res = perm_test_orientation(C, ORI_11m1, ORI_m111, nperm=2000)
    # print("Contrast (within-cross) =", res["obs"])
    # print("within =", res["within"], "cross =", res["cross"])
    # print("Permutation p-value =", res["p"])
    
    # plt.figure()
    # plt.hist(res["null"], bins=40)
    # plt.axvline(res["obs"], ls="--")
    # plt.xlabel("within - cross (|C|)")
    # plt.ylabel("count")
    # plt.title(f"Permutation test, p={res['p']:.3g}")

    C = out["C_time"]
    diag = np.real(np.diag(C))
    print("diag mean±std:", diag.mean(), diag.std(), "min/max:", diag.min(), diag.max())
    v1 = out["v1_time"]
    w = np.abs(v1)
    w = w / np.linalg.norm(w)

    PR = 1.0 / np.sum(w**4)          # participation ratio ~ #effective NVs
    M = out["M"]
    print("Participation ratio:", PR, "out of M =", M)
    plt.figure(figsize=(7,3))
    plt.plot(np.abs(out["v1_time"]), "o", ms=3)
    plt.xlabel("NV index"); plt.ylabel("|v1|"); plt.title("Top-mode weights |v1|")
    plt.grid(True, ls="--", lw=0.5); plt.tight_layout(); plt.show()
    
    z = out["z"]                          # (M, Nshots)
    shot_energy = np.mean(np.abs(z)**2, axis=0)  # should be ~1 after whitening

    plt.figure(figsize=(7,3))
    plt.plot(shot_energy, lw=1)
    plt.axhline(np.median(shot_energy), ls="--")
    plt.xlabel("shot"); plt.ylabel("mean |z|^2 across NVs")
    plt.title("Shot energy (find global glitches)")
    plt.grid(True, ls="--", lw=0.5); plt.tight_layout(); plt.show()

    print("energy percentiles:", np.percentile(shot_energy, [50, 90, 95, 99, 99.5, 99.9]))
    
    M, R, S, P = out["M"], out["R"], out["S"], out["P"]
    z = out["z"]
    shot_energy = np.mean(np.abs(z)**2, axis=0)

    # reshape into runs if possible
    if shot_energy.size == R*S*P:
        E = shot_energy.reshape(R, S*P)           # (R, shots_per_run)
        E_run = np.mean(E, axis=1)               # mean energy per run

        plt.figure(figsize=(7,3))
        plt.plot(E_run, "o-", ms=3)
        plt.xlabel("run index")
        plt.ylabel("mean shot_energy in run")
        plt.title("Run-averaged shot energy")
        plt.grid(True, ls="--", lw=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print("Cannot reshape by runs because shots were dropped (NaNs).")
    
    top = np.array([107,186,98,51,189,14,201,50,199,8,202,59,21,45,57,73,108,155,114,97])
    M = out["M"]
    mask_top = np.zeros(M, bool); mask_top[top] = True

    z = out["z"]
    E_all  = np.mean(np.abs(z)**2, axis=0)
    E_top  = np.mean(np.abs(z[mask_top])**2, axis=0)
    E_rest = np.mean(np.abs(z[~mask_top])**2, axis=0)

    plt.figure(figsize=(8,3))
    plt.plot(E_all,  lw=1, label="all NVs")
    plt.plot(E_top,  lw=1, label="top subset")
    plt.plot(E_rest, lw=1, label="rest")
    plt.xlabel("shot")
    plt.ylabel("mean |z|^2")
    plt.title("Shot energy by group")
    plt.grid(True, ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


    def per_shot_normalize(z, eps=1e-12):
        E = np.mean(np.abs(z)**2, axis=0)
        return z / np.sqrt(np.maximum(E, eps))[None, :]

    z = out["z"]
    C0 = out["C_time"]
    w0, _ = eig_hermitian(C0)

    zN = per_shot_normalize(z)
    CN = coherence_matrix_time(zN)
    wN, _ = eig_hermitian(CN)

    print("Top evals before:", w0[:6])
    print("Top evals after per-shot norm:", wN[:6])


    p = [0.01, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    print("shot_energy percentiles:")
    for q, v in zip(p, np.percentile(shot_energy, p)):
        print(f"{q:>5}%  {v:.4f}")

    print("min / max:", float(np.min(shot_energy)), float(np.max(shot_energy)))


    E = shot_energy
    w = 200  # smoothing window (shots)
    E_s = np.convolve(E, np.ones(w)/w, mode="same")

    thr = 0.8  # adjust
    bad = E_s < thr

    plt.figure(figsize=(8,3))
    plt.plot(E, lw=0.5, alpha=0.4, label="shot_energy")
    plt.plot(E_s, lw=2, label=f"smoothed ({w})")
    plt.axhline(thr, ls="--")
    plt.legend()
    plt.xlabel("shot")
    plt.ylabel("mean |z|^2")
    plt.title("Detect low-energy regime")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    print("fraction below thr:", bad.mean())
    sys.exit()
    def top_eig_from_z(z):
        C = coherence_matrix_time(z)
        w, V = eig_hermitian(C)
        return float(w[0]), w, V[:,0]

    for prc in [99.0, 99.5, 99.9]:
        thr = np.percentile(shot_energy, prc)
        keep = shot_energy <= thr
        lam1, evals, v1 = top_eig_from_z(z[:, keep])
        print(f"drop top {(100-prc):.1f}% energy shots -> keep {keep.mean()*100:.2f}% shots, lam1={lam1:.6f}")

    z = out["z"]
    v1 = out["v1_time"]
    a  = out["a_time"]

    z_perp = z - np.outer(v1, a)      # remove rank-1 component

    lam1_before = out["evals_time"][0]
    lam1_after, evals_after, _ = top_eig_from_z(z_perp)

    print("lam1 before:", lam1_before)
    print("lam1 after removing mode1:", lam1_after)


    def top_mode_loadings(out, K=5):
        V = np.column_stack([out["v1_time"]])  # placeholder if you only stored v1
        # Better: recompute eigenvectors once from C_time so you have top K:
        w, Vfull = eig_hermitian(out["C_time"])
        return w[:K], Vfull[:, :K]

    evalsK, VK = top_mode_loadings(out, K=6)
    print("Top evals:", evalsK)

    # NV importance score: sum of squared magnitudes across first K modes
    score = np.sum(np.abs(VK)**2, axis=1)  # (M,)
    idx = np.argsort(score)[::-1]

    print("Top 20 NVs by correlated-mode score:\n", idx[:20])
    print("Scores:", score[idx[:20]])
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,3))
    plt.plot(score, "o", ms=3)
    plt.xlabel("NV index"); plt.ylabel("sum_k |v_k|^2")
    plt.title("NV participation in top correlated modes")
    plt.grid(True, ls="--", lw=0.5); plt.tight_layout(); plt.show()

    order = np.argsort(score)[::-1]
    Cp = out["C_time"][np.ix_(order, order)]
    plot_matrix(Cp, title="C_time sorted by correlated-mode score", show_phase=False)


    xy_pix, valid = extract_xy_from_nv_list(nv_list, key="pixel")
    score, evalsK, VK = correlated_participation_score(out["C_time"], K=6)
    fig, ax, top = plot_nv_map_scores(xy_pix, score, valid=valid, top_n=40,
                                    title="Pixel map: participation score")
    print("Top indices:", top[:20])


    r_all, m_all = corr_abs_vs_distance(out["C_time"], xy_pix, valid=valid, nbins=25)
    plot_corr_vs_r(r_all, m_all, title="C_time: mean |C_ij| vs pixel distance (all NVs)")

    # focus only on the strongly participating subset (say top 40 by score)
    top40 = np.argsort(score)[::-1][:40]
    r_top, m_top = corr_abs_vs_distance(out["C_time"], xy_pix, valid=valid, nbins=15, subset=top40)
    plot_corr_vs_r(r_top, m_top, title="C_time: mean |C_ij| vs distance (top-40 subset)")


    def fraction_in_group(indices, group):
        s = set(group)
        return np.mean([i in s for i in indices])

    top40 = np.argsort(score)[::-1][:40]
    print("Top40 fraction ORI_11m1:", fraction_in_group(top40, ORI_11m1))
    print("Top40 fraction ORI_m111:", fraction_in_group(top40, ORI_m111))

    # z = out["z"]
    # emax_null = shuffle_null_eigs(z, nperm=300, seed=1, mode="time_shuffle")

    # lam1_obs = out["evals_time"][0]
    # p = (np.sum(emax_null >= lam1_obs) + 1) / (len(emax_null) + 1)

    # print("Top eigen (obs):", lam1_obs)
    # print("Null mean±std:", emax_null.mean(), emax_null.std())
    # print("p-value:", p)

    # plt.figure()
    # plt.hist(emax_null, bins=35)
    # plt.axvline(lam1_obs, ls="--")
    # plt.xlabel("Top eigenvalue (null)")
    # plt.ylabel("count")
    # plt.title(f"Top-eigen null, p={p:.3g}")
    # plt.grid(True, ls="--", lw=0.5)
    # plt.show()
    
    
    # ---- run it ----
    xy_pix, valid = extract_xy_from_nv_list(nv_list, key="pixel")

    top = np.array([107,186,98,51,189,14,201,50,199,8,202,59,21,45,57,73,108,155,114,97])
    top = top[valid[top]]   # safety

    plot_highlight(xy_pix, top, title="Top correlated NVs (pixel coords)", invert_y=True)

    obs, null, p = perm_test_spatial_cluster(xy_pix[valid], np.where(valid)[0].searchsorted(top), nperm=2000)
    # The indexing above is only needed if you subselect valid; simplest is keep valid==all True for your data.
    print("cluster radius (obs):", obs, "p(clustered):", p)

    plt.figure()
    plt.hist(null, bins=40)
    plt.axvline(obs, ls="--")
    plt.xlabel("radius of gyration (random subsets)")
    plt.ylabel("count")
    plt.title(f"Spatial clustering test (smaller=more clustered), p={p:.3g}")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    xy_aod, valid_aod = extract_xy_from_nv_list(nv_list, key="laser_INTE_520_aod")
    plot_highlight(xy_aod, top, title="Top correlated NVs (520 AOD coords)", invert_y=False)

    w, V = eig_hermitian(out["C_time"])  # V[:,k]
    K = 6
    for k in range(K):
        vk = V[:,k]
        pr = 1.0 / np.sum((np.abs(vk)/np.linalg.norm(vk))**4)
        idxk = np.argsort(np.abs(vk))[::-1][:20]
        print(f"mode {k+1}: eval={w[k]:.6f}, PR~{pr:.1f}, top idx={idxk.tolist()}")
        plot_highlight(xy_pix, idxk, title=f"Mode {k+1} top NVs (pixel)", invert_y=True, label_first=12)


    score = np.sum(np.abs(V[:, :6])**2, axis=1)

    bright = np.mean(np.abs(out["s0"]), axis=1)  # proxy; or use raw counts if you prefer

    plt.figure(figsize=(4,4))
    plt.plot(bright, score, "o", ms=3)
    plt.xlabel("mean |s0| (brightness proxy)")
    plt.ylabel("participation score (sum |v_k|^2)")
    plt.title("Brightness vs correlated participation")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    print("corrcoef:", float(np.corrcoef(bright, score)[0,1]))


    plt.show()

    # Replace this with your actual counts:
    # raise SystemExit("Edit __main__ with your counts loader (npz / dm.get_raw_data / etc.)")
