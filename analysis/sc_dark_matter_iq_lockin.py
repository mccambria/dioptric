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
    flip_times = np.array([tau, 3*tau, 5*tau, 7*tau])
    flips = np.sum(t[:, None] >= flip_times[None, :], axis=1)
    return np.where(flips % 2 == 0, 1.0, -1.0)

def Y_from_y(t, y, w):
    dt = t[1] - t[0]
    phase = np.exp(1j * w[:, None] * t[None, :])
    return (phase @ y) * dt

def filter_plot(tau_hahn=15e-6, tau_xy4=3.75e-6, fmax=200e3):
    f = np.linspace(1e3, fmax, 5000)
    w = 2*np.pi*f

    # Hahn
    T_h = 2*tau_hahn
    t_h = np.linspace(0, T_h, 60000, endpoint=False)
    y_h = y_hahn(t_h, tau_hahn)
    F_h = np.abs(Y_from_y(t_h, y_h, w))**2
    F_h /= F_h.max()

    # XY4-1
    T_x = 8*tau_xy4
    t_x = np.linspace(0, T_x, 60000, endpoint=False)
    y_x = y_xy4_standard(t_x, tau_xy4)
    F_x = np.abs(Y_from_y(t_x, y_x, w))**2
    F_x /= F_x.max()

    plt.figure()
    plt.plot(f/1e3, F_h, label=f"Hahn tau={tau_hahn*1e6:.2f} us (T={2*tau_hahn*1e6:.1f} us)")
    plt.plot(f/1e3, F_x, label=f"XY4-1 tau={tau_xy4*1e6:.2f} us (T={8*tau_xy4*1e6:.1f} us)")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized |Y(ω)|²")
    plt.title("Filter functions (numerical)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("filter_functions.pdf")

def tau_mapping_plot():
    tau = np.linspace(1e-6, 30e-6, 400)
    f0_h = 1/(2*tau)   # Hahn approx
    f0_x = 1/(4*tau)   # XY4 approx (standard timing)

    plt.figure()
    plt.plot(tau*1e6, f0_h/1e3, label="Hahn: f0 ≈ 1/(2τ)")
    plt.plot(tau*1e6, f0_x/1e3, label="XY4: f0 ≈ 1/(4τ)")
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
    sigma = np.sqrt(np.mean(np.abs(s0)**2, axis=1, keepdims=True)) + eps
    sw = s0 / sigma
    R = (sw @ np.conjugate(sw.T)) / sw.shape[1]
    # enforce Hermitian numerically
    R = 0.5 * (R + np.conjugate(R.T))
    return sw, R

def mp_bounds(M, N):
    q = M / N
    lam_minus = (1 - np.sqrt(q))**2
    lam_plus  = (1 + np.sqrt(q))**2
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
    print(f"[DM lock-in] MP noise frac ~ {frac_noise_mp:.4f}   (q={q:.3f}, lam+={lam_plus:.3f})")

    # ---- Null distribution of frac ----
    fracs_null = null_frac_distribution(sw, n_null=n_null, method="roll", seed=1)
    pval = (np.sum(fracs_null >= frac_R) + 1) / (len(fracs_null) + 1)

    figs = []

    # A) Eigenvalue spectrum with MP bounds
    figA, axA = plt.subplots(figsize=(7, 3.8))
    axA.plot(np.arange(M), evals_R, marker=".", linewidth=1, label="Data eigvals (whitened R)")
    axA.axhline(lam_minus, linestyle="--", linewidth=1, label=r"MP $\lambda_-$")
    axA.axhline(lam_plus,  linestyle="--", linewidth=1, label=r"MP $\lambda_+$")
    axA.set_title("Eigenvalue spectrum (whitened correlation matrix)")
    axA.set_xlabel("mode index")
    axA.set_ylabel("eigenvalue")
    axA.legend()
    figs.append(figA)

    # B) Null histogram for top-mode fraction
    figB, axB = plt.subplots(figsize=(6.5, 3.5))
    axB.hist(fracs_null, bins=35, alpha=0.7, density=True, label="Null (roll each NV)")
    axB.axvline(frac_R, linewidth=2, label=f"Data frac={frac_R:.4f}")
    axB.axvline(frac_noise_mp, linestyle="--", linewidth=1.5, label=f"MP est={frac_noise_mp:.4f}")
    axB.set_title(f"Top-mode fraction null test (p ≈ {pval:.3f})")
    axB.set_xlabel(r"$\lambda_1 / \mathrm{Tr}$")
    axB.set_ylabel("density")
    axB.legend()
    figs.append(figB)

    # C) Heatmaps of whitened R
    figC, axC = plt.subplots(figsize=(7, 6))
    matC = np.real(0.5 * (R + R.conj().T))
    np.fill_diagonal(matC, np.nan)
    kpl.imshow(axC, matC, title="Re[R] (whitened)", cbar_label="Re", cmap="RdBu_r",
               nan_color=kpl.KplColors.GRAY)
    figs.append(figC)

    figD, axD = plt.subplots(figsize=(7, 6))
    matD = np.abs(R)
    np.fill_diagonal(matD, np.nan)
    kpl.imshow(axD, matD, title="|R| (whitened)", cbar_label="|.|", cmap="viridis",
               nan_color=kpl.KplColors.GRAY)
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


if __name__ == "__main__":
    kpl.init_kplotlib()
    # filter_plot(tau_hahn=15e-6, tau_xy4=3.75e-6, fmax=200e3)
    # tau_mapping_plot()
    
    raw_data = dm.get_raw_data(file_stem="2025_12_23-16_10_36-johnson-nv0_2025_10_21", load_npz=True)
    figs, proc = process_and_plot_dm_lockin(raw_data, show=False)
    kpl.show(block=True)
