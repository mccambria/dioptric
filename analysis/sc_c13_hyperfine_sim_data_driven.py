
# -*- coding: utf-8 -*-
"""
Spin-echo simulator for a single NV with flexible 13C configuration.

Key features:
- Fixed NV frame & B-field (no accidental re-randomization of orientation)
- Choose exact 13C sites (fixed_site_ids) or a full mask (fixed_presence_mask)
- Or Bernoulli-thin once and reuse (quenched) / redraw each time (ensemble)
- Correct scalar A_parallel and B_perp; angular frequency in Mk_tau
- Optional phenomenological fine-decay comb (can disable)

Dependencies: numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import data_manager as dm

# ---------- Optional numba (falls back gracefully) ----------
try:
    from numba import njit
except Exception:
    def njit(*_args, **_kwargs):
        def wrap(fn):
            return fn
        return wrap

# ---------- Physical constants ----------
D_NV = 2.87e9          # Hz (unused here, but kept for completeness)
gamma_e = 28e9         # Hz/T (unused here)
gamma_C13 = 10.705e6   # Hz/T  (13C gyromagnetic ratio)

# Default lab field in Gauss (example), rotated later into NV frame
B_vec_G = np.array([-6.18037755, -18.54113264, -43.26264283], dtype=float)
B_vec_T = B_vec_G * 1e-4

# =============================================================================
# Fine-decay (phenomenological)
# =============================================================================
def fine_decay(
    tau_us,
    baseline=1.0,
    comb_contrast=0.6,
    revival_time=37.0,
    width0_us=6.0,
    T2_ms=0.08,
    T2_exp=1.0,
    amp_taper_alpha=0.0,
    width_slope=0.0,
    revival_chirp=0.0,
    # NEW (additive, signed; zero-mean cos carrier(s))
    osc_amp=0.0,
    osc_f0=0.0,
    osc_phi0=0.0,
    osc_f1=0.0,
    osc_phi1=0.0,
):
    """
    signal(τ) = baseline
                - comb_contrast * envelope(τ) * COMB(τ)               [dip]
                +                 envelope(τ) * COMB(τ) * OSC(τ)      [signed oscillation]

    envelope(τ) = exp[-(τ / (1000*T2_ms))^T2_exp]
    COMB(τ)     = Σ_k [ 1/(1+k)^amp_taper_alpha ] * exp(-((τ-μ_k)/w_k)^4)
                    μ_k = k * revival_time * (1 + k*revival_chirp)
                    w_k = width0_us * (1 + k*width_slope)

    OSC(τ)      = osc_amp * [ cos(2π f0 τ + φ0) + cos(2π f1 τ + φ1) ]   # zero mean
    τ in microseconds, f in cycles/μs, phases in rad.
    """
    tau = np.asarray(tau_us, dtype=float).ravel()
    width0_us    = max(1e-9, float(width0_us))
    revival_time = max(1e-9, float(revival_time))
    T2_us        = max(1e-9, 1000.0 * float(T2_ms))
    T2_exp       = float(T2_exp)

    # envelope
    envelope = np.exp(-((tau / T2_us) ** T2_exp))

    # number of revivals to include
    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / revival_time)) + 1))

    comb = _comb_quartic_powerlaw(
        tau,
        revival_time,
        width0_us,
        amp_taper_alpha,
        width_slope,
        revival_chirp,
        n_guess
    )


    # beating lives in MOD; comb_contrast is the overall amplitude (once)
    if (osc_amp != 0.0) and (osc_f0 != 0.0 or osc_f1 != 0.0):
        s0 = np.sin(np.pi * osc_f0 * tau + osc_phi0)
        s1 = np.sin(np.pi * osc_f1 * tau + osc_phi1)
        beat = (s0 * s0) * (s1 * s1)
        mod = comb_contrast - osc_amp* beat
    else:
        mod = comb_contrast
    
    return baseline - envelope * mod * comb

@njit
def _comb_quartic_powerlaw(
    tau,
    revival_time,
    width0_us,
    amp_taper_alpha,
    width_slope,
    revival_chirp,
    n_guess
):
    n = tau.shape[0]
    out = np.zeros(n, dtype=np.float64)
    tmax = 0.0
    for i in range(n):
        if tau[i] > tmax:
            tmax = tau[i]

    for k in range(n_guess):
        mu_k = k * revival_time * (1.0 + k * revival_chirp)
        w_k  = width0_us * (1.0 + k * width_slope)
        if w_k <= 0.0:
            continue
        if mu_k > tmax + 5.0 * w_k:
            break

        amp_k = 1.0 / ((1.0 + k) ** amp_taper_alpha)
        inv_w4 = 1.0 / (w_k ** 4)

        for i in range(n):
            x = tau[i] - mu_k
            out[i] += amp_k * np.exp(- (x * x) * (x * x) * inv_w4)

    return out

def revivals_only_mapping(microscopic, taus_s, p, power=2.0):
    """
    Gate microscopic deviations to revivals AND add a zero-mean oscillatory term
    so the signal can go above baseline near revivals (as seen experimentally).

    p expects (in addition to your usual fine params):
      baseline, comb_contrast,
      revival_time (us), width0_us (us), T2_ms, T2_exp,
      amp_taper_alpha, width_slope, revival_chirp,
      # NEW (all optional):
      osc_add_amp     : amplitude of above/below-baseline oscillation (0.. ~0.5)
      osc_f0_MHz      : first oscillation frequency in MHz (cycles/μs)
      osc_f1_MHz      : (optional) second frequency in MHz (set 0 to disable)
      osc_phi0        : phase (rad) for f0
      osc_phi1        : phase (rad) for f1
    """
    # ---- unpack ----
    baseline      = float(p.get("baseline", 0.6))
    comb_contrast = float(p.get("comb_contrast", 0.4))
    Trev_us       = max(1e-9, float(p.get("revival_time", 37.0)))
    w0_us         = max(1e-9, float(p.get("width0_us", 6.0)))
    T2_ms         = float(p.get("T2_ms", 0.08))
    T2_exp        = float(p.get("T2_exp", 1.2))
    taper         = float(p.get("amp_taper_alpha", 0.0))
    w_slope       = float(p.get("width_slope", 0.0))
    chirp         = float(p.get("revival_chirp", 0.0))

    # NEW oscillation controls
    osc_add_amp = float(p.get("osc_add_amp", 0.0))        # amplitude around 0
    f0_MHz      = float(p.get("osc_f0_MHz", 0.0))
    f1_MHz      = float(p.get("osc_f1_MHz", 0.0))
    phi0        = float(p.get("osc_phi0", 0.0))
    phi1        = float(p.get("osc_phi1", 0.0))

    taus_us = np.asarray(taus_s, float) * 1e6
    m = np.asarray(microscopic, float)
    if not np.all(np.isfinite(m)):
        m = np.nan_to_num(m, nan=1.0)

    # ---- comb mask (0..1), tightened by 'power' ----
    tau_max = float(np.nanmax(taus_us)) if taus_us.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / Trev_us)) + 1))
    mask = _comb_quartic_powerlaw(taus_us, Trev_us, w0_us, taper, w_slope, chirp, n_guess)
    mask = np.nan_to_num(mask, nan=0.0)
    mmax = mask.max() if mask.size else 0.0
    mask = (mask / mmax) if mmax > 1e-12 else np.ones_like(taus_us)
    if power != 1.0:
        mask = np.power(np.clip(mask, 0.0, 1.0), float(power))

    # ---- global stretched-exponential envelope ----
    T2_us = max(1e-9, 1000.0 * T2_ms)
    env = np.exp(-((taus_us / T2_us) ** T2_exp))

    # ---- keep microscopic dips ONLY near revivals ----
    dev = (1.0 - m)            # deviation below 1
    gated_dev = dev * mask
    gated = 1.0 - gated_dev    # back near 1 outside revivals

    # ---- additive zero-mean oscillation (gated + enveloped) ----
    osc = 0.0
    if osc_add_amp > 0.0 and (f0_MHz > 0.0 or f1_MHz > 0.0):
        th0 = 2*np.pi*(f0_MHz * 1e6) * (taus_s) + phi0  # f in Hz, tau in s
        car = np.sin(th0)
        if f1_MHz > 0.0:
            th1 = 2*np.pi*(f1_MHz * 1e6) * (taus_s) + phi1
            car = car * np.sin(th1)   # product -> beating
        # Gate and decay the *additive* oscillation so it lives only at revivals
        osc = osc_add_amp * mask * env * car

    # ---- final mapping: baseline + dips + gated oscillation ----
    # dips: baseline - comb_contrast * (1 - gated) * env
    y = baseline - comb_contrast * (1.0 - gated) * env + osc

    # Keep in display range if you normalize 0..1
    return np.clip(y, 0.0, 1.0)

def _synthesize_comb_only(taus_sec, p):
    return fine_decay(
        tau_us=taus_sec * 1e6,
        baseline=float(p.get("baseline", 1.0)),
        comb_contrast=float(p.get("comb_contrast", 0.6)),
        revival_time=float(p.get("revival_time", 37.0)),
        width0_us=float(p.get("width0_us", 6.0)),
        T2_ms=float(p.get("T2_ms", 0.08)),
        T2_exp=float(p.get("T2_exp", 1.0)),
        amp_taper_alpha=float(p.get("amp_taper_alpha", 0.0)),
        width_slope=float(p.get("width_slope", 0.0)),
        revival_chirp=float(p.get("revival_chirp", 0.0)),
        osc_amp=0.0,          # <- explicitly no oscillation when synthesizing comb-only
        osc_f0=0.0, osc_f1=0.0, osc_phi0=0.0, osc_phi1=0.0,
    )

# =============================================================================
# Hyperfine handling
# =============================================================================
def _orthonormal_basis_from_z(z):
    """Build an ONB {ez=z/|z|, e1, e2}."""
    ez = np.asarray(z, float)
    ez /= np.linalg.norm(ez)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - ez * np.dot(tmp, ez)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ez, e1)
    return ez, e1, e2

def compute_hyperfine_components(A_tensor, B_unit):
    """
    Return scalar A_parallel and scalar B_perp (effective transverse coupling),
    consistent with central-spin secular reduction used in Mk_tau.
    """
    ez, e1, e2 = _orthonormal_basis_from_z(B_unit)
    A_par  = ez @ A_tensor @ ez
    Aperp1 = e1 @ A_tensor @ ez
    Aperp2 = e2 @ A_tensor @ ez
    B_eff  = np.sqrt(Aperp1**2 + Aperp2**2)
    return A_par, B_eff

# =============================================================================
# Single-nucleus echo factor Mk(τ) and many-spin product
# =============================================================================
def Mk_tau(A_Hz, B_Hz, tau_s, omegaL_Hz):
    """
    Single-spin Hahn-echo contribution (dimensionless).
    A_Hz, B_Hz, omegaL_Hz are in Hz; internally convert to angular frequency.
    """
    # Convert to angular frequency (rad/s)
    omega = 2*np.pi * np.sqrt(B_Hz**2 + (A_Hz - omegaL_Hz)**2)
    return 1.0 - 2.0 * ( (2*np.pi*B_Hz)**2 / (omega**2) ) * (np.sin(omega * tau_s / 2.0)**4)

def compute_echo_signal(hyperfine_tensors, tau_array_s, B_field_vec_T):
    B_mag = np.linalg.norm(B_field_vec_T)
    if B_mag == 0.0:
        raise ValueError("B-field magnitude is zero.")
    B_unit = B_field_vec_T / B_mag
    omega_L = gamma_C13 * B_mag  # Hz

    signal = np.empty_like(tau_array_s, dtype=float)
    for i, tau in enumerate(tau_array_s):
        Mk_prod = 1.0
        for A_tensor in hyperfine_tensors:
            A, B = compute_hyperfine_components(A_tensor, B_unit)
            Mk_prod *= Mk_tau(A, B, tau, omega_L)
        signal[i] = 0.5 * (1.0 + Mk_prod)
    return signal

def Mk_tau(A_Hz, B_Hz, tau_s, omegaL_Hz):
    omega = 2*np.pi * np.sqrt(B_Hz**2 + (A_Hz - omegaL_Hz)**2)
    return 1.0 - 2.0 * ((2*np.pi*B_Hz)**2 / (omega**2)) * (np.sin(omega * tau_s / 2.0)**4)

def compute_echo_signal(hyperfine_tensors, tau_array_s, B_field_vec_T,
                        sigma_B_G=0.0, eps_contrast=0.0, rng=None):
    B_vec = np.array(B_field_vec_T, float)
    if sigma_B_G > 0.0 and rng is not None:
        B_vec = B_vec + 1e-4 * rng.normal(0.0, sigma_B_G, size=3)  # G→T

    B_mag = np.linalg.norm(B_vec)
    if B_mag == 0.0:
        raise ValueError("B-field magnitude is zero.")
    B_unit = B_vec / B_mag
    omega_L = gamma_C13 * B_mag  # Hz

    eta = max(0.0, min(1.0, 1.0 - eps_contrast))  # 0..1
    signal = np.empty_like(tau_array_s, dtype=float)

    for i, tau in enumerate(tau_array_s):
        Mk_prod = 1.0
        for A_tensor in hyperfine_tensors:
            A_par, B_perp = compute_hyperfine_components(A_tensor, B_unit)
            Mk = Mk_tau(A_par, B_perp, tau, omega_L)
            if eps_contrast > 0.0:
                Mk = 1.0 - eta * (1.0 - Mk)  # gentle contrast damping
            Mk_prod *= Mk
        signal[i] = 0.5 * (1.0 + Mk_prod)

    return signal

# =============================================================================
# RNG helpers and selection
# =============================================================================
def _spawn_streams(seed, num_streams, run_salt=None):
    if seed is None:
        ss = np.random.SeedSequence()
    else:
        if run_salt is None:
            ss = np.random.SeedSequence(int(seed))
        else:
            run_salt = int(run_salt) & 0xFFFFFFFF
            ss = np.random.SeedSequence(int(seed), spawn_key=[run_salt])
    child_seqs = ss.spawn(num_streams)
    return [np.random.default_rng(cs) for cs in child_seqs]

def _choose_sites(rng, present_sites, num_spins, selection_mode="uniform"):
    present_sites = list(present_sites)
    if num_spins is None or num_spins >= len(present_sites):
        return present_sites

    if selection_mode == "top_Apar":
        order = np.argsort([-abs(s["Apar_Hz"]) for s in present_sites])
        idx = order[:num_spins]
    elif selection_mode == "distance_weighted":
        r = np.array([max(s["dist"], 1e-12) for s in present_sites], float)
        w = (1.0 / (r**3)); w /= w.sum()
        idx = np.sort(rng.choice(len(present_sites), size=num_spins, replace=False, p=w))
    else:  # uniform
        idx = np.sort(rng.choice(len(present_sites), size=num_spins, replace=False))
    return [present_sites[i] for i in idx]

# =============================================================================
# Main simulator
# =============================================================================
def simulate_random_spin_echo_average(
    hyperfine_path,
    tau_range_us,
    num_spins=30,
    num_realizations=1,
    distance_cutoff=None,
    Ak_min_kHz=None,   # keep if A∥ ≥ Ak_min_kHz (if set)
    Ak_max_kHz=None,   # keep if A∥ ≤ Ak_max_kHz (if set)
    Ak_abs=True,       # compare |A∥| if True, signed A∥ if False
    R_NV=np.eye(3),
    fine_params=None,              # set None for microscopic-only
    abundance_fraction=0.011,
    rng_seed=None,
    run_salt=None,
    randomize_positions=False,     # keep False for single NV
    selection_mode="top_Apar",
    ensure_unique_across_realizations=False,  # usually False for fixed NV
    annotate_from_realization=0,
    keep_nv_orientation=True,      # keep True for single NV
    fixed_site_ids=None,           # exact sites to include
    fixed_presence_mask=None,      # boolean mask of length N_sites
    reuse_present_mask=True,       # draw Bernoulli once and reuse (quenched)
):
    """
    Returns:
      taus_us, avg_signal, aux
      aux = {
        "positions": (N,3) of annotated realization (NV frame),
        "site_info": [{site_id, Apar_kHz, r}, ...],
        "revivals_us": array of k*revival_time for plotting,
        "picked_ids_per_realization": [[...], ...],
        "stats": {...}
      }
    """
    # Time axis
    taus_s = np.linspace(float(tau_range_us[0]), float(tau_range_us[1]), num=300) * 1e-6

    # RNG
    rng_streams = _spawn_streams(rng_seed, max(1, num_realizations), run_salt=run_salt)

    # Rotate B into NV frame once
    B_vec_NV = R_NV @ B_vec_T
    B_hat_NV = B_vec_NV / np.linalg.norm(B_vec_NV)

    # Load hyperfine file (formatted .txt with header then table)
    file_path = Path(hyperfine_path)
    with open(file_path, "r") as f:
        lines = f.readlines()
    data_start = next(i for i, line in enumerate(lines) if line.strip().startswith("1 "))
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        skiprows=data_start,
        header=None,
        names=["index", "distance", "x", "y", "z", "Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz"],
    )
    if distance_cutoff is not None:
        df = df[df["distance"] < distance_cutoff]

    # Build site list in NV frame (positions rotated once; tensors rotated once)
    sites = []
    for _, row in df.iterrows():
        A = np.array([[row.Axx, row.Axy, row.Axz],
                      [row.Axy, row.Ayy, row.Ayz],
                      [row.Axz, row.Ayz, row.Azz]], float) * 1e6  # MHz -> Hz
        A_nv = R_NV @ A @ R_NV.T

        # Apparent A_parallel for current B (NV frame)
        A_par, _ = compute_hyperfine_components(A_nv, B_hat_NV)
        # --------- NEW A∥ filter (replaces: if np.abs(A_par) > Ak_cutoff_kHz * 1e3:) ---------
        A_par_kHz = (abs(A_par) if Ak_abs else A_par) / 1e3
        keep_A = True
        if Ak_min_kHz is not None:
            keep_A &= (A_par_kHz >= float(Ak_min_kHz))
        if Ak_max_kHz is not None:
            keep_A &= (A_par_kHz <= float(Ak_max_kHz))
        if not keep_A:
            continue
        # ---------------------------------------------------------------------------
        pos_crystal = np.array([row.x, row.y, row.z], float)
        pos_nv = R_NV @ pos_crystal
        sites.append({
            "site_id": int(row["index"]),
            "A0": A_nv,
            "pos0": pos_nv,
            "dist": float(row.distance),
            "Apar_Hz": float(A_par),
        })
    N_candidates = len(sites)
    if N_candidates == 0:
        taus_us = taus_s * 1e6
        flat = np.ones_like(taus_us)
        return taus_us, flat, {
            "positions": None, "site_info": [], "revivals_us": None,
            "picked_ids_per_realization": [], "stats": {}
        }

    id_to_idx = {s["site_id"]: i for i, s in enumerate(sites)}

    present_mask_global = None
    if (fixed_site_ids is None) and (fixed_presence_mask is None) and reuse_present_mask:
        rng_once = rng_streams[0]
        present_mask_global = (rng_once.random(N_candidates) < abundance_fraction)

    # Containers
    all_signals = []
    picked_ids_per_realization = []
    present_counts = []
    chosen_counts  = []
    anno_positions = None
    anno_site_info = None
    anno_rev_times = None
    used_site_ids = set()

    for r in range(num_realizations):
        rng_r = rng_streams[r]

        # Decide occupancy
        if fixed_site_ids:
            present_idxs = np.array([id_to_idx[i] for i in fixed_site_ids if i in id_to_idx], int)
            present_mask = np.zeros(N_candidates, dtype=bool)
            present_mask[present_idxs] = True
        elif fixed_presence_mask is not None:
            mask = np.asarray(fixed_presence_mask, bool)
            if mask.size != N_candidates:
                raise ValueError("fixed_presence_mask length does not match candidate site count.")
            present_mask = mask
            present_idxs = np.flatnonzero(present_mask)
        elif present_mask_global is not None:
            present_mask = present_mask_global
            present_idxs = np.flatnonzero(present_mask)
        else:
            present_mask = (rng_r.random(N_candidates) < abundance_fraction)
            present_idxs = np.flatnonzero(present_mask)

        present_counts.append(int(present_mask.sum()))
        if present_idxs.size == 0:
            all_signals.append(np.ones_like(taus_s))
            picked_ids_per_realization.append([])
            continue

        present_sites = [sites[i] for i in present_idxs]

        # Optional cross-realization uniqueness (usually False for fixed NV)
        if ensure_unique_across_realizations:
            filtered = [s for s in present_sites if s["site_id"] not in used_site_ids]
            if len(filtered) >= max(1, num_spins if num_spins is not None else 1):
                present_sites = filtered

        # Choose final subset (or take all)
        chosen_sites = _choose_sites(rng_r, present_sites, num_spins, selection_mode)
        chosen_counts.append(len(chosen_sites))
        picked_ids = [s["site_id"] for s in chosen_sites]
        picked_ids_per_realization.append(picked_ids)
        used_site_ids.update(picked_ids)

        # Prepare tensors for microscopic echo
        tensors = [s["A0"] for s in chosen_sites]

        # Compute echo
        signal = compute_echo_signal(tensors, taus_s, B_vec_NV)
        all_signals.append(signal)

        # Annotation (first realization by default)
        if r == annotate_from_realization:
            anno_positions = np.array([s["pos0"] for s in chosen_sites]) if chosen_sites else None
            anno_site_info = [{
                "site_id": s["site_id"],
                "Apar_kHz": float(abs(compute_hyperfine_components(s["A0"], B_hat_NV)[0]) / 1e3),
                "r": float(np.linalg.norm(s["pos0"]))
            } for s in chosen_sites]
            if fine_params is not None and "revival_time" in fine_params:
                revT_us = float(fine_params["revival_time"])
                kmax = int(np.ceil((taus_s.max()*1e6) / revT_us))
                anno_rev_times = np.arange(0, kmax+1) * revT_us

    # Average (for single realization this is just identity)
    avg_signal = np.mean(all_signals, axis=0)

    # phenomenological gating
    if fine_params is not None:
        if np.nanmax(avg_signal) - np.nanmin(avg_signal) < 1e-4:
            # no microscopic modulation -> synthesize clean comb at your baseline
            avg_signal = _synthesize_comb_only(taus_s, fine_params)
        else:
            # <-- key line: gate deviations so oscillations live only near revivals
            avg_signal = revivals_only_mapping(avg_signal, taus_s, fine_params, power=2.0)

    stats = {
        "N_candidates": N_candidates,
        "abundance_fraction": float(abundance_fraction),
        "present_counts": present_counts,
        "chosen_counts": chosen_counts,
    }

    return taus_s * 1e6, avg_signal, {
        "positions": anno_positions,
        "site_info": anno_site_info if anno_site_info is not None else [],
        "revivals_us": anno_rev_times,
        "picked_ids_per_realization": picked_ids_per_realization,
        "stats": stats,
        "all_candidate_positions": np.array([s["pos0"] for s in sites], float),  # NEW
    }

# =============================================================================
# Plotting
# =============================================================================
def set_axes_equal_3d(ax):
    """Make 3D axes have equal scale (so spheres look like spheres)."""
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xmid = 0.5*(xlim[0]+xlim[1]); ymid = 0.5*(ylim[0]+ylim[1]); zmid = 0.5*(zlim[0]+zlim[1])
    max_range = 0.5*max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
    ax.set_xlim3d(xmid - max_range, xmid + max_range)
    ax.set_ylim3d(ymid - max_range, ymid + max_range)
    ax.set_zlim3d(zmid - max_range, zmid + max_range)

def _echo_summary_lines(taus_us, echo):
    if len(echo) == 0:
        return []
    arr = np.asarray(echo, float)
    n = max(3, len(arr)//3)
    early = float(np.nanmean(arr[:n])); late = float(np.nanmean(arr[-n:]))
    return [f"Echo range: {arr.min():.3f} … {arr.max():.3f}",
            f"⟨early⟩→⟨late⟩: {early:.3f} → {late:.3f}"]

def _fine_param_lines(fine_params):
    if not fine_params: return []
    pretty = {
        "revival_time":"T_rev (μs)",
        "width0_us":"width₀ (μs)",
        "T2_ms":"T₂ (ms)",
        "T2_exp":"stretch n",
        "amp_taper_alpha":"amp taper α",
        "width_slope":"width slope",
        "revival_chirp":"rev chirp",
    }
    keys = ["revival_time","width0_us","T2_ms","T2_exp","amp_taper_alpha","width_slope","revival_chirp"]
    out = []
    for k in keys:
        if k in fine_params:
            v = fine_params[k]
            sval = f"{v:.3g}" if isinstance(v,(int,float)) else f"{v}"
            out.append(f"{pretty[k]}: {sval}")
    return out

def _site_table_lines(site_info, max_rows=8):
    if not site_info: return ["(no annotated realization)"]
    rows = sorted(site_info, key=lambda d: -abs(d.get("Apar_kHz", 0.0)))
    lines = ["site  |A∥|(kHz)   r", "------------------------"]
    for d in rows[:max_rows]:
        sid  = d.get("site_id","?")
        apar = float(abs(d.get("Apar_kHz",0.0)))
        rmag = float(d.get("r", np.nan))
        lines.append(f"{sid:<5} {apar:>8.0f}  {rmag:>6.2f}")
    if len(rows) > max_rows:
        lines.append(f"... (+{len(rows)-max_rows} more)")
    return lines

def plot_echo_with_sites(taus_us, echo, aux, title="Spin Echo (single NV)", rmax=None, fine_params=None, units_label="(arb units)"):
    """
    Panels:
      [0] Echo vs τ with optional revival lines + info boxes
      [1] 3D positions of chosen 13C sites (NV at origin), equal aspect, symmetric limits,
          optional backdrop of all candidates (light gray), NO color scale, with per-site annotations.
    """
    fig = plt.figure(figsize=(12, 5))

    # ---------------- Echo panel ----------------
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.plot(taus_us, echo, lw=1.0)
    ax0.set_xlabel("τ (μs)")
    ax0.set_ylabel(f"Coherence {units_label}")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)

    revs = aux.get("revivals_us", None)
    if revs is not None:
        for t in np.atleast_1d(revs):
            ax0.axvline(t, ls="--", lw=0.7, alpha=0.35)

    stats = aux.get("stats", {}) or {}
    lines_stats = []
    if "N_candidates" in stats:
        lines_stats.append(f"Candidates: {stats['N_candidates']}")
    if "abundance_fraction" in stats:
        lines_stats.append(f"Abundance p: {100*stats['abundance_fraction']:.2f}%")
    if "chosen_counts" in stats and stats["chosen_counts"]:
        cc = np.asarray(stats["chosen_counts"], int)
        lines_stats.append(f"Chosen/site per realization: {int(np.median(cc))} (med)")
    if lines_stats:
        ax0.text(0.61, 0.02, "\n".join(lines_stats), transform=ax0.transAxes,
                 fontsize=9, va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6))

    fp_lines = _fine_param_lines(fine_params)
    if fp_lines:
        ax0.text(0.99, 0.5, "\n".join(fp_lines), transform=ax0.transAxes,
                 fontsize=9, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6))

    # es_lines = _echo_summary_lines(taus_us, echo)
    # if es_lines:
    #     ax0.text(0.99, 0.98, "\n".join(es_lines), transform=ax0.transAxes,
    #              fontsize=9, va="top", ha="right",
    #              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, lw=0.6))

    picked_all = aux.get("picked_ids_per_realization", [])
    if picked_all and len(picked_all[0]) > 0:
        print("picked:", picked_all[0])

    # ---------------- 3D positions panel ----------------
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    # Optional backdrop of ALL candidate sites (light gray)
    bg = aux.get("all_candidate_positions", None)
    if bg is not None and len(bg) > 0:
        ax1.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=8, alpha=0.15)

    # Chosen sites (NO color mapping, uniform markers) + annotations
    pos  = aux.get("positions", None)
    info = aux.get("site_info", [])
    if pos is not None and len(pos) > 0:
        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=42, depthshade=True)
        # annotate every chosen site with id, |A∥| and r
        for pnt, meta in zip(pos, info):
            sid  = meta.get("site_id","?")
            apar = meta.get("Apar_kHz", 0.0)
            rmag = meta.get("r", np.nan)
            label = f'{sid}\n|A∥|={apar:.0f} kHz\nr={rmag:.2f}'
            ax1.text(pnt[0], pnt[1], pnt[2], label, fontsize=8, ha="left", va="bottom")

    # NV at origin
    ax1.scatter([0], [0], [0], s=70, marker="*", zorder=5)
    ax1.text(0, 0, 0, "NV", fontsize=9, ha="right", va="top")
    ax1.set_title("¹³C positions (NV frame)")
    ax1.set_xlabel("x (Å)"); ax1.set_ylabel("y (Å)"); ax1.set_zlabel("z (Å)")

    # Symmetric limits about NV
    if rmax is None:
        if bg is not None and len(bg) > 0:
            rmax = float(np.max(np.linalg.norm(bg, axis=1)))
        elif pos is not None and len(pos) > 0:
            rmax = float(np.max(np.linalg.norm(pos, axis=1)))
        else:
            rmax = 1.0
    rpad = 0.05 * rmax
    ax1.set_xlim(-rmax - rpad, rmax + rpad)
    ax1.set_ylim(-rmax - rpad, rmax + rpad)
    ax1.set_zlim(-rmax - rpad, rmax + rpad)

    set_axes_equal_3d(ax1)

    # Info boxes (overlay as 2D text on the 3D axes)
    n_real = len(picked_all) if picked_all is not None else 0
    n_chosen = len(info) if info is not None else 0
    left_box = [f"Chosen in annotated: {n_chosen}",
                f"Realizations: {n_real}"]
    if stats.get("N_candidates") is not None:
        left_box.append(f"Candidates: {stats['N_candidates']}")
    # ax1.text2D(0.01, 0.02, "\n".join(left_box), transform=ax1.transAxes,
    #            fontsize=9, va="bottom", ha="left",
    #            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, lw=0.6))

    table_lines = _site_table_lines(info, max_rows=8)
    ax1.text2D(0.99, 0.02, "\n".join(table_lines), transform=ax1.transAxes,
               fontsize=9, family="monospace", va="bottom", ha="right",
               bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6))

    plt.tight_layout()
    return fig


# =============================================================================
# Example usage
# =============================================================================
# if __name__ == "__main__":

# --- 0) Pull your saved fit dict (as you outlined) ---
file_stem = "2025_11_02-19_55_17-johnson_204nv_s3-003c56"
fit = dm.get_raw_data(file_stem=file_stem)

keys = fit["unified_keys"]
def _asdict(p):
    d = {k: None for k in keys}
    if p is None: 
        return d
    for k, v in zip(keys, p + [None]*(len(keys)-len(p))):
        d[k] = v
    return d

labels = list(map(int, fit["nv_labels"]))
popts  = fit["popts"]
chis   = fit.get("red_chi2", [None]*len(popts))

# --- 1) Make a clean parameter matrix (NaNs for missing), in unified_keys order ---
P = np.full((len(popts), len(keys)), np.nan, float)
for i, p in enumerate(popts):
    d = _asdict(p)
    for j, k in enumerate(keys):
        v = d[k]
        if v is None: continue
        try:
            P[i, j] = float(v)
        except Exception:
            pass

# Convenience indices
K = {k: j for j, k in enumerate(keys)}
k_base = K["baseline"]; k_cc  = K["comb_contrast"]
k_Trev = K["revival_time_us"]; k_w0 = K["width0_us"]
k_T2   = K["T2_ms"]; k_n     = K["T2_exp"]
k_a    = K["amp_taper_alpha"]; k_ws = K["width_slope"]; k_ch = K["revival_chirp"]
k_A    = K["osc_amp"]; k_f0   = K["osc_f0"]; k_f1 = K["osc_f1"]; k_p0 = K["osc_phi0"]; k_p1 = K["osc_phi1"]

# --- 2) Build cohort priors (median & MAD) for each parameter ---
def _nanmedian(x): return np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else np.nan
def _mad(x):
    med = _nanmedian(x)
    if not np.isfinite(med): return np.nan
    return _nanmedian(np.abs(x - med)) * 1.4826

cohort_med = np.array([_nanmedian(P[:,j]) for j in range(P.shape[1])])
cohort_mad = np.array([_mad(P[:,j])       for j in range(P.shape[1])])

# Reasonable hard bounds/guards (edit to taste)
bounds = {
    "baseline": (0.0, 1.2),
    "comb_contrast": (0.0, 1.1),
    "revival_time_us": (25.0, 55.0),
    "width0_us": (1.0, 25.0),
    "T2_ms": (1e-3, 2e3),
    "T2_exp": (0.6, 4.0),
    "amp_taper_alpha": (0.0, 2.0),
    "width_slope": (-0.2, 0.2),
    "revival_chirp": (-0.06, 0.06),
    "osc_amp": (-0.3, 0.3),     # allow above-baseline crests
    "osc_f0": (0.0, 0.50),      # cycles/μs ~ MHz
    "osc_f1": (0.0, 0.50),
    "osc_phi0": (-np.pi, np.pi),
    "osc_phi1": (-np.pi, np.pi),
}

def _clip_by_key(k, v):
    lo, hi = bounds[k]
    return float(np.minimum(hi, np.maximum(lo, v)))

# --- 3) NV-specific jitter scale (use MAD around each NV if you have repeats; else cohort MAD) ---
# Here we’ll just use cohort MAD and a mixing factor so we don’t over-jitter.
mix_global = 0.3   # 30% cohort prior noise
rng = np.random.default_rng(20251102)

def _nv_prior_draw(i):
    out = {}
    for k, j in K.items():
        # center at NV’s fitted value if finite; else cohort median
        mu = P[i, j] if np.isfinite(P[i, j]) else cohort_med[j]
        if not np.isfinite(mu):
            # final fallback constants
            if k == "baseline": mu = 0.6
            elif k == "comb_contrast": mu = 0.45
            elif k == "revival_time_us": mu = 38.0
            elif k == "width0_us": mu = 7.0
            elif k == "T2_ms": mu = 0.08
            elif k == "T2_exp": mu = 1.2
            else:
                mu = 0.0

        # jitter σ: a blend of cohort MAD and a small absolute floor
        sig = cohort_mad[j]
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1e-3
        # symmetric jitter for all (including phases); widen a bit for osc_amp
        widen = 1.0 if k != "osc_amp" else 1.5
        draw = mu + widen * mix_global * sig * rng.standard_normal()

        out[k] = _clip_by_key(k, draw)
    return out

# --- 4) Run stochastic synthetic echoes per NV, then aggregate ---
def synth_per_nv(nv_idx, R=8, tau_range_us=(0, 100),
                 hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",
                 abundance_fraction=0.011,
                 distance_cutoff=8.0,
                 num_spins=None,               # None = all present, else subsample
                 selection_mode="top_Apar",    # or "uniform" / "distance_weighted"
                 reuse_present_mask=True):
    lbl = labels[nv_idx]
    # precompute a reproducible stream salt per NV
    salt = int(lbl) & 0xFFFFFFFF

    # Pull phenomenology draw for each realization
    traces = []
    fine_list = []
    for r in range(R):
        fp = _nv_prior_draw(nv_idx)
        fine_params = dict(
            baseline=fp["baseline"],
            comb_contrast=fp["comb_contrast"],
            revival_time=fp["revival_time_us"],
            width0_us=fp["width0_us"],
            T2_ms=fp["T2_ms"],
            T2_exp=fp["T2_exp"],
            amp_taper_alpha=fp["amp_taper_alpha"],
            width_slope=fp["width_slope"],
            revival_chirp=fp["revival_chirp"],
            # Additive oscillation around revivals (your model uses MHz==cycles/us)
            osc_add_amp=fp["osc_amp"],
            osc_f0_MHz=fp["osc_f0"],
            osc_f1_MHz=fp["osc_f1"],
            osc_phi0=fp["osc_phi0"],
            osc_phi1=fp["osc_phi1"],
        )

        taus, echo, aux = simulate_random_spin_echo_average(
            hyperfine_path=hyperfine_path,
            tau_range_us=tau_range_us,
            num_spins=num_spins,
            num_realizations=1,
            distance_cutoff=8.0,
            Ak_min_kHz=0,   # keep if A∥ ≥ Ak_min_kHz (if set)
            Ak_max_kHz=600,   # keep if A∥ ≤ Ak_max_kHz (if set)
            Ak_abs=True,       # compare |A∥| if True, signed A∥ if False
            R_NV=np.eye(3),
            fine_params=fine_params,
            abundance_fraction=abundance_fraction,
            rng_seed=4242, run_salt=salt + r,   # reproducible per NV + realization
            randomize_positions=False,
            selection_mode=selection_mode,
            ensure_unique_across_realizations=False,
            annotate_from_realization=0,
            keep_nv_orientation=True,
            fixed_site_ids=None,
            fixed_presence_mask=None,
            reuse_present_mask=reuse_present_mask,
        )
        traces.append(echo)
        fine_list.append(fine_params)
        # plot_echo_with_sites(taus, echo, aux, title="Spin Echo (single NV)")

    traces = np.asarray(traces, float)        # [R, T]
    mean = np.nanmean(traces, axis=0)
    p16  = np.nanpercentile(traces, 16, axis=0)
    p84  = np.nanpercentile(traces, 84, axis=0)
    return dict(label=int(lbl), taus_us=taus, mean=mean, p16=p16, p84=p84,
                fine_draws=fine_list)

# --- 5) Orchestrate over a subset / all NVs ---
# Example: pick K best NVs by chi2 and simulate each
def _to_float_or_inf(x):
    try:
        if x is None: return np.inf
        v = float(x);  return v if np.isfinite(v) else np.inf
    except Exception:
        return np.inf
    
def main():
    chi_vals = np.array([_to_float_or_inf(c) for c in chis], float)
    order = np.argsort(chi_vals)  # lowest χ² first, infs at end
    keep = [i for i in order if popts[i] is not None][:200]  # e.g., top 20 NVs

    results = []
    for i in keep:
        res = synth_per_nv(i, R=1, tau_range_us=(0, 100),
                        hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",
                        num_spins=None, selection_mode="uniform")
        results.append(res)

# 'results' now has per-NV mean/p16/p84 bands you can plot quickly.
# (Use your plot_echo_with_sites for single-NV visualization if you want detail.)
if __name__ == "__main__":
    main()
    plt.show()

    
