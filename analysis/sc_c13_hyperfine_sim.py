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

    carrier = envelope * comb

    # baseline minus revival dip
    dip = comb_contrast * carrier

    # additive, zero-mean oscillation (can push above baseline)
    osc = 0.0
    if osc_amp != 0.0:
        if osc_f0 != 0.0:
            osc += np.cos(2*np.pi*osc_f0 * tau + osc_phi0)
        if osc_f1 != 0.0:
            osc += np.cos(2*np.pi*osc_f1 * tau + osc_phi1)

    return baseline - dip + carrier * (osc_amp * osc)

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
    Ak_cutoff_kHz=0.0,
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
        if np.abs(A_par) > Ak_cutoff_kHz * 1e3:
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
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, lw=0.6))

    fp_lines = _fine_param_lines(fine_params)
    if fp_lines:
        ax0.text(0.99, 0.5, "\n".join(fp_lines), transform=ax0.transAxes,
                 fontsize=9, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, lw=0.6))

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
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

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
    ax1.text2D(0.01, 0.02, "\n".join(left_box), transform=ax1.transAxes,
               fontsize=9, va="bottom", ha="left",
               bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, lw=0.6))

    table_lines = _site_table_lines(info, max_rows=8)
    ax1.text2D(0.99, 0.02, "\n".join(table_lines), transform=ax1.transAxes,
               fontsize=9, family="monospace", va="bottom", ha="right",
               bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, lw=0.6))

    plt.tight_layout()
    return fig


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Example file path (replace with your actual file)
    input_file = r"analysis/nv_hyperfine_coupling/nv-2.txt"
    # Phenomenological params (set to None for microscopic-only)
    fine_params = dict(
        baseline=0.6,
        comb_contrast=0.45,
        revival_time=37.0,
        width0_us=7.0,
        T2_ms=0.08,
        T2_exp=1.2,
        amp_taper_alpha=0.0,
        width_slope=0.0,
        revival_chirp=0.0,
        osc_contrast=0.4,  # usually 0 for echo (no extra beating)
        osc_f0=0.0,
        osc_f1=0.0,
        osc_phi0=0.0,
        osc_phi1=0.0,
    )

    # 1) Exact sites (lock configuration)
    fixed_ids = []  # e.g., [12, 45, 78]

    # 2) Or pass a fixed presence mask (same length as candidate sites) after loading once
    fixed_mask = None

    taus, echo, aux = simulate_random_spin_echo_average(
        hyperfine_path=input_file,
        tau_range_us=(0, 100),
        num_spins=None,                    # take all chosen/present sites
        num_realizations=1,                # single trace
        distance_cutoff=5.0,               # Å or whatever your file uses
        Ak_cutoff_kHz=0.0,
        R_NV=np.eye(3),
        fine_params=fine_params,           # set to None for microscopic-only
        abundance_fraction=0.011,          # natural abundance, ignored if fixed_ids/mask used
        rng_seed=4242, run_salt=2000,
        randomize_positions=False,
        selection_mode="uniform",
        ensure_unique_across_realizations=False,
        annotate_from_realization=0,
        keep_nv_orientation=True,
        fixed_site_ids=fixed_ids if fixed_ids else None,
        fixed_presence_mask=fixed_mask,
        reuse_present_mask=True,           # if not fixing sites, draw once and reuse
    )

    plot_echo_with_sites(taus, echo, aux, title="Spin Echo (single NV)")
    
    plt.show()
    
    file_stem = "2025_11_02-19_55_17-johnson_204nv_s3-003c56"
    fit = dm.get_raw_data(file_stem=file_stem)

    keys = fit["unified_keys"]
    def _asdict(p):
        # map the packed vector to a dict with missing-extras handled
        d = {k: None for k in keys}
        if p is None: 
            return d
        for k, v in zip(keys, p + [None]*(len(keys)-len(p))):
            d[k] = v
        return d

    # choose an NV label (e.g., best χ²)
    def _to_float_or_inf(x):
        try:
            if x is None:
                return np.inf
            v = float(x)
            return v if np.isfinite(v) else np.inf
        except Exception:
            return np.inf

    chi_list = fit.get("red_chi2", [])
    chis = np.array([_to_float_or_inf(c) for c in chi_list], dtype=float)

    if chis.size == 0:
        raise RuntimeError("No red_chi2 entries in fit.")

    idx = int(np.argmin(chis))

    # If all χ² are inf, fall back to the first NV with a non-None popt
    if not np.isfinite(chis[idx]):
        popts_list = fit.get("popts", [])
        fallback = next((i for i, p in enumerate(popts_list) if p is not None), None)
        if fallback is None:
            raise RuntimeError("No valid fits found (all χ² invalid and no popts).")
        idx = int(fallback)

    nv_id = int(fit["nv_labels"][idx])
    popt  = fit["popts"][idx]
    par   = _asdict(popt)   # your existing mapper from vector -> dict


    # Robust defaults, then overwrite from the fit if present:
    fine_params = dict(
        baseline       = 0.6 if par["baseline"]       is None else float(par["baseline"]),
        comb_contrast  = 0.40 if par["comb_contrast"] is None else float(par["comb_contrast"]),
        revival_time   = 38.0 if par["revival_time_us"] is None else float(par["revival_time_us"]),
        width0_us      = 6.0 if par["width0_us"]      is None else float(par["width0_us"]),
        T2_ms          = 0.08 if par["T2_ms"]         is None else float(par["T2_ms"]),
        T2_exp         = 1.2 if par["T2_exp"]         is None else float(par["T2_exp"]),
        amp_taper_alpha= 0.0 if par["amp_taper_alpha"]is None else float(par["amp_taper_alpha"]),
        width_slope    = 0.0 if par["width_slope"]    is None else float(par["width_slope"]),
        revival_chirp  = 0.0 if par["revival_chirp"]  is None else float(par["revival_chirp"]),
        # Oscillation block (additive, signed). Convert cycles/μs → MHz 1:1
        osc_add_amp    = 0.0 if par["osc_amp"]        is None else float(par["osc_amp"]),
        osc_f0_MHz     = 0.0 if par["osc_f0"]         is None else float(par["osc_f0"]),
        osc_f1_MHz     = 0.0 if par["osc_f1"]         is None else float(par["osc_f1"]),
        osc_phi0       = 0.0 if par["osc_phi0"]       is None else float(par["osc_phi0"]),
        osc_phi1       = 0.0 if par["osc_phi1"]       is None else float(par["osc_phi1"]),
    )

    
    taus, echo, aux = simulate_random_spin_echo_average(
        hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",  # your file
        tau_range_us=(0, 100),
        num_spins=None,                 # or a number (e.g., 30) to subsample strongest sites
        num_realizations=1,
        distance_cutoff=5.0,
        Ak_cutoff_kHz=0.0,
        R_NV=np.eye(3),
        fine_params=fine_params,        # <-- this brings in your fitted comb+oscillation
        abundance_fraction=0.011,
        rng_seed=4242, run_salt=nv_id,  # salt by NV id to make it reproducible per NV
        randomize_positions=False,
        selection_mode="top_Apar",      # or "uniform"/"distance_weighted"
        reuse_present_mask=True,
    )
    plot_echo_with_sites(taus, echo, aux, title=f"Spin Echo — NV {nv_id}", fine_params=fine_params)
    plt.show()


    plt.show()
