# c13_echo_simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import curve_fit
import json
import os
from pathlib import Path

# --- Optional numba (falls back gracefully) ----------------------------------
try:
    from numba import njit
except Exception:
    def njit(*_args, **_kwargs):
        def wrap(fn):
            return fn
        return wrap
    
def fine_decay(
    tau,
    baseline,
    comb_contrast,
    revival_time,
    width0_us,
    T2_ms,
    T2_exp,
    amp_taper_alpha=None,
    width_slope=None,
    revival_chirp=None,
    osc_contrast=None,
    osc_f0=None,
    osc_f1=None,
    osc_phi0=None,
    osc_phi1=None,
):
    """
    signal(τ) = baseline - envelope(τ) * MOD(τ) * COMB(τ)

    envelope(τ) = exp[-((τ / (1000*T2_ms)) ** T2_exp)]

    COMB(τ) = sum_k  [ 1/(1+k)^amp_taper_alpha ] * exp(-((τ - μ_k)/w_k)^4)
        μ_k = k * revival_time * (1 + k*revival_chirp)
        w_k = width0_us * (1 + k*width_slope)

    MOD(τ) = comb_contrast - osc_contrast * sin^2(π f0 τ + φ0) * sin^2(π f1 τ + φ1)
    """
    # defaults
    if amp_taper_alpha is None: amp_taper_alpha = 0.0
    if width_slope     is None: width_slope     = 0.0
    if revival_chirp   is None: revival_chirp   = 0.0
    if osc_contrast    is None: osc_contrast    = 0.0
    if osc_f0          is None: osc_f0          = 0.0
    if osc_f1          is None: osc_f1          = 0.0
    if osc_phi0        is None: osc_phi0        = 0.0
    if osc_phi1        is None: osc_phi1        = 0.0

    tau = np.asarray(tau, dtype=float).ravel()
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
    if (osc_contrast != 0.0) and (osc_f0 != 0.0 or osc_f1 != 0.0):
        s0 = np.sin(np.pi * osc_f0 * tau + osc_phi0)
        s1 = np.sin(np.pi * osc_f1 * tau + osc_phi1)
        beat = (s0 * s0) * (s1 * s1)
        mod = comb_contrast - osc_contrast * beat
    else:
        mod = comb_contrast

    return baseline - envelope * mod * comb


def fine_decay_fixed_revival(
    tau,
    baseline,
    comb_contrast,
    width0_us,
    T2_ms,
    T2_exp,
    amp_taper_alpha=None,
    width_slope=None,
    revival_chirp=None,
    osc_contrast=None,
    osc_f0=None,
    osc_f1=None,
    osc_phi0=None,
    osc_phi1=None,
    _fixed_rev_time_us=50.0
):
    return fine_decay(
        tau,
        baseline,
        comb_contrast,
        _fixed_rev_time_us,
        width0_us,
        T2_ms,
        T2_exp,
        amp_taper_alpha,
        width_slope,
        revival_chirp,
        osc_contrast,
        osc_f0,
        osc_f1,
        osc_phi0,
        osc_phi1,
    )


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
    """
    NOTE: no overall amplitude factor here (no comb_contrast).
    """
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

        amp_k = 1.0 / ((1.0 + k) ** amp_taper_alpha)  # <- amplitude taper only
        inv_w4 = 1.0 / (w_k ** 4)

        for i in range(n):
            x = tau[i] - mu_k
            out[i] += amp_k * np.exp(- (x * x) * (x * x) * inv_w4)

    return out



# -------------------------------------------------------------------------------------
# Decoherence Envelope
# -------------------------------------------------------------------------------------
def decay_envelope(t, T2, p, rev_times, tau_dev, C0=1.0):
    decay = C0 * np.exp(-(t / T2)**p)
    for tm in rev_times:
        decay *= np.exp(-((t - tm)**2) / (2 * tau_dev**2))
    return decay

def apply_envelope(signal, t, fit_params):
    return signal * fine_decay(t * 1e6, **fit_params)  # t in μs

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
D_NV = 2.87e9  # NV zero-field splitting in Hz
gamma_e = 28e9  # Electron gyromagnetic ratio in Hz/T
gamma_C13 = 10.7e6  # 13C gyromagnetic ratio in Hz/T

# Default magnetic field from user
B_vec_G = np.array([-6.18037755, -18.54113264, -43.26264283])  # in Gauss
B_vec_T = B_vec_G * 1e-4  # convert Gauss → Tesla
B_hat = B_vec_T / np.linalg.norm(B_vec_T)

# -------------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------------
def load_hyperfine_data_txt(path):
    """Load hyperfine data from formatted .txt file (MHz units)."""
    file_path = Path(path)
    with open(file_path, "r") as f:
        lines = f.readlines()

    data_start = next(i for i, line in enumerate(lines) if line.strip().startswith("1 "))
    data = pd.read_csv(
        file_path,
        delim_whitespace=True,
        skiprows=data_start,
        header=None,
        names=["index", "distance", "x", "y", "z", "Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz"],
    )

    tensors = []
    for _, row in data.iterrows():
        A_tensor = np.array([
            [row.Axx, row.Axy, row.Axz],
            [row.Axy, row.Ayy, row.Ayz],
            [row.Axz, row.Ayz, row.Azz]
        ]) * 1e6  # Convert MHz → Hz
        tensors.append(A_tensor)
    return tensors


def transform_tensor_to_NV_frame(A_tensor, R_NV):
    return R_NV @ A_tensor @ R_NV.T


def compute_hyperfine_components(A_tensor, B_unit):
    A_parallel = B_unit @ A_tensor @ B_unit
    perp_proj = np.eye(3) - np.outer(B_unit, B_unit)
    A_perp_tensor = perp_proj @ A_tensor @ perp_proj
    B_perp = np.linalg.norm(A_perp_tensor)
    return A_parallel, B_perp

# -------------------------------------------------------------------------------------
# Echo Simulation Functions
# -------------------------------------------------------------------------------------
def Mk_tau(A, B, tau, omega_L):
    omega = np.sqrt(B**2 + (A - omega_L)**2)
    return 1 - 2 * (B**2 / omega**2) * np.sin(omega * tau / 2)**4


def compute_echo_signal(hyperfine_list, tau_array, B_field_vec=B_vec_T):
    B_mag = np.linalg.norm(B_field_vec)
    B_unit = B_field_vec / B_mag
    omega_L = gamma_C13 * B_mag
    signal = []

    for i, tau in enumerate(tau_array):
        Mk_product = 1.0
        for j, A_tensor in enumerate(hyperfine_list):
            A, B = compute_hyperfine_components(A_tensor, B_unit)
            Mk = Mk_tau(A, B, tau, omega_L)

            # Debug print at midpoint τ and for first few spins
            # if i == len(tau_array) // 2 and j < 5:
            #     print(f"[DEBUG] τ = {tau*1e6:.1f} μs | Spin {j+1}: A = {A/1e3:.2f} kHz, B = {B/1e3:.2f} kHz, Mk = {Mk:.6f}")

            Mk_product *= Mk
        signal.append(0.5 * (1 + Mk_product))
    return np.array(signal)

def apply_revival_gated_modulation_from_avg(avg_signal, taus_sec, fit_params, mask_power=1.0, verbose=True):
    """
    Make oscillations visible only near 13C revivals, while keeping a global stretched-exp decay.
    Robust to zeros/NaNs and parameter edge cases.

    avg_signal : array (Monte-Carlo averaged echo), dimensionless
    taus_sec   : array of times in SECONDS (your simulate_* uses seconds internally)
    fit_params : dict with keys:
        revival_time (us), width0_us (us), amp_taper_alpha, width_slope, revival_chirp,
        T2_ms, T2_exp
    mask_power : >1 sharpens, <1 softens the revival mask
    """
    eps = 1e-12

    # ---- 1) normalize the averaged microscopic signal to start at 1
    avg = np.asarray(avg_signal, dtype=float).copy()
    if not np.all(np.isfinite(avg)):
        avg = np.nan_to_num(avg, nan=1.0, posinf=1.0, neginf=1.0)

    denom = avg[0] if (avg.size > 0 and abs(avg[0]) > eps) else np.max(np.abs(avg))
    if denom is None or denom < eps:
        denom = 1.0
    avg /= denom

    # ---- 2) build revival mask (μs) with the quartic comb; guard against all-zeros
    tau_us = np.asarray(taus_sec, dtype=float) * 1e6
    revival_time  = float(fit_params.get("revival_time", 37.0))
    width0_us     = float(fit_params.get("width0_us", 6.0))
    amp_taper     = float(fit_params.get("amp_taper_alpha", 0.0))
    width_slope   = float(fit_params.get("width_slope", 0.0))
    revival_chirp = float(fit_params.get("revival_chirp", 0.0))

    revival_time  = max(revival_time, 1e-9)
    width0_us     = max(width0_us,   1e-9)

    n_guess = max(1, min(64, int(np.ceil(1.2 * (tau_us.max() if tau_us.size else 0.0) / revival_time)) + 1))

    comb = _comb_quartic_powerlaw(
        tau_us,
        revival_time,
        width0_us,
        amp_taper,
        width_slope,
        revival_chirp,
        n_guess
    )

    comb = np.nan_to_num(comb, nan=0.0, posinf=0.0, neginf=0.0)
    comb_max = np.max(comb) if comb.size else 0.0

    if comb_max > eps:
        mask = comb / comb_max
    else:
        # Fallback: put a single Gaussian at k=round(t/revival_time)
        mask = np.zeros_like(tau_us)
        if tau_us.size:
            ks = np.round(tau_us / revival_time).astype(int)
            # Limit to a few neighbours
            for k in np.unique(ks):
                mu_k = k * revival_time
                w_k  = width0_us
                mask += np.exp(-((tau_us - mu_k)**2) / (2.0 * (w_k**2)))
            mmax = mask.max()
            mask = mask / mmax if mmax > eps else np.ones_like(mask)

    # Optional sharpening/softening
    if mask_power != 1.0:
        mask = np.power(np.clip(mask, 0.0, 1.0), mask_power)

    # ---- 3) gate deviations-from-1 by the mask (0 = collapse flat, 1 = keep avg)
    avg_gated = 1.0 - (1.0 - avg) * mask

    # ---- 4) global stretched-exponential envelope (no comb here)
    T2_ms  = float(fit_params.get("T2_ms", 0.1))
    T2_exp = float(fit_params.get("T2_exp", 2.0))
    T2_us  = max(1000.0 * T2_ms, 1e-9)

    env = np.exp(-((tau_us / T2_us) ** T2_exp))
    env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)

    final = 1.0 - (1.0 - avg_gated) * env
    final = np.clip(final, -1e3, 1e3)  # keep sane

    if verbose:
        print(f"[REV-GATE] avg range: {avg.min():.4f}..{avg.max():.4f}, "
              f"mask max: {mask.max():.3f}, env end: {env[-1]:.4e}, final range: {final.min():.4f}..{final.max():.4f}")
    return final


# -------------------------------------------------------------------------------------
# Monte Carlo Echo  over Random Spin Environments
# ------------------------------------------------------------------------------------

# Args you can add to the function signature (defaults shown):
def _rand_rotation_matrix(rng):
    """Random 3D rotation (Haar) via QR."""
    M = rng.normal(size=(3,3))
    Q, R = np.linalg.qr(M)
    Q *= np.sign(np.diag(R)).prod()
    return Q

def random_rotate_tensor_and_pos(A_tensor, pos_vec, rng):
    """
    Apply the same random rotation to the hyperfine tensor and the lattice position.
    Keeps the physical relationship while 'randomizing' orientation.
    """
    R = _rand_rotation_matrix(rng)
    return (R @ A_tensor @ R.T, R @ pos_vec)

def _choose_sites(rng, present_sites, num_spins, selection_mode="uniform"):
    # Always randomize order first as a tie-breaker
    present_sites = list(present_sites)
    rng.shuffle(present_sites)

    if num_spins <= 0 or len(present_sites) <= num_spins:
        return present_sites

    if selection_mode == "uniform":
        idx = rng.choice(len(present_sites), size=num_spins, replace=False)

    elif selection_mode == "distance_weighted":
        r = np.array([max(s["dist"], 1e-9) for s in present_sites], float)
        w = (1.0 / (r**3)); w /= w.sum()
        idx = rng.choice(len(present_sites), size=num_spins, replace=False, p=w)

    elif selection_mode == "Apar_weighted":
        apar = np.array([abs(s["Apar_Hz"]) for s in present_sites], float)
        if apar.sum() <= 0:
            idx = rng.choice(len(present_sites), size=num_spins, replace=False)
        else:
            w = apar / apar.sum()
            idx = rng.choice(len(present_sites), size=num_spins, replace=False, p=w)

    elif selection_mode == "top_Apar":
        # Still deterministic, but shuffle above makes ties random
        order = np.argsort([-abs(s["Apar_Hz"]) for s in present_sites])
        idx = order[:num_spins]

    else:
        idx = rng.choice(len(present_sites), size=num_spins, replace=False)

    return [present_sites[i] for i in idx]


# --- RNG helpers -------------------------------------------------------------
def _spawn_streams(seed, num_streams, run_salt=None):
    """
    Returns a list of independent np.random.Generator objects.
    Behavior:
      - seed=None        -> non-deterministic (new draw each call)
      - seed=<int>       -> deterministic
      - seed=<int> + run_salt=<int/None>
           * if run_salt provided, different streams across calls reproducibly
    """
    if seed is None:
        ss = np.random.SeedSequence()  # draws from OS entropy
    else:
        # Mix in an optional run_salt to change streams across calls while staying reproducible
        if run_salt is None:
            ss = np.random.SeedSequence(seed)
        else:
            # spawn_key must be small ints; mask is just to be safe
            run_salt = int(run_salt) & 0xFFFFFFFF
            ss = np.random.SeedSequence(seed, spawn_key=[run_salt])
    child_seqs = ss.spawn(num_streams)
    return [np.random.default_rng(cs) for cs in child_seqs]

# --- main simulation ----------------------------------------------------------
# def simulate_random_spin_echo_average(
#     hyperfine_path,
#     tau_range_us,
#     num_spins=30,
#     num_realizations=10,
#     distance_cutoff=None,
#     Ak_cutoff_kHz=0,
#     R_NV=np.eye(3),
#     fit_params=None,
#     abundance_fraction=0.011,
#     seed=None,                 # <- default None = new sites each call
#     run_salt=None,             # <- optional extra variability while reproducible
#     randomize_positions=True,
#     selection_mode="uniform",              # <- NEW
#     ensure_unique_across_realizations=True,  # <- NEW
#     annotate_from_realization=0,
# ):
#     """
#     Returns:
#       taus_us, avg_signal, aux
#       aux = {
#         "positions": (N,3) positions from the annotated realization (NV frame, after rotation if used),
#         "site_info": list of dicts for annotated realization (site_id, Apar_kHz, r_nm),
#         "revivals_us": array of k*revival_time for plotting vertical lines,
#         "picked_ids_per_realization": list of lists of site_ids picked each realization
#       }
#     """
#     taus = np.linspace(*tau_range_us, num=300) * 1e-6  # seconds

#     rng_streams = _spawn_streams(seed, num_realizations, run_salt=run_salt)
#     # ---- Load and parse file ----
#     file_path = Path(hyperfine_path)
#     with open(file_path, "r") as f:
#         lines = f.readlines()
#     data_start = next(i for i, line in enumerate(lines) if line.strip().startswith("1 "))
#     df = pd.read_csv(
#         file_path,
#         delim_whitespace=True,
#         skiprows=data_start,
#         header=None,
#         names=["index", "distance", "x", "y", "z", "Axx", "Ayy", "Azz", "Axy", "Axz", "Ayz"],
#     )
#     if distance_cutoff is not None:
#         df = df[df["distance"] < distance_cutoff]

#     # ---- Build candidate site list (NV frame) ----
#     sites = []
#     for _, row in df.iterrows():
#         A = np.array([
#             [row.Axx, row.Axy, row.Axz],
#             [row.Axy, row.Ayy, row.Ayz],
#             [row.Axz, row.Ayz, row.Azz],
#         ], dtype=float) * 1e6  # MHz → Hz
#         A_nv = R_NV @ A @ R_NV.T
#         # Apparent A_parallel for *canonical* B_hat (no randomization yet)
#         A_par, _ = compute_hyperfine_components(A_nv, B_hat)
#         if np.abs(A_par) > Ak_cutoff_kHz * 1e3:
#             sites.append({
#                 "site_id": int(row["index"]),                               # site id from file column "index"
#                 "A0": A_nv,                                                 # tensor in NV frame
#                 "pos0": np.array([row.x, row.y, row.z], dtype=float),       # lattice pos (nm or Å, whatever file uses)
#                 "dist": float(row.distance),                                 # same units as file
#                 "Apar_Hz": float(A_par),
#             })

#     print(f"[INFO] Filtered tensor pool: {len(sites)} spins")
#     print(f"[INFO] Abundance per realization: {abundance_fraction*100:.2f}%")

#     if not sites:
#         taus_us = taus * 1e6
#         flat = np.ones_like(taus)
#         if fit_params:
#             flat = apply_revival_gated_modulation_from_avg(flat, taus, fit_params, mask_power=2.0, verbose=False)
#         return taus_us, flat, {"positions": None, "site_info": [], "revivals_us": None, "picked_ids_per_realization": []}



#     # after you build `sites = [...]`
#     N_candidates = len(sites)
#     p = float(abundance_fraction)

#     # Total expected and variance
#     expected_present_mean = N_candidates * p
#     expected_present_std  = (N_candidates * p * (1.0 - p))**0.5
#     prob_none = (1.0 - p)**N_candidates
#     prob_at_least_one = 1.0 - prob_none

#     # Optional: radial bin stats (distance histogram)
#     # Choose bin edges (in the same units as df.distance, e.g., nm)
#     bin_edges = np.linspace(0, float(distance_cutoff or max(s["dist"] for s in sites)), 8)
#     dists = np.array([s["dist"] for s in sites], float)
#     bin_ids = np.digitize(dists, bin_edges, right=True)

#     sites_per_bin = np.array([(bin_ids == i).sum() for i in range(1, len(bin_edges))], int)
#     exp_per_bin   = sites_per_bin * p
#     std_per_bin   = np.sqrt(sites_per_bin * p * (1.0 - p))
    
#     # Keep per-realization counts too (filled later)
#     present_counts = []
#     chosen_counts  = []
#     # History of used sites if we want cross-realization uniqueness
#     all_signals = []
#     picked_ids_per_realization = []
#     anno_positions = None
#     anno_site_info = None
#     anno_rev_times = None
#     used_site_ids = set()

#     for r in range(num_realizations):

#         rng_r = rng_streams[r]

#         present_mask = rng_r.random(len(sites)) < abundance_fraction
#         present_idxs = np.flatnonzero(present_mask)
#         if present_idxs.size == 0:
#             all_signals.append(np.ones_like(taus))
#             picked_ids_per_realization.append([])
#             continue

#         present_sites = [sites[i] for i in present_idxs]
    
#         if ensure_unique_across_realizations:
#             filtered = [s for s in present_sites if s["site_id"] not in used_site_ids]
#             if len(filtered) >= max(1, num_spins):
#                 present_sites = filtered

#         chosen_sites = _choose_sites(rng_r, present_sites, num_spins, selection_mode=selection_mode)

#         present_counts.append(int(present_idxs.size))
#         chosen_counts.append(len(chosen_sites))

#         picked_ids = [s["site_id"] for s in chosen_sites]
#         picked_ids_per_realization.append(picked_ids)
#         if ensure_unique_across_realizations:
#             used_site_ids.update(picked_ids)

#         tensors, pos_list, info_list = [], [], []
#         for s in chosen_sites:
#             if randomize_positions:
#                 A_use, pos_use = random_rotate_tensor_and_pos(s["A0"], s["pos0"], rng_r)
#             else:
#                 A_use, pos_use = s["A0"], s["pos0"]
#             A_par_now, _ = compute_hyperfine_components(A_use, B_hat)
#             tensors.append(A_use)
#             pos_list.append(pos_use)
#             info_list.append({
#                 "site_id": s["site_id"],
#                 "pos": pos_use.copy(),
#                 "r": float(np.linalg.norm(pos_use)),
#                 "Apar_kHz": float(abs(A_par_now) / 1e3),
#             })

#         signal = compute_echo_signal(tensors, taus)
#         all_signals.append(signal)

#         if r == annotate_from_realization:
#             anno_positions = np.array(pos_list) if pos_list else None
#             anno_site_info = info_list
#             if fit_params is not None and "revival_time" in fit_params:
#                 revT_us = float(fit_params["revival_time"])
#                 kmax = int(np.ceil((taus.max()*1e6) / revT_us))
#                 anno_rev_times = np.arange(0, kmax+1) * revT_us

#     avg_signal = np.mean(all_signals, axis=0)

#     if fit_params:
#         avg_signal = apply_revival_gated_modulation_from_avg(
#             avg_signal, taus, fit_params, mask_power=2.0, verbose=False
#         )

#     # ----- gather metadata/stats for reporting -----
#     meta = {
#         "seed": seed,
#         "selection_mode": selection_mode,
#         "abundance_fraction": float(abundance_fraction),
#         "num_realizations": int(num_realizations),
#         "num_spins_target": int(num_spins),
#         "total_available_sites": int(len(sites)),
#         "per_realization": [],   # list of dicts with counts each realization
#         "available_distances": [float(s["dist"]) for s in sites],  # units = your input file
#     }

#     # We already collected these:
#     # - picked_ids_per_realization
#     meta["picked_ids_per_realization"] = picked_ids_per_realization

#     # Build per-realization counts
#     # (You can populate this inside the loop too, but here’s a quick summary
#     #  from the existing structures)
#     for picked in picked_ids_per_realization:
#         meta["per_realization"].append({
#             "chosen_count": len(picked)
#         })

#     # For the annotated realization, include distances & IDs of plotted spins
#     if anno_site_info is not None:
#         meta["annotated"] = {
#             "chosen_ids": [it["site_id"] for it in anno_site_info],
#             "chosen_Apar_kHz": [it["Apar_kHz"] for it in anno_site_info],
#             "chosen_r": [it["r"] for it in anno_site_info],  # norm of rotated pos
#         }
#     else:
#         meta["annotated"] = {"chosen_ids": [], "chosen_Apar_kHz": [], "chosen_r": []}

#     return taus * 1e6, avg_signal, {
#         "positions": anno_positions,
#         "site_info": anno_site_info if anno_site_info is not None else [],
#         "revivals_us": anno_rev_times,
#         "picked_ids_per_realization": picked_ids_per_realization,
#         "stats": {
#             "N_candidates": N_candidates,
#             "abundance_fraction": p,
#             "expected_present_mean": expected_present_mean,
#             "expected_present_std":  expected_present_std,
#             "prob_none": prob_none,
#             "prob_at_least_one": prob_at_least_one,
#             "present_counts": present_counts,
#             "chosen_counts":  chosen_counts, 
#             "bin_edges":      bin_edges,
#             "sites_per_bin":  sites_per_bin,
#             "exp_per_bin":    exp_per_bin,
#             "std_per_bin":    std_per_bin,
#         }
#     }



def simulate_random_spin_echo_average(
    hyperfine_path,
    tau_range_us,
    num_spins=30,
    num_realizations=10,
    distance_cutoff=None,
    Ak_cutoff_kHz=0,
    R_NV=np.eye(3),
    fit_params=None,
    abundance_fraction=0.011,
    seed=None,                 # default None = new sites each call
    run_salt=None,             # extra variability while reproducible
    randomize_positions=True,  # only used if keep_nv_orientation=False
    selection_mode="uniform",
    ensure_unique_across_realizations=True,
    annotate_from_realization=0,
    keep_nv_orientation=True,  # <<< NEW: keep NV orientation fixed
):
    """
    Returns:
      taus_us, avg_signal, aux
      aux = {
        "positions": (N,3) pos from the annotated realization (NV frame; no rotation if keep_nv_orientation=True),
        "site_info": list of dicts for annotated realization (site_id, Apar_kHz, r_norm),
        "revivals_us": array of k*revival_time (μs),
        "picked_ids_per_realization": list[list[int]],
        "stats": {...}
    }
    """
    taus = np.linspace(*tau_range_us, num=300) * 1e-6  # seconds
    rng_streams = _spawn_streams(seed, num_realizations, run_salt=run_salt)

    # --- rotate B into NV frame ONCE ---
    B_vec_NV = R_NV @ B_vec_T
    B_hat_NV = B_vec_NV / np.linalg.norm(B_vec_NV)

    # ---- Load and parse file ----
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

    # ---- Build candidate site list (rotated once into the NV frame) ----
    sites = []
    for _, row in df.iterrows():
        A = np.array([[row.Axx, row.Axy, row.Axz],
                      [row.Axy, row.Ayy, row.Ayz],
                      [row.Axz, row.Ayz, row.Azz]], dtype=float) * 1e6
        A_nv = R_NV @ A @ R_NV.T
        A_par, _ = compute_hyperfine_components(A_nv, B_hat_NV)

        if np.abs(A_par) > Ak_cutoff_kHz * 1e3:
            pos_crystal = np.array([row.x, row.y, row.z], dtype=float)
            pos_nv      = R_NV @ pos_crystal
            sites.append({
                "site_id": int(row["index"]),
                "A0": A_nv,
                "pos0": pos_nv,                 # <— rotated position
                "dist": float(row.distance),
                "Apar_Hz": float(A_par),
            })

    print(f"[INFO] Filtered tensor pool: {len(sites)} spins")
    print(f"[INFO] Abundance per realization: {abundance_fraction*100:.2f}%")

    if not sites:
        taus_us = taus * 1e6
        flat = np.ones_like(taus)
        if fit_params:
            flat = apply_revival_gated_modulation_from_avg(flat, taus, fit_params, mask_power=2.0, verbose=False)
        return taus_us, flat, {
            "positions": None, "site_info": [], "revivals_us": None,
            "picked_ids_per_realization": [], "stats": {}
        }

    # ---- Pre-compute simple stats for the left-panel textbox/inset ----
    N_candidates = len(sites)
    p = float(abundance_fraction)
    expected_present_mean = N_candidates * p
    expected_present_std  = (N_candidates * p * (1.0 - p))**0.5
    prob_none = (1.0 - p)**N_candidates
    prob_at_least_one = 1.0 - prob_none
    # radial stats
    max_r = float(distance_cutoff or max(s["dist"] for s in sites))
    bin_edges = np.linspace(0, max_r, 8)
    dists = np.array([s["dist"] for s in sites], float)
    bin_ids = np.digitize(dists, bin_edges, right=True)
    sites_per_bin = np.array([(bin_ids == i).sum() for i in range(1, len(bin_edges))], int)
    exp_per_bin   = sites_per_bin * p
    std_per_bin   = np.sqrt(sites_per_bin * p * (1.0 - p))

    # ---- Monte Carlo over realizations ----
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

        # Bernoulli thinning for natural abundance
        present_mask = rng_r.random(len(sites)) < abundance_fraction
        present_idxs = np.flatnonzero(present_mask)
        present_counts.append(int(present_idxs.size))

        if present_idxs.size == 0:
            all_signals.append(np.ones_like(taus))
            picked_ids_per_realization.append([])
            continue

        present_sites = [sites[i] for i in present_idxs]

        # Optional cross-realization uniqueness
        if ensure_unique_across_realizations:
            filtered = [s for s in present_sites if s["site_id"] not in used_site_ids]
            if len(filtered) >= max(1, num_spins):
                present_sites = filtered

        # Choose the K sites according to selection_mode
        chosen_sites = _choose_sites(rng_r, present_sites, num_spins, selection_mode=selection_mode)
        chosen_counts.append(len(chosen_sites))

        picked_ids = [s["site_id"] for s in chosen_sites]
        picked_ids_per_realization.append(picked_ids)
        if ensure_unique_across_realizations:
            used_site_ids.update(picked_ids)

        # Build tensors/positions:
        # If keep_nv_orientation=True -> use A0,pos0 directly (no random rotations).
        # Else fall back to your previous behavior (optional randomization).
        tensors, pos_list, info_list = [], [], []
        for s in chosen_sites:
            if keep_nv_orientation:
                A_use, pos_use = s["A0"], s["pos0"]
            else:
                if randomize_positions:
                    A_use, pos_use = random_rotate_tensor_and_pos(s["A0"], s["pos0"], rng_r)
                else:
                    A_use, pos_use = s["A0"], s["pos0"]

            A_par_now, _ = compute_hyperfine_components(A_use, B_hat_NV)
            tensors.append(A_use)
            pos_list.append(pos_use)
            info_list.append({
                "site_id": s["site_id"],
                "pos": pos_use.copy(),
                "r": float(np.linalg.norm(pos_use)),
                "Apar_kHz": float(abs(A_par_now) / 1e3),
            })

        signal = compute_echo_signal(tensors, taus, B_field_vec=B_vec_NV)
        all_signals.append(signal)

        # Keep annotation from one realization (unchanged)
        if r == annotate_from_realization:
            anno_positions = np.array(pos_list) if pos_list else None
            anno_site_info = info_list
            if fit_params is not None and "revival_time" in fit_params:
                revT_us = float(fit_params["revival_time"])
                kmax = int(np.ceil((taus.max()*1e6) / revT_us))
                anno_rev_times = np.arange(0, kmax+1) * revT_us

    # Average across realizations
    avg_signal = np.mean(all_signals, axis=0)

    # Revival-gated envelope/modulation (your helper)
    if fit_params:
        avg_signal = apply_revival_gated_modulation_from_avg(
            avg_signal, taus, fit_params, mask_power=2.0, verbose=False
        )

    # Package stats
    stats = {
        "N_candidates": N_candidates,
        "abundance_fraction": p,
        "expected_present_mean": expected_present_mean,
        "expected_present_std":  expected_present_std,
        "prob_none": prob_none,
        "prob_at_least_one": prob_at_least_one,
        "present_counts": present_counts,
        "chosen_counts":  chosen_counts,
        "bin_edges":      bin_edges,
        "sites_per_bin":  sites_per_bin,
        "exp_per_bin":    exp_per_bin,
        "std_per_bin":    std_per_bin,
    }

    return taus * 1e6, avg_signal, {
        "positions": anno_positions,
        "site_info": anno_site_info if anno_site_info is not None else [],
        "revivals_us": anno_rev_times,
        "picked_ids_per_realization": picked_ids_per_realization,
        "stats": stats,
    }

def plot_echo_with_sites(taus_us, echo, aux, title="Monte Carlo Averaged Spin Echo"):
    """
    Panels:
      [0] Echo vs tau (μs) with revival lines + summary stats
      [1] Expected # of 13C per radial bin (bar chart, optional)
      [2] 3D positions of annotated 13C sites + NV at origin
    """
    fig = plt.figure(figsize=(15, 4.8))

    # ---------------- Echo panel ----------------
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(taus_us, echo, lw=1.8)
    ax0.set_xlabel("Tau (μs)")
    ax0.set_ylabel("Coherence")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)

    revs = aux.get("revivals_us", None)
    if revs is not None:
        for t in np.atleast_1d(revs):
            ax0.axvline(t, ls="--", lw=0.7, alpha=0.35)

    # Stats box
    stats = aux.get("stats", {}) or {}
    Ncand = stats.get("N_candidates")
    p     = stats.get("abundance_fraction")
    mu    = stats.get("expected_present_mean")
    sd    = stats.get("expected_present_std")
    p0    = stats.get("prob_none")
    pge1  = stats.get("prob_at_least_one")
    present_counts = stats.get("present_counts", [])
    chosen_counts  = stats.get("chosen_counts", [])

    lines = []
    if Ncand is not None: lines.append(f"Candidates (within cutoff): {Ncand}")
    if p is not None:     lines.append(f"Abundance p: {100.0*p:.2f}%")
    if mu is not None:    lines.append(f"E[#¹³C present]: {mu:.2f} ± {sd:.2f}")
    if p0 is not None:    lines.append(f"P(0 present): {p0:.3f}")
    if pge1 is not None:  lines.append(f"P(≥1 present): {pge1:.3f}")
    # if present_counts:
    #     lines.append(f"Present per realization (min/med/max): "
    #                  f"{min(present_counts)}/{int(np.median(present_counts))}/{max(present_counts)}")
    # if chosen_counts:
    #     lines.append(f"Chosen per realization (min/med/max): "
    #                  f"{min(chosen_counts)}/{int(np.median(chosen_counts))}/{max(chosen_counts)}")

    if lines:
        ax0.text(0.5, 0.02, "\n".join(lines), transform=ax0.transAxes, fontsize=9,
                 va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, lw=0.5))

    # ---------------- Expected-per-bin panel ----------------
    ax1 = fig.add_subplot(1, 3, 2)
    bin_edges = stats.get("bin_edges", None)
    exp_bin   = stats.get("exp_per_bin", None)
    std_bin   = stats.get("std_per_bin", None)
    sites_bin = stats.get("sites_per_bin", None)

    if bin_edges is not None and exp_bin is not None and len(exp_bin) > 0:
        centers = 0.5 * (np.asarray(bin_edges[:-1]) + np.asarray(bin_edges[1:]))
        widths  = (np.asarray(bin_edges[1:]) - np.asarray(bin_edges[:-1])) * 0.85
        ax1.bar(centers, exp_bin, width=widths, align="center", alpha=0.9)
        if std_bin is not None:
            ax1.errorbar(centers, exp_bin, yerr=std_bin, fmt="none", lw=1)
        ax1.set_title("Expected #¹³C per radial bin")
        ax1.set_xlabel("r (Å)")
        ax1.set_ylabel("E[count]")
        ax1.grid(True, alpha=0.2)

        # Optional overlay: # candidate sites per bin (secondary axis)
        if sites_bin is not None and len(sites_bin) == len(exp_bin):
            ax1b = ax1.twinx()
            ax1b.plot(centers, sites_bin, lw=1.2)
            ax1b.set_ylabel("# candidate sites", rotation=270, labelpad=14)
            ax1b.grid(False)
    else:
        ax1.text(0.5, 0.5, "No expected-bin stats provided",
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()

    # ---------------- 3D positions panel ----------------
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    pos  = aux.get("positions", None)
    info = aux.get("site_info", [])
    if pos is not None and len(pos) > 0:
        ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=38, depthshade=True)
        for pnt, meta in zip(pos, info):
            ax2.text(pnt[0], pnt[1], pnt[2],
                     f'{meta["site_id"]}\n|A//|={meta["Apar_kHz"]:.0f} kHz',
                     fontsize=8, ha="left", va="bottom")
        # NV at origin
        ax2.scatter([0], [0], [0], s=70, marker="*")
        ax2.text(0, 0, 0, "NV", fontsize=9, ha="right", va="top")
        ax2.set_title("¹³C positions")
        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    else:
        ax2.text(0.5, 0.5, 0.5, "No ¹³C kept in annotated realization",
                 transform=ax2.transAxes, ha="center", va="center")
        ax2.set_axis_off()

    plt.tight_layout()
    return fig


# -------------------------------------------------------------------------------------
# Plotting Example
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    from utils import kplotlib as kpl
    kpl.init_kplotlib()
    input_file = r"analysis/nv_hyperfine_coupling/nv-2.txt"
    fit_params = {
    "baseline": 1.0,
    "comb_contrast": 0.6,
    "revival_time": 37.0,
    "width0_us": 6.0,
    "T2_ms": 0.08,
    "T2_exp": 1.0,
    "amp_taper_alpha": 0.0,
    "width_slope": 0.00,
    "revival_chirp": 0.0,
    "osc_contrast": 0.0,
    "osc_f0": 0.0,
    "osc_f1": 0.0,
    "osc_phi0": 0.0,
    "osc_phi1": 0.0,
    }

    taus, avg, aux = simulate_random_spin_echo_average(
        hyperfine_path=input_file,
        tau_range_us=(0, 100),
        num_spins=20,
        num_realizations=1,
        distance_cutoff=8.0,
        Ak_cutoff_kHz=0,
        abundance_fraction=0.011,
        seed=2, run_salt=17,
        randomize_positions=False,
        selection_mode="uniform",
        ensure_unique_across_realizations=True,
        fit_params = fit_params, 
        annotate_from_realization=0,
    )
    
    fig = plot_echo_with_sites(taus, avg, aux)
    # -----------------------------------------------------------
    # Optional: Visualize Mk(tau) for first 5 strongest couplings
    # -----------------------------------------------------------
    # data = load_hyperfine_data_txt(input_file)
    # filtered = []
    # for A_tensor in data:
    #     A_tensor = transform_tensor_to_NV_frame(A_tensor, np.eye(3))
    #     Ak, _ = compute_hyperfine_components(A_tensor, B_hat)
    #     filtered.append((np.abs(Ak), A_tensor))

    # # Sort by |A_parallel| and keep top 5
    # filtered.sort(key=lambda tup: -tup[0])
    # strongest = [A_tensor for _, A_tensor in filtered[:5]]

    # taus_plot = np.linspace(*tau_range_us, num=300) * 1e-6
    # omega_L = gamma_C13 * np.linalg.norm(B_vec_T)

    # plt.figure()
    # for i, A_tensor in enumerate(strongest):
    #     A, B = compute_hyperfine_components(A_tensor, B_hat)
    #     Mk = [Mk_tau(A, B, tau, omega_L) for tau in taus_plot]
    #     plt.plot(taus_plot * 1e6, Mk, label=f"Spin {i+1}")
    # plt.xlabel("Tau (μs)")
    # plt.ylabel("M_k(τ)")
    # plt.title("Individual Spin Echo Contributions")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    kpl.show(block=True)
