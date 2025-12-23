import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz, factorial
from matplotlib.colors import LinearSegmentedColormap

# -------------------- main knobs --------------------
E0 = 1.95  # eV, ZPL center
S = 4.0  # Huang–Rhys factor (sets mean of Poisson ladder)
dE_meV = 7.0  # meV spacing between phonon replicas
Nmax = 4  # max phonon order included (>= ~S + 4*sqrt(S))
n_frames = 30
n_E = 80
E_span_meV_low, E_span_meV_up = 50, 20

# Shot-noise / randomness control for per-frame amplitudes
counts_per_frame = 1200  # larger -> lower relative shot noise
intensity_scale = 1.0  # global scale (unitless)

# Jitter (energy drift of each order over time)
jitter_sin_meV = 1.5
jitter_walk_meV = 0.2
jitter_white_meV = 0.6

# Voigt widths (meV) and broadening growth with time/order
sigma0_meV, gamma0_meV = 1.1, 0.4  # base Gaussian/Lorentzian at n=0, frame 0
alpha_T, beta_T = 0.30, 0.40  # time growth (0->1 across frames)
alpha_n, beta_n = 0.12, 0.15  # order growth per phonon number

# Background
white_noise_std = 1.5
drift_amp = 0.0
# ----------------------------------------------------

rng = np.random.default_rng(123)
# rng = np.random.default_rng(134003)

# Units/axes
to_eV = 1e-3
dE_eV = dE_meV * to_eV
E = np.linspace(E0 - E_span_meV_low * to_eV, E0 + E_span_meV_up * to_eV, n_E)

# Phonon orders and *deterministic* Huang–Rhys weights
n = np.arange(Nmax + 1)
w = np.exp(-S) * (S**n) / factorial(n)  # sums to ~1 if Nmax large enough

# Base peak positions: Stokes ladder (red side), n=0 is ZPL at E0
base_positions = E0 - n * dE_eV

# --- Emphasize specific phonon orders (3rd–4th from high-energy side: n≈2–3)
target_centers = [2]  # peaks between n=2 and n=3
order_width = 0.9  # tightness around target orders (0.5 tight ←→ 1.5 broad)

order_gain = np.array(
    [
        max(np.exp(-((i - c) ** 2) / (2 * order_width**2)) for c in target_centers)
        for i in n
    ]
)  # 0..1 per order; highest at n≈2–3

# Random-walk state for each order (in meV)
rw_state_meV = np.zeros_like(n, dtype=float)


def voigt(E, center, sigma_eV, gamma_eV):
    z = ((E - center) + 1j * gamma_eV) / (sigma_eV * np.sqrt(2))
    return np.real(wofz(z)) / (sigma_eV * np.sqrt(2 * np.pi))


spectra = []

for f in range(n_frames):
    spec = np.zeros_like(E)
    tfrac = f / (n_frames - 1 + 1e-12)
    phase = 2 * np.pi * f / 40.0

    # --- Random Poisson amplitudes per order from Huang–Rhys weights
    # # Expected counts proportional to w_n
    # lam = counts_per_frame * w
    # A = rng.poisson(lam) * intensity_scale  # random per frame
    # --- Frame-dependent intensity (photobleaching / power decay)
    # tfrac is already defined above
    # --- Put this inside the frame loop, before sampling A ---
    # Rise-then-decay envelope: peaks at early frames (~4), then rolls off
    tau_rise = 2.0  # frames to rise
    tau_decay = 10.0  # frames to decay
    floor_frac = 0.25  # late-time floor (0..1 of initial max)

    # Use f (0..n_frames-1), not tfrac, so the peak sits at small integer frames
    rise = 1.0 - np.exp(-f / tau_rise)
    decay = np.exp(-f / tau_decay)
    frame_gain = floor_frac + (1.0 - floor_frac) * (rise * decay)

    # small frame-to-frame fluctuation (keep tiny)
    frame_gain *= max(0.0, 1 + rng.normal(0, 0.01))

    lam = counts_per_frame * frame_gain * w
    A = rng.poisson(lam) * intensity_scale

    # Order-local amplitude mixing (bleed to neighbors), strongest at n≈2–3
    mix_base = 0.15  # 0=no mix; 0.1–0.3 is modest
    A_mixed = A.copy()
    for i_ord in range(1, len(A) - 1):
        eps_i = mix_base * (0.5 + 0.5 * order_gain[i_ord])  # 0.5..1 × mix_base
        leak = eps_i * A[i_ord]
        A_mixed[i_ord] -= leak
        A_mixed[i_ord - 1] += 0.5 * leak
        A_mixed[i_ord + 1] += 0.5 * leak
    A = A_mixed

    for i, (center0, Ai) in enumerate(zip(base_positions, A)):

        # # energy jitter (meV): sinusoid + white + random walk
        # sin_term = jitter_sin_meV * np.sin(phase + i * 0.9)
        # rw_state_meV[i] += rng.normal(0, jitter_walk_meV)
        # white_term = rng.normal(0, jitter_white_meV)
        # jitter_eV = (sin_term + rw_state_meV[i] + white_term) * to_eV

        # center = center0 + jitter_eV

        # # order- and time-dependent Voigt widths
        # sigma_meV = sigma0_meV * (1 + alpha_T * tfrac) * (1 + alpha_n * i)
        # gamma_meV = gamma0_meV * (1 + beta_T * tfrac) * (1 + beta_n * i)
        # sigma_eV = sigma_meV * to_eV
        # gamma_eV = gamma_meV * to_eV
        # Emphasis for this order (0..1, peaks at n≈2–3)
        og = order_gain[i]

        # --- Jitter (energy wander) scaled by order emphasis
        k_jitter = 1.2  # increase to make n≈2–3 wobblier
        sin_term = (jitter_sin_meV * (1 + k_jitter * og)) * np.sin(phase + i * 0.9)
        rw_state_meV[i] += rng.normal(0, jitter_walk_meV * (1 + k_jitter * og))
        white_term = rng.normal(0, jitter_white_meV * (1 + k_jitter * og))
        jitter_eV = (sin_term + rw_state_meV[i] + white_term) * to_eV
        center = center0 + jitter_eV

        # --- Voigt widths: keep mild time growth, add order-peaked growth
        alpha_T_eff, beta_T_eff = 0.05, 0.06  # smaller time growth than before
        k_sigma, k_gamma = 0.8, 0.8  # how much broader at target orders

        sigma_meV = (
            sigma0_meV
            * (1 + alpha_T_eff * tfrac)
            * (1 + alpha_n * i)
            * (1 + k_sigma * og)
        )
        gamma_meV = (
            gamma0_meV
            * (1 + beta_T_eff * tfrac)
            * (1 + beta_n * i)
            * (1 + k_gamma * og)
        )
        sigma_eV = sigma_meV * to_eV
        gamma_eV = gamma_meV * to_eV

        # add this order's contribution
        spec += Ai * voigt(E, center, sigma_eV, gamma_eV)

    # frame-level multiplicative jitter (laser power drift)
    spec *= max(0.0, 1 + rng.normal(0, 0.08))

    # additive noise + slow baseline drift
    spec += rng.normal(0, white_noise_std, size=spec.shape)
    drift = drift_amp * np.sin(
        2 * np.pi * (E - E.min()) / (E.max() - E.min()) * rng.uniform(0.6, 1.5)
    )
    spec += drift

    # clip & normalize per frame
    spec = np.clip(spec, 0, None)
    # spec /= spec.max() + 1e-12
    spectra.append(spec)

spectra = np.array(spectra)
spectra /= spectra.max() + 1e-12

# ----- save outputs -----
out_dir = "C:\\Users\\Saroj Chand\\OneDrive - CUNY\\EXPERIMENTS\\20220329_WSe2_dark_exciton_strain_pockets\\Figure_for_papers\\sim_spectra_txt"
os.makedirs(out_dir, exist_ok=True)

# 1) Save the common energy axis (once)
np.savetxt(
    os.path.join(out_dir, "energy_eV.txt"),
    E,
    fmt="%.9f",
    delimiter="\t",
    header="Energy (eV)",
)

# 2) Save each frame's spectrum as its own file
#    (uses zero-padded indices: spectrum_000.txt, spectrum_001.txt, ...)
for f_idx, spec in enumerate(spectra):
    np.savetxt(
        os.path.join(out_dir, f"spectrum_{f_idx:03d}.txt"),
        spec,
        fmt="%.9f",
        delimiter="\t",
        header=f"Normalized intensity (frame {f_idx})",
    )

# Define colors: black → red → white
colors = ["black", "red", "white"]
# Create the colormap
cmap_name = "BlackWhiteRed"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
# ---------- plots ----------
plt.figure(figsize=(4, 5), dpi=150)
extent = [E.min(), E.max(), 0, n_frames]
plt.imshow(spectra, aspect="auto", origin="lower", extent=extent, cmap=custom_cmap)
plt.xlabel("Energy (eV)", fontsize=13)
# plt.ylabel("Time (frames)", fontsize=13)
# plt.title(
#     f"Phonon ladder (Huang–Rhys S={S:.1f}, Poisson per-frame)",
#     fontsize=9,
# )
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.yticks([])
cbar = plt.colorbar()
cbar.set_label("Normalized intensity", fontsize=13)
plt.tight_layout()
plt.show()


import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import wofz, factorial
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap


# ===================== USER INPUTS =====================
exp_data_path = r"analysis/franck_condon/Table_E.txt"
# Optional: provide a real energy axis file (1 column, length = n_pixels)
energy_axis_path = r"analysis/franck_condon/Table0.txt"

# Physics anchor
E0 = 1.95  # eV, define this as the ZPL energy for the highest-energy peak in the ROI

# If you DO NOT have a true energy axis file, this guess is used to set meV/pixel scale
dE_meV_guess = (
    7.0  # meV between adjacent replicas (used only if no energy axis is available)
)

# ROI control:
# If an energy axis exists, ROI is picked around E0 using these spans.
E_span_meV_low, E_span_meV_up = 50, 20  # show [E0-50meV, E0+20meV]

# If no energy axis exists, ROI is auto-found by scanning. You can override manually:
force_roi_pixels = None  # e.g., (500, 620) or None for auto

# Simulation controls
Nmax = 8  # include enough orders (recommend ~S + 4*sqrt(S))
rng_seed = 123

# Output directory
out_dir = r"C:\Users\Saroj Chand\OneDrive - CUNY\EXPERIMENTS\20220329_WSe2_dark_exciton_strain_pockets\Figure_for_papers\sim_fit_roi"
# =======================================================


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_exp_matrix(path: str):
    """
    Your file format: header line, then a row that is exactly 0..(n_frames-1),
    then n_pixels rows of counts.
    Returns counts[pixel, frame]
    """
    raw = np.loadtxt(path, skiprows=1)  # skip header
    # If first data row is 0..n-1, drop it
    first = raw[0]
    if np.allclose(first, np.arange(first.size), atol=1e-9):
        raw = raw[1:]
    counts = raw.astype(float)
    return counts  # (n_pixels, n_frames)


def try_load_energy_axis(path: str, n_pixels: int, exp_md5: str):
    """
    Tries to load an energy axis (eV) from file.
    Accepts:
      - 1D array length n_pixels
      - 2D array with one column length n_pixels
      - 2D array with two columns length n_pixels (uses last col)
    If file is identical to exp_data, returns None (common mistaken upload).
    """
    if path is None or (not os.path.exists(path)):
        return None

    if md5(path) == exp_md5:
        # looks like the same file re-uploaded
        return None

    arr = np.loadtxt(path)
    arr = np.array(arr)

    if arr.ndim == 1 and arr.size == n_pixels:
        return arr.astype(float)

    if arr.ndim == 2 and arr.shape[0] == n_pixels:
        if arr.shape[1] == 1:
            return arr[:, 0].astype(float)
        if arr.shape[1] >= 2:
            return arr[:, -1].astype(float)

    return None


def preprocess_vec(y):
    y = y - np.percentile(y, 10)
    y = np.clip(y, 0, None)
    y = y / (np.linalg.norm(y) + 1e-12)
    return y


def estimate_global_shifts(Y_roi, max_shift=50):
    """
    Rigid drift per frame by maximizing normalized dot product vs frame 0.
    Returns integer pixel shifts (same length as n_frames).
    """
    ref = preprocess_vec(Y_roi[:, 0])
    shifts = []
    for f in range(Y_roi.shape[1]):
        y = preprocess_vec(Y_roi[:, f])
        best_s, best_c = 0, -1e18
        for s in range(-max_shift, max_shift + 1):
            if s < 0:
                c = np.dot(ref[-s:], y[: len(y) + s])
            elif s > 0:
                c = np.dot(ref[:-s], y[s:])
            else:
                c = np.dot(ref, y)
            if c > best_c:
                best_c, best_s = c, s
        shifts.append(best_s)
    return np.array(shifts, dtype=int)


def apply_shift(y, s):
    """
    Shift vector y by integer s (pixels), padding with zeros.
    Convention here matches estimate_global_shifts usage.
    """
    out = np.zeros_like(y)
    if s < 0:
        out[-s:] = y[: len(y) + s]
    elif s > 0:
        out[:-s] = y[s:]
    else:
        out[:] = y
    return out


def find_three_peaks(y, prominence_frac=0.01, distance=6):
    """
    Find the top 3 peaks by prominence.
    Returns indices (sorted). If fewer than 3, returns None.
    """
    y_sm = gaussian_filter1d(y, sigma=1.0)
    pk, pr = find_peaks(
        y_sm, prominence=np.max(y_sm) * prominence_frac, distance=distance
    )
    if len(pk) < 3:
        pk, pr = find_peaks(
            y_sm,
            prominence=np.max(y_sm) * (0.5 * prominence_frac),
            distance=max(4, distance - 1),
        )
    if len(pk) < 3:
        return None

    idx = np.argsort(pr["prominences"])[-3:]
    pk = np.sort(pk[idx])
    return pk


def fit_S_from_peak_areas(Y_roi_aligned, peak_idx3, halfwin=12):
    """
    Estimate Huang–Rhys S from area ratios of the first three replicas:
      A1/A0 ~ S
      A2/A0 ~ S^2/2
    """
    n_pix, n_frames = Y_roi_aligned.shape
    areas = np.zeros((n_frames, 3), dtype=float)

    for f in range(n_frames):
        y = Y_roi_aligned[:, f].astype(float)
        y = y - np.percentile(y, 10)
        y = np.clip(y, 0, None)
        for j in range(3):
            c = int(peak_idx3[j])
            lo, hi = max(0, c - halfwin), min(n_pix, c + halfwin + 1)
            seg = y[lo:hi]
            base = np.median(np.r_[seg[:3], seg[-3:]]) if seg.size >= 6 else np.min(seg)
            seg2 = np.clip(seg - base, 0, None)
            areas[f, j] = np.sum(seg2)

    Abar = np.mean(areas, axis=0)
    r1 = Abar[1] / (Abar[0] + 1e-12)
    r2 = Abar[2] / (Abar[0] + 1e-12)

    Ss = np.linspace(0.2, 10.0, 5001)
    err = (Ss - r1) ** 2 + (Ss**2 / 2.0 - r2) ** 2
    S_fit = Ss[np.argmin(err)]
    return float(S_fit), Abar


def voigt(E, center, sigma_eV, gamma_eV):
    z = ((E - center) + 1j * gamma_eV) / (sigma_eV * np.sqrt(2))
    return np.real(wofz(z)) / (sigma_eV * np.sqrt(2 * np.pi))


def auto_find_roi_by_ladder(counts, win=140, step=10, n_check_frames=7):
    """
    Scan windows; prefer regions where (i) 3 peaks exist in many frames,
    and (ii) peak spacings are roughly uniform (ladder-like).
    """
    n_pixels, n_frames = counts.shape
    frame_idx = np.linspace(0, n_frames - 1, n_check_frames).astype(int)

    best = None  # (score, lo, hi)
    for lo in range(0, n_pixels - win, step):
        hi = lo + win
        Yw = counts[lo:hi, :]

        ok = 0
        spacing_pen = 0.0
        intensity = np.mean(Yw)

        for f in frame_idx:
            y = Yw[:, f].astype(float)
            y = y - np.percentile(y, 10)
            y = np.clip(y, 0, None)
            pk = find_three_peaks(y, prominence_frac=0.01, distance=6)
            if pk is None:
                continue
            ok += 1
            d01 = pk[1] - pk[0]
            d12 = pk[2] - pk[1]
            spacing_pen += abs(d01 - d12)

        frac_ok = ok / len(frame_idx)
        if frac_ok < 0.6:
            continue

        spacing_pen /= max(ok, 1)
        # score: want high intensity, high frac_ok, low spacing_pen
        score = (intensity / (1 + spacing_pen)) * (0.5 + frac_ok)

        if (best is None) or (score > best[0]):
            best = (score, lo, hi)

    if best is None:
        # fallback: middle chunk
        return (int(0.45 * n_pixels), int(0.60 * n_pixels))
    return (best[1], best[2])


# ===================== MAIN =====================
os.makedirs(out_dir, exist_ok=True)

counts = load_exp_matrix(exp_data_path)  # (n_pixels, n_frames)
n_pixels, n_frames = counts.shape
exp_md5 = md5(exp_data_path)

E_axis = try_load_energy_axis(energy_axis_path, n_pixels=n_pixels, exp_md5=exp_md5)

# ROI selection
if force_roi_pixels is not None:
    roi_lo, roi_hi = force_roi_pixels
else:
    if E_axis is not None:
        to_eV = 1e-3
        roi_lo = np.searchsorted(E_axis, E0 - E_span_meV_low * to_eV, side="left")
        roi_hi = np.searchsorted(E_axis, E0 + E_span_meV_up * to_eV, side="right")
        roi_lo = max(0, roi_lo - 5)
        roi_hi = min(n_pixels, roi_hi + 5)
        if roi_hi - roi_lo < 80:
            roi_lo = max(0, roi_lo - 40)
            roi_hi = min(n_pixels, roi_hi + 40)
    else:
        roi_lo, roi_hi = auto_find_roi_by_ladder(counts, win=140, step=10)

Y_roi = counts[roi_lo:roi_hi, :]  # (n_roi_pix, n_frames)
n_roi = Y_roi.shape[0]

# Drift estimation (in pixels)
shifts_pix = estimate_global_shifts(Y_roi, max_shift=60)

# Align ROI by shifting each frame
Y_aligned = np.zeros_like(Y_roi)
for f in range(n_frames):
    Y_aligned[:, f] = apply_shift(Y_roi[:, f], shifts_pix[f])

# Find 3 peaks in aligned *average* spectrum (more stable)
y_avg = np.mean(Y_aligned, axis=1)
y_avg = y_avg - np.percentile(y_avg, 10)
y_avg = np.clip(y_avg, 0, None)
pk3 = find_three_peaks(y_avg, prominence_frac=0.01, distance=6)
if pk3 is None:
    # emergency fallback: take top 3 bins
    pk3 = np.sort(np.argsort(y_avg)[-3:])

# Peak spacing in pixels
d01 = pk3[1] - pk3[0]
d12 = pk3[2] - pk3[1]
d_pix = float(np.median([d01, d12]))

# Fit S from peak areas
S_fit, Abar = fit_S_from_peak_areas(Y_aligned, pk3, halfwin=12)

# Estimate linewidth (FWHM) of dominant peak across frames (pixels)
fwhm_pix_list = []
for f in range(n_frames):
    y = Y_aligned[:, f].astype(float)
    y = y - np.percentile(y, 10)
    y = np.clip(y, 0, None)
    y_sm = gaussian_filter1d(y, sigma=1.0)
    pk, pr = find_peaks(y_sm, prominence=np.max(y_sm) * 0.01, distance=6)
    if len(pk) == 0:
        continue
    idx = np.argmax(pr["prominences"])
    w = peak_widths(y_sm, [pk[idx]], rel_height=0.5)[0][0]
    fwhm_pix_list.append(w)
fwhm_pix_med = float(np.median(fwhm_pix_list)) if len(fwhm_pix_list) else 12.0

# Frame gain (formation/collection fluctuations) from total ROI intensity
I_frame = np.sum(Y_roi, axis=0)
frame_gain = I_frame / (np.median(I_frame) + 1e-12)
frame_gain_smooth = gaussian_filter1d(frame_gain, sigma=1.0)

# Additive noise estimate from low-intensity pixels in ROI
m = np.mean(Y_roi, axis=1)
baseline_mask = m < np.percentile(m, 30)
noise_std_counts = float(np.std(Y_roi[baseline_mask, :].ravel(), ddof=1))

# Build energy axis for ROI
to_eV = 1e-3

if E_axis is not None:
    E_roi = E_axis[roi_lo:roi_hi].copy()
    # Learn dE_meV from energy differences of the three peaks in aligned average:
    E_pk = E_roi[pk3]
    dE_meV_01 = (E_pk[0] - E_pk[1]) * 1e3
    dE_meV_12 = (E_pk[1] - E_pk[2]) * 1e3
    dE_meV = float(np.median([dE_meV_01, dE_meV_12]))
    meV_per_pix = dE_meV / (d_pix + 1e-12)
else:
    # No true energy axis: define a linear axis anchored by E0 and using guessed dE_meV
    dE_meV = float(dE_meV_guess)
    meV_per_pix = dE_meV / (d_pix + 1e-12)

    # Put the highest-energy peak (pk3[0]) at E0
    p0 = pk3[0]
    pix = np.arange(n_roi)
    E_roi = E0 - (pix - p0) * meV_per_pix * 1e-3

# Convert drift shifts to energy drift (eV)
drift_eV_per_frame = shifts_pix * meV_per_pix * 1e-3

# Convert linewidth from pixels to meV and choose Voigt parts
fwhm_meV = fwhm_pix_med * meV_per_pix
sigma0_meV = max(0.5, fwhm_meV / 2.355)  # assume mostly Gaussian
gamma0_meV = 0.4 * sigma0_meV  # modest Lorentzian tail

# Print learned parameters
print("\n=== Learned (ROI) ===")
print(f"ROI pixels: [{roi_lo}, {roi_hi})  (n={n_roi})")
print(
    f"Peak spacing: d_pix ~ {d_pix:.2f} px  =>  dE ~ {dE_meV:.2f} meV (if calibrated; else from guess)"
)
print(f"meV per pixel: {meV_per_pix:.4f} meV/px")
print(f"Huang–Rhys S (from area ratios): S ~ {S_fit:.2f}")
print(f"FWHM (dominant peak): ~ {fwhm_pix_med:.2f} px  (~{fwhm_meV:.2f} meV)")
print(f"Noise std (counts, rough): ~ {noise_std_counts:.1f}")
print(
    "Frame gain (median=1) range:",
    float(frame_gain.min()),
    "to",
    float(frame_gain.max()),
)
print(
    "Drift shifts (pixels) range:", int(shifts_pix.min()), "to", int(shifts_pix.max())
)

# ===================== SIMULATION (localization-motivated) =====================
rng = np.random.default_rng(rng_seed)

# phonon orders & Poisson weights (Franck–Condon)
n = np.arange(Nmax + 1)
# Localization/strain can vary S slightly frame-to-frame
S0 = S_fit
w_base = np.exp(-S0) * (S0**n) / factorial(n)

# counts scaling (match typical ROI total intensity)
counts_per_frame = float(np.median(I_frame))
intensity_scale = 1.0

# Jitter model:
# - global drift uses measured drift_eV_per_frame (this captures “spectral walk/jumps”)
# - extra per-order residual jitter (small)
jitter_white_meV = 0.4 * meV_per_pix
jitter_walk_meV = 0.6 * meV_per_pix

rw_state_meV = np.zeros_like(n, dtype=float)

# Broadening vs order (localization broadens finite-q access)
alpha_n = 0.10  # order growth
alpha_loc = 0.35  # extra broadening when localization stronger (proxy from frame_gain)

# Order-mixing (intensity leaks to neighbors), stronger when localization stronger
mix_base = 0.10

# Build simulated spectra in ROI energy grid
E = E_roi.copy()
spectra_sim = np.zeros((n_frames, n_roi), dtype=float)

for f in range(n_frames):
    # Use measured formation/collection fluctuations as a proxy for “localization/strain strength”
    # (you can swap this proxy later for something more physical)
    loc = float(frame_gain_smooth[f] - 1.0)

    # Allow S to vary mildly with loc (localization broadens F(q-Q) -> more multi-phonon weight)
    S_frame = max(0.1, S0 * (1.0 + 0.10 * loc))
    w = np.exp(-S_frame) * (S_frame**n) / factorial(n)

    # Expected counts in each order
    lam = counts_per_frame * frame_gain_smooth[f] * (w / (np.sum(w) + 1e-12))
    A = rng.poisson(lam) * intensity_scale

    # Order mixing (bleed) increases with |loc|
    A_mixed = A.copy()
    mix = mix_base * (1.0 + 0.6 * abs(loc))
    for i_ord in range(1, len(A) - 1):
        leak = mix * A[i_ord]
        A_mixed[i_ord] -= leak
        A_mixed[i_ord - 1] += 0.5 * leak
        A_mixed[i_ord + 1] += 0.5 * leak
    A = A_mixed

    # Global drift from experiment
    global_shift_eV = drift_eV_per_frame[f]

    # Spectrum build
    spec = np.zeros_like(E)

    for i_ord, Ai in enumerate(A):
        # Stokes ladder
        center0 = E0 - i_ord * (dE_meV * 1e-3)

        # residual per-order jitter (meV)
        rw_state_meV[i_ord] += rng.normal(0, jitter_walk_meV)
        white_term = rng.normal(0, jitter_white_meV)
        jitter_eV = (rw_state_meV[i_ord] + white_term) * 1e-3

        center = center0 + global_shift_eV + jitter_eV

        # Voigt widths in meV -> eV
        sigma_meV = sigma0_meV * (1.0 + alpha_n * i_ord) * (1.0 + alpha_loc * abs(loc))
        gamma_meV = gamma0_meV * (1.0 + 0.08 * i_ord) * (1.0 + 0.25 * abs(loc))

        sigma_eV = sigma_meV * 1e-3
        gamma_eV = gamma_meV * 1e-3

        spec += Ai * voigt(E, center, sigma_eV, gamma_eV)

    # Additive noise (match rough scale)
    spec += rng.normal(0, noise_std_counts, size=spec.shape)

    spec = np.clip(spec, 0, None)
    spectra_sim[f] = spec

# ===================== NORMALIZATION & SAVING =====================
# Normalize experimental ROI and simulation together (global max = 1)
Y_roi_norm = Y_roi / (np.max(Y_roi) + 1e-12)
spectra_sim_norm = spectra_sim / (np.max(spectra_sim) + 1e-12)

np.savetxt(os.path.join(out_dir, "energy_roi_eV.txt"), E, fmt="%.9f")
np.savetxt(os.path.join(out_dir, "exp_roi_norm_framesxE.txt"), Y_roi_norm.T, fmt="%.9f")
np.savetxt(
    os.path.join(out_dir, "sim_roi_norm_framesxE.txt"), spectra_sim_norm, fmt="%.9f"
)

# Save per-frame spectra (optional convenience)
for f in range(n_frames):
    np.savetxt(
        os.path.join(out_dir, f"exp_roi_{f:03d}.txt"), Y_roi_norm[:, f], fmt="%.9f"
    )
    np.savetxt(
        os.path.join(out_dir, f"sim_roi_{f:03d}.txt"), spectra_sim_norm[f], fmt="%.9f"
    )

# ===================== PLOTS =====================
colors = ["black", "red", "white"]
custom_cmap = LinearSegmentedColormap.from_list("BlackWhiteRed", colors)

plt.figure(figsize=(9, 4), dpi=150)

plt.subplot(1, 2, 1)
extent = [E.min(), E.max(), 0, n_frames]
plt.imshow(Y_roi_norm.T, aspect="auto", origin="lower", extent=extent, cmap=custom_cmap)
plt.title("Experiment (ROI, normalized)")
plt.xlabel("Energy (eV)")
plt.yticks([])
plt.colorbar(label="Norm. intensity", fraction=0.046, pad=0.04)

plt.subplot(1, 2, 2)
plt.imshow(
    spectra_sim_norm, aspect="auto", origin="lower", extent=extent, cmap=custom_cmap
)
plt.title("Simulation (fit-guided, normalized)")
plt.xlabel("Energy (eV)")
plt.yticks([])
plt.colorbar(label="Norm. intensity", fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# One-frame overlay (quick sanity check)
f_show = 10
plt.figure(figsize=(7, 3), dpi=150)
plt.plot(E, Y_roi_norm[:, f_show], label=f"Exp frame {f_show}", lw=1.5)
plt.plot(E, spectra_sim_norm[f_show], label=f"Sim frame {f_show}", lw=1.2, alpha=0.85)
plt.xlabel("Energy (eV)")
plt.ylabel("Normalized intensity")
plt.legend()
plt.tight_layout()
plt.show()
