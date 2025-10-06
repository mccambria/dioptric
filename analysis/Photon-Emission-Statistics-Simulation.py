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
