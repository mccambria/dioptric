# -*- coding: utf-8 -*-
"""
Extract the magnitic fiel
Created on March 23th, 2025
@author: Saroj Chand
"""


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from  utils import kplotlib as kpl
# Comprehensive simulation of Néel and Brownian relaxation for magnetic nanoparticles
# Author: you :)
# Dependencies: numpy, pandas, matplotlib
# Usage: python magnetic_relaxation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# ---------- CONFIG (edit these as you like) ----------
material = "Magnetite (Fe3O4)"
T = 298                     # temperature [K]
eta = 1e-3                       # dynamic viscosity [Pa*s] (≈ water at room temp)
K = 13e3                         # anisotropy constant [J/m^3] (typical magnetite)
tau0 = 1e-9                      # attempt time [s] (1e-12 to 1e-9 is common)
shell_thickness_nm = 5.0         # hydrodynamic shell (e.g., surfactant) [nm]
d_min_nm, d_max_nm, d_step_nm = 8.0, 25.0, 0.5  # core diameter range [nm]
csv_path = "magnetic_relaxation_10to100nm.csv"
# ----------------------------------------------------

kB = 1.380649e-23  # Boltzmann constant [J/K]

def core_volume_m3(d_nm: np.ndarray) -> np.ndarray:
    """Volume of a spherical core of diameter d_nm [nm] in m^3."""
    r_m = (d_nm * 1e-9) / 2.0
    return (4.0/3.0) * pi * r_m**3

def hydro_volume_m3(d_core_nm: np.ndarray, shell_nm: float) -> np.ndarray:
    """Hydrodynamic volume (core + shell) as a sphere in m^3."""
    d_h_nm = d_core_nm + 2.0*shell_nm
    r_h_m = (d_h_nm * 1e-9) / 2.0
    return (4.0/3.0) * pi * r_h_m**3

def tau_neel(d_nm: np.ndarray, K_Jm3: float, tau0_s: float, T_K: float) -> np.ndarray:
    """Néel relaxation time τ_N = τ0 * exp(K*V / (kB*T)). Uses overflow-safe exponent."""
    V = core_volume_m3(d_nm)
    expo = (K_Jm3 * V) / (kB * T_K)
    # Avoid overflow in exp for very large particles:
    expo = np.clip(expo, -700, 700)   # exp(700) ~ 1e304 (huge but finite)
    return tau0_s * np.exp(expo)

def tau_brownian(d_nm: np.ndarray, eta_Pas: float, T_K: float, shell_nm: float) -> np.ndarray:
    """Brownian relaxation τ_B = 3*η*V_h / (kB*T). Uses hydrodynamic volume."""
    Vh = hydro_volume_m3(d_nm, shell_nm)
    return (3.0 * eta_Pas * Vh) / (kB * T_K)

def tau_effective(tn: np.ndarray, tb: np.ndarray) -> np.ndarray:
    """
    Effective relaxation time for parallel pathways:
    1/τ_eff = 1/τ_N + 1/τ_B  => τ_eff = (τ_N * τ_B) / (τ_N + τ_B)
    numerically stable for large values.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return (tn * tb) / (tn + tb)

def frac_contribution_neel(tn: np.ndarray, tb: np.ndarray) -> np.ndarray:
    """Fractional contribution of Néel channel to overall relaxation rate."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rateN = 1.0 / tn
        rateB = 1.0 / tb
        return rateN / (rateN + rateB)

def main():
    # Prepare diameter grid
    d_core_nm = np.arange(d_min_nm, d_max_nm + d_step_nm/2.0, d_step_nm)

    # Compute relaxations
    tauN = tau_neel(d_core_nm, K, tau0, T)
    tauB = tau_brownian(d_core_nm, eta, T, shell_thickness_nm)
    tauEff = tau_effective(tauN, tauB)
    fracN = frac_contribution_neel(tauN, tauB)  # 0..1

    # Characteristic frequency
    with np.errstate(divide='ignore'):
        f_opt = 1.0 / (2.0 * pi * tauEff)

    # Find crossover diameter where τN ≈ τB
    # Use log-distance; ignore NaNs/infs by masking
    logN = np.log10(tauN)
    logB = np.log10(tauB)
    mask = np.isfinite(logN) & np.isfinite(logB)
    if np.any(mask):
        log_ratio = np.abs(logN[mask] - logB[mask])
        idx_local = int(np.argmin(log_ratio))
        idx_cross = np.arange(len(d_core_nm))[mask][idx_local]
    else:
        idx_cross = np.nan

    d_cross_nm = float(d_core_nm[idx_cross]) if np.isfinite(idx_cross) else np.nan
    tau_cross_s = float(tauN[idx_cross]) if np.isfinite(idx_cross) else np.nan
    f_cross = (1.0 / (2.0 * pi * tau_cross_s)) if np.isfinite(tau_cross_s) else np.nan

    # Build table
    df = pd.DataFrame({
        "d_core_nm": d_core_nm,
        "tau_N_s": tauN,
        "tau_B_s": tauB,
        "tau_eff_s": tauEff,
        "f_opt_Hz": f_opt,
        "frac_Neel": fracN
    })
    df.to_csv(csv_path, index=False)

    # --- Plots (each plot in its own figure, no styles or explicit colors) ---
    plt.figure(figsize=(7,5))
    # plt.loglog(d_core_nm, tauN, label="τ_N (Néel)")
    # plt.loglog(d_core_nm, tauB, label="τ_B (Brownian)")
    # plt.loglog(d_core_nm, tauEff, label="τ_eff")
    plt.plot(d_core_nm, tauN, label="τ_N (Néel)")
    plt.plot(d_core_nm, tauB, label="τ_B (Brownian)")
    plt.plot(d_core_nm, tauEff, label="τ_eff")
    if np.isfinite(d_cross_nm):
        plt.axvline(d_cross_nm, linestyle="--")
    plt.title(f"Relaxation vs Core Diameter @ {T:.1f} K, η={eta} Pa·s\n"
              f"{material}, K={K:.1e} J/m³, τ_0={tau0:.0e} s, shell={shell_thickness_nm} nm", fontsize=15)
    # --- Add equations ---
    plt.text(
        0.4*d_max_nm, 100e-6,   # adjust placement
        r"$\tau_N = \tau_0 \exp\!\left(\frac{K V}{k_B T}\right)$" "\n"
        r"$\tau_B = \frac{3 \eta V_h}{k_B T}$" "\n"
        r"$\tau_{\mathrm{eff}} = \frac{\tau_N \tau_B}{\tau_N + \tau_B}$",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )
    plt.xlabel("Core diameter d (nm)")
    plt.ylabel("Relaxation time τ (s)")
    plt.yscale ("log")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(7,5))
    plt.semilogy(d_core_nm, f_opt)
    if np.isfinite(d_cross_nm):
        plt.axvline(d_cross_nm, linestyle="--")
    plt.title("Characteristic frequency ~ 1/(2π τ_eff)")
    plt.xlabel("Core diameter d (nm)")
    plt.ylabel("f (Hz)")
    plt.grid(True)

    plt.figure(figsize=(7,5))
    plt.plot(d_core_nm, fracN)
    if np.isfinite(d_cross_nm):
        plt.axvline(d_cross_nm, linestyle="--")
    plt.title("Fractional Néel contribution")
    plt.xlabel("Core diameter d (nm)")
    plt.ylabel("Néel fraction of rate (0..1)")
    plt.grid(True)
    print(f"""
Summary for {material} at T={T:.1f} K, η={eta} Pa·s
---------------------------------------------------
Anisotropy K      : {K:.3e} J/m^3
Attempt time τ0   : {tau0:.1e} s
Shell thickness   : {shell_thickness_nm} nm
Size range        : {d_min_nm}–{d_max_nm} nm (step {d_step_nm} nm)

Crossover (τ_N ≈ τ_B):
  d_core ≈ {d_cross_nm:.2f} nm
  τ_cross ≈ {tau_cross_s:.3e} s
  f_cross ≈ {f_cross:.3e} Hz

CSV saved to: {csv_path}
""")


if __name__ == "__main__":
    kpl.init_kplotlib()
    # main()

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from math import pi

    # --- CONFIG: superparamagnetic (no remanence) ---
    material = "Magnetite (Fe3O4)"
    T = 298.15                 # [K]
    eta = 1e-3                 # [Pa*s]
    K = 13e3                   # [J/m^3]
    tau0 = 1e-9                # [s]
    Ms = 4.8e5                 # [A/m]
    shell_thickness_nm = 5.0   # [nm]

    d_vals = np.linspace(5.0, 20.0, 161)     # core diameter [nm]
    r_vals = np.linspace(5.0, 20.0, 161)     # NV distance [nm]
    # ------------------------------------------------

    mu0 = 4e-7*np.pi
    kB = 1.380649e-23

    def core_volume_m3(d_nm):
        r = (d_nm*1e-9)/2.0
        return (4/3)*np.pi*r**3

    def hydro_volume_m3(d_core_nm, shell_nm):
        d_h_nm = d_core_nm + 2*shell_nm
        r_h = (d_h_nm*1e-9)/2.0
        return (4/3)*np.pi*r_h**3

    def tau_neel(d_nm):
        V = core_volume_m3(d_nm)
        expo = (K*V)/(kB*T)
        expo = np.clip(expo, -700, 700)
        return tau0*np.exp(expo)

    def tau_brownian(d_nm):
        Vh = hydro_volume_m3(d_nm, shell_thickness_nm)
        return (3*eta*Vh)/(kB*T)

    def tau_eff(d_nm):
        tN, tB = tau_neel(d_nm), tau_brownian(d_nm)
        return (tN*tB)/(tN+tB)

    def field_NSD_map(d_vals, r_vals):
        D, R = np.meshgrid(d_vals, r_vals)
        NSD = np.zeros_like(D, dtype=float)
        # omega = 2*np.pi*f_Hz
        for i in range(R.shape[0]):
            for j in range(D.shape[1]):
                dnm = D[i,j]
                V = core_volume_m3(dnm)
                mu = Ms * V
                chi0 = mu0 * mu**2 / (3.0 * kB * T)
                tauE = tau_eff(dnm)
                omega = 1/tauE 
                Sm = 2.0 * kB * T * chi0 * tauE / (1.0 + (omega * tauE)**2)  # (A·m^2)^2/Hz
                kern = (mu0/(4*np.pi)) / ((R[i,j]*1e-9)**3)
                geom_rms = np.sqrt(2.0/3.0)  # isotropic average
                NSD[i,j] = kern * geom_rms * np.sqrt(Sm)  # T / sqrt(Hz)
        return D, R, NSD

    D, R, NSD_T_per_sqrtHz = field_NSD_map(d_vals, r_vals)

    # Save CSV
    df = pd.DataFrame({"core_d_nm": D.flatten(),
                    "distance_nm": R.flatten(),
                    "B_NSD_nT_per_sqrtHz": (NSD_T_per_sqrtHz.flatten()*1e9)})
    df.to_csv("NV_field_NSD_superparamagnetic.csv", index=False)

    # Plot heatmap + contours
    plt.figure(figsize=(7,5))
    Z = NSD_T_per_sqrtHz * 1e9  # nT/√Hz
    im = plt.pcolormesh(D, R, Z, shading='auto')
    plt.colorbar(im, label=r"$B_\mathrm{NSD}$ (nT/$\sqrt{\mathrm{Hz}}$)")
    plt.xlabel("Core diameter (nm)")
    plt.ylabel("NV distance (nm)")
    plt.title(f"Superparamagnetic thermal field at NV\n{material}")
    levels = [0.1, 0.3, 1, 3, 10, 30, 100]  # nT/√Hz
    CS = plt.contour(D, R, Z, levels=levels, colors='w', linewidths=0.7)
    plt.clabel(CS, inline=True, fmt="%.1f", fontsize=8)
    plt.show(block=True)

