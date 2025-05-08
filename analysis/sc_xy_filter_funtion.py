# -*- coding: utf-8 -*-
"""
Extract the magnitic fiel
Created on March 23th, 2025
@author: Saroj Chand
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma_C13 = 1.0705e3  # Hz/G
B_field = 66  # Gauss
f_Larmor = gamma_C13 * B_field  # Larmor frequency in Hz
T_Larmor = 1 / f_Larmor  # Larmor period in seconds


# Filter function for XY8 (simplified envelope)
def filter_function(tau, N):
    """Returns the filter function amplitude vs. tau for XY8-N"""
    omega = 2 * np.pi * f_Larmor
    T = 8 * N * tau
    sinc_arg = omega * tau / 2
    envelope = (np.sin(omega * T / 2) / np.sin(sinc_arg)) ** 2
    return envelope / np.max(envelope)


# Parameters
taus = np.linspace(0.1e-6, 60e-6, 1000)  # 0.1 to 20 microseconds
N = 8
# XY8-8

filter_vals = filter_function(taus, N)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(taus * 1e6, filter_vals, label=f"XY8-{N} Filter Function")
plt.axvline(T_Larmor * 1e6 / 2, color="r", linestyle="--", label="τ = T_Larmor / 2")
plt.axvline(T_Larmor * 1e6, color="g", linestyle="--", label="τ = T_Larmor")
plt.title("XY8 Filter Function vs τ (Targeting $^{13}$C Larmor)")
plt.xlabel("τ (µs)")
plt.ylabel("Normalized Filter Strength")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Simulate NV coherence decay under a 13C spin bath using XY8 filter function


def lorentzian_spectral_density(omega, omega_L, width):
    """Lorentzian noise spectrum centered at omega_L with a given width"""
    return 1 / (1 + ((omega - omega_L) / width) ** 2)


# Parameters
omega_L = 2 * np.pi * f_Larmor  # center frequency in rad/s
width = 2 * np.pi * 10e3  # 10 kHz width, moderate nuclear spin bath


# Simulate coherence decay: integrate spectral density * filter function
def simulate_coherence(taus, N, omega_L, width):
    coherence = []
    for tau in taus:
        T = 8 * N * tau
        omega = np.linspace(omega_L - 5 * width, omega_L + 5 * width, 1000)
        S = lorentzian_spectral_density(omega, omega_L, width)
        F = (np.sin(omega * T / 2) / np.sin(omega * tau / 2)) ** 2
        F /= np.max(F)  # normalize filter strength
        integrand = S * F
        A = np.trapz(integrand, omega)
        coherence.append(np.exp(-A))  # exponential decay due to bath
    return np.array(coherence)


# Calculate
coherence_vals = simulate_coherence(taus, N=8, omega_L=omega_L, width=width)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(taus * 1e6, coherence_vals, label="NV Coherence (XY8-8)")
plt.axvline(T_Larmor * 1e6 / 2, color="r", linestyle="--", label="τ = T_Larmor / 2")
plt.axvline(T_Larmor * 1e6, color="g", linestyle="--", label="τ = T_Larmor")
plt.title("Simulated NV Coherence vs τ (XY8-8 under ¹³C Spin Bath)")
plt.xlabel("τ (µs)")
plt.ylabel("Normalized Coherence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Re-import necessary packages after code execution state reset
import numpy as np
import matplotlib.pyplot as plt

# Constants
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)
hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
gamma_e = 2.802495e10  # Gyromagnetic ratio of electron (rad/s/T)


# Dipolar coupling strength (Jzz) between two electron spins
def dipolar_coupling(d_nm):
    d_m = d_nm * 1e-9  # Convert nm to m
    J = (mu0 / (4 * np.pi)) * (gamma_e**2 * hbar) / (d_m**3)  # in Joules
    J_Hz = J / (2 * np.pi * hbar)  # Convert to Hz
    return J_Hz  # Return in Hz


# Entangling time te = π / Jzz
def entangling_time(J_Hz):
    return np.pi / J_Hz  # in seconds


# Distances from 2 nm to 20 nm
distances = np.linspace(2, 20, 300)
Jzz_values = dipolar_coupling(distances)
te_values = entangling_time(Jzz_values) * 1e6  # Convert to µs

# Plot
fig, ax1 = plt.subplots(figsize=(8, 5))

color = "tab:blue"
ax1.set_xlabel("Distance between NVs (nm)")
ax1.set_ylabel("Dipolar Coupling Jzz (Hz)", color=color)
ax1.plot(distances, Jzz_values, color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.axvline(6, linestyle="--", color="gray", label="d ≈ 6 nm")

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Entanglement Time te = π / Jzz (µs)", color=color)
ax2.plot(distances, te_values, color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Dipolar Coupling and Entanglement Time vs NV-NV Distance")
fig.tight_layout()
plt.grid(True)
plt.legend(loc="upper right")
plt.show()
