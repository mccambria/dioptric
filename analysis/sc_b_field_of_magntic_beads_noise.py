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

# Constants
mu_0 = 4 * np.pi * 1e-7  # T·m/A
gamma_nv = 2 * np.pi * 28e9  # rad/T·s (gyromagnetic ratio of NV center)

# Parameters for simulation
bead_diameters = [150e-9, 2e-6]  # in meters
bead_moment = [1e-18, 1e-15]  # A·m^2 (example magnetic moments, scale with volume)
nv_distance = 100e-9  # NV depth (distance from bead), in meters
tau = np.logspace(-7, -4, 200)  # Free evolution times from 100 ns to 10 µs


# Function to simulate Lorentzian noise spectrum and phase variance
def simulate_phase_variance(m, d, tau_m, tau_vals):
    B_dip = mu_0 * m / (4 * np.pi * d**3)  # Approximate dipolar field at distance d
    omega_c = 1 / tau_m  # Characteristic fluctuation frequency

    # Lorentzian noise spectral density at NV Larmor frequency (simplified model)
    S_B = B_dip**2 * tau_m / (1 + (2 * np.pi / tau_vals / omega_c) ** 2)

    # Phase variance using simplified model: phi^2 ~ (gamma * B)^2 * tau^2
    phi2 = (gamma_nv * B_dip) ** 2 * S_B * tau_vals**2
    return phi2


# Simulate for both bead sizes
tau_m_small = 1e-7  # 150 nm bead: faster fluctuations (e.g., 10 MHz)
tau_m_large = 1e-4  # 2 µm bead: slower fluctuations (e.g., 10 kHz)

phi2_small = simulate_phase_variance(bead_moment[0], nv_distance, tau_m_small, tau)
phi2_large = simulate_phase_variance(bead_moment[1], nv_distance, tau_m_large, tau)

# Convert phase variance to coherence
C_small = np.exp(-phi2_small / 2)
C_large = np.exp(-phi2_large / 2)

# Plotting
plt.figure(figsize=(10, 5))
plt.loglog(tau * 1e6, C_small, label="150 nm bead (fast noise)", lw=2)
plt.loglog(tau * 1e6, C_large, label="2 µm bead (slow noise)", lw=2)
plt.xlabel("Free Evolution Time τ (µs)")
plt.ylabel("NV Coherence C(τ)")
plt.title("Simulated NV Spin Echo Decay from Paramagnetic Beads")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()


# Calculate phase variance for both bead cases (already done, just reuse)
# Now compute corresponding Lorentzian noise spectral densities at a range of frequencies

omega = np.logspace(4, 8, 500)  # Frequency range from 10 kHz to 100 MHz


def lorentzian_spectrum(m, d, tau_m, omega_vals):
    B_dip = mu_0 * m / (4 * np.pi * d**3)
    S_B = B_dip**2 * tau_m / (1 + (omega_vals * tau_m) ** 2)
    return S_B


# Compute noise spectra
S_small = lorentzian_spectrum(bead_moment[0], nv_distance, tau_m_small, omega)
S_large = lorentzian_spectrum(bead_moment[1], nv_distance, tau_m_large, omega)

# Plot phase variance
plt.figure(figsize=(10, 4.5))
plt.subplot(1, 2, 1)
plt.loglog(tau * 1e6, phi2_small, label="150 nm bead (fast)", lw=2)
plt.loglog(tau * 1e6, phi2_large, label="2 µm bead (slow)", lw=2)
plt.xlabel("Free Evolution Time τ (µs)")
plt.ylabel("Phase Variance ⟨ϕ²(τ)⟩")
plt.title("NV Phase Variance vs Time")
plt.grid(True, which="both", ls="--")
plt.legend()

# Plot noise spectral densities
plt.subplot(1, 2, 2)
plt.loglog(omega / (2 * np.pi * 1e6), S_small, label="150 nm bead (fast)", lw=2)
plt.loglog(omega / (2 * np.pi * 1e6), S_large, label="2 µm bead (slow)", lw=2)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Noise Spectral Density N(ω) (T²/Hz)")
plt.title("Magnetic Noise Spectral Density")
plt.grid(True, which="both", ls="--")
plt.legend()

plt.tight_layout()
plt.show()
