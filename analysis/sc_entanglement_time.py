# -*- coding: utf-8 -*-
"""
Extract the magnitic fiel
Created on March 23th, 2025
@author: Saroj Chand
"""


# Re-import necessary packages after code execution state reset
import numpy as np
import matplotlib.pyplot as plt

# Constants
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)
muB = 9.2740100783e-24  # Bohr magneton (J/T)
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
g = 2.00231930436256  # Electron g-factor


# Correct dipolar coupling (absolute value, assuming theta = 0)
def dipolar_coupling(d_nm):
    d_m = d_nm * 1e-9
    J = (mu0 / (4 * np.pi)) * (2 * (g * muB) ** 2) / (hbar * d_m**3)  # in rad/s
    J_Hz = J / (2 * np.pi)  # Convert to Hz
    return J_Hz


# Entangling time te = π / Jzz
def entangling_time(J_Hz):
    return 1 / J_Hz  # in seconds
    # return np.pi / J_Hz  # in seconds


# Distances from 2 nm to 20 nm
distances = np.linspace(1, 10, 100)
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


# Define the dipolar coupling strength in Hz (can vary this)
Jzz_values_kHz = [100, 250, 400]  # in kHz for different NV pairs
Jzz_values_kHz = [100, 200, 400, 800]  # in kHz for different NV pairs
Jzz_values_Hz = [j * 1e3 for j in Jzz_values_kHz]

# Time range (in µs)
t_us = np.linspace(0, 10, 1000)  # 0 to 10 µs
t_s = t_us * 1e-6  # convert to seconds

# Bell state oscillation fidelity: F(t) = sin^2(Jzz * t)
plt.figure(figsize=(8, 5))

for Jzz in Jzz_values_Hz:
    fidelity = np.sin(np.pi * Jzz * t_s) ** 2  # sin^2(π Jzz t)
    label = f"Jzz = {Jzz/1e3:.0f} kHz, te = {1e6 * np.pi / Jzz:.1f} µs"
    plt.plot(t_us, fidelity, label=label)

plt.title("Entanglement Oscillations vs Time for Different NV-NV Couplings")
plt.xlabel("Time (µs)")
plt.ylabel("Fidelity with Bell State (sin²(π·Jzz·t))")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Define a few Jzz values with realistic variation (in Hz)
Jzz_values_Hz = np.array([250e3, 310e3, 310e3, 410e3, 510e3])  # 250, 270, 310 kHz

# Define time range in microseconds
t_us = np.linspace(0, 100, 10000)  # 0 to 100 µs
t_s = t_us * 1e-6  # convert to seconds

# Calculate fidelity for each Jzz
fidelities = []
for Jzz in Jzz_values_Hz:
    F = np.sin(np.pi * Jzz * t_s) ** 2
    fidelities.append(F)

fidelities = np.array(fidelities)

# Compute max fidelity difference at each time point
fidelity_spread = np.max(fidelities, axis=0) - np.min(fidelities, axis=0)

# Plot the spread in fidelity
# plt.figure(figsize=(10, 5))
# plt.plot(t_us, fidelity_spread, label="Max Fidelity Spread")
# plt.title("Fidelity Spread Among NV Pairs vs Time")
# plt.xlabel("Time (µs)")
# plt.ylabel("Max Fidelity Difference")
# plt.grid(True)
# plt.axhline(0.05, color="r", linestyle="--", label="5% Spread")
# plt.legend()
# plt.tight_layout()
# plt.show()
# Compute average fidelity across all NV pairs
average_fidelity = np.mean(fidelities, axis=0)

# Plot average fidelity and fidelity spread
plt.figure(figsize=(10, 5))
plt.plot(t_us, average_fidelity, label="Average Fidelity", color="tab:blue")
plt.plot(t_us, fidelity_spread, label="Fidelity Spread", color="tab:red")
plt.axhline(0.05, color="gray", linestyle="--", label="5% Spread Threshold")
plt.title("Average Bell Fidelity and Spread Across NV Pairs")
plt.xlabel("Time (µs)")
plt.ylabel("Fidelity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
